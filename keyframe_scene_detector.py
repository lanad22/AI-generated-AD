import os
import argparse
import json
import subprocess
import glob
import shutil
import math
import time

import numpy as np
import torch
import clip
from PIL import Image


def get_video_info(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    fps_output = subprocess.check_output(cmd).decode().strip()
    num, den = fps_output.split('/')
    fps = float(num) / float(den)

    cmd2 = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    total_frames_str = subprocess.check_output(cmd2).decode().strip()
    try:
        total_frames = int(total_frames_str)
    except Exception as e:
        print(f"Error parsing total frames: {e}")
        total_frames = None
    return fps, total_frames


def extract_frames_ffmpeg(video_path, output_folder, sample_rate=1):
    os.makedirs(output_folder, exist_ok=True)
    command = [
        "ffmpeg", "-i", video_path,
        "-vf", f"select='not(mod(n\\,{sample_rate}))'",
        "-vsync", "vfr",
        "-q:v", "2",
        os.path.join(output_folder, "frame_%06d.jpg")
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print("Error extracting frames:", e.stderr.decode())
        return []

    frame_files = sorted(glob.glob(os.path.join(output_folder, "frame_*.jpg")))
    print(f"Extracted {len(frame_files)} frames to {output_folder}")
    return frame_files


def cosine_similarity(emb1, emb2):
    return torch.nn.functional.cosine_similarity(emb1, emb2).item()


def compute_embeddings_batched(frame_files, model, preprocess, device, batch_size=32, log_every=10):
    """Encode frames in batches for much higher CPU/GPU throughput.

    Returns a list of (1, D) tensors so downstream cosine_similarity calls
    remain unchanged.
    """
    embeddings = []
    total = len(frame_files)
    start = time.time()

    for i in range(0, total, batch_size):
        batch_files = frame_files[i:i + batch_size]
        images = []
        for f in batch_files:
            try:
                img = Image.open(f).convert("RGB")
                images.append(preprocess(img))
            except Exception as e:
                print(f"Error loading image {f}: {e}")

        if not images:
            continue

        batch = torch.stack(images).to(device)
        with torch.no_grad():
            embs = model.encode_image(batch)
            embs = embs / embs.norm(dim=-1, keepdim=True)

        # Split back into per-frame (1, D) tensors to match the rest of the pipeline.
        for j in range(embs.shape[0]):
            embeddings.append(embs[j:j + 1])

        batch_idx = i // batch_size
        if batch_idx % log_every == 0 or (i + batch_size) >= total:
            done = min(i + batch_size, total)
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(f"  Embedded {done}/{total} frames "
                  f"({rate:.1f} frames/s, ETA {eta:.0f}s)")

    return embeddings


def detect_keyframes_and_scene_boundaries(embeddings, keyframe_threshold, scene_boundary_threshold):
    if not embeddings:
        return [], []

    keyframes = [0]  # Always mark the first frame as candidate keyframe
    scene_boundaries = [0]  # And as a scene boundary

    for i in range(1, len(embeddings)):
        sim_prev = cosine_similarity(embeddings[i], embeddings[i - 1])
        if sim_prev < keyframe_threshold:
            keyframes.append(i)
        if sim_prev < scene_boundary_threshold:
            scene_boundaries.append(i)

    return keyframes, scene_boundaries


def segment_video_indices(scene_boundaries, total_frames):
    """Build (start_frame, end_frame) tuples in RAW video frame coordinates.

    Note: callers must pass scene_boundaries already converted from sampled
    indices to raw frame indices via multiply-by-sample_rate.
    """
    segments = []
    for i in range(len(scene_boundaries) - 1):
        start = scene_boundaries[i]
        end = scene_boundaries[i + 1] - 1
        segments.append((start, end))
    if scene_boundaries and scene_boundaries[-1] < total_frames:
        segments.append((scene_boundaries[-1], total_frames - 1))
    print(f"Segmented video into {len(segments)} scenes based on detected boundaries")
    return segments


def extract_video_segment_ffmpeg(video_path, start_time, end_time, output_path):
    duration = end_time - start_time
    command = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(duration),
        "-c", "copy",
        output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Extracted scene segment: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting segment: {e.stderr.decode()}")


def adjust_scene_segments_target_duration(segments, fps, target_duration):
    """Merge consecutive short scenes until each is at least target_duration."""
    merged_segments = []
    i = 0
    while i < len(segments):
        start, end = segments[i]
        seg_duration = (end - start + 1) / fps
        while seg_duration < target_duration and i < len(segments) - 1:
            i += 1
            next_start, next_end = segments[i]
            end = next_end
            seg_duration = (end - start + 1) / fps
        merged_segments.append((start, end))
        i += 1
    return merged_segments


def split_oversized_scenes(segments, fps, max_duration):
    """Safety net: split any scene longer than max_duration into evenly-sized
    chunks of ~max_duration each. Handles cases where the boundary detector
    misses cuts in long visually-stable stretches (talking heads, B-roll, etc.).
    Splits are arbitrary timing-wise, but the captioner handles shorter chunks
    much more reliably than one long chunk."""
    result = []
    for start, end in segments:
        duration = (end - start + 1) / fps
        if duration <= max_duration:
            result.append((start, end))
            continue

        num_pieces = math.ceil(duration / max_duration)
        frames_per_piece = (end - start + 1) // num_pieces
        piece_duration = duration / num_pieces
        print(f"Splitting oversized scene ({duration:.1f}s) into {num_pieces} pieces of ~{piece_duration:.1f}s each.")

        for i in range(num_pieces):
            piece_start = start + i * frames_per_piece
            piece_end = (start + (i + 1) * frames_per_piece - 1) if i < num_pieces - 1 else end
            result.append((piece_start, piece_end))

    return result


def raw_frame_to_sampled_idx(raw_frame, sample_rate, max_idx):
    """Map a raw video frame index back to the embedding-list index.

    The sampled frame at sampled index k corresponds to raw frame k * sample_rate.
    """
    idx = raw_frame // sample_rate
    return min(idx, max_idx - 1) if max_idx > 0 else 0


def process_video_folder(video_folder, sample_rate, keyframe_threshold, scene_boundary_threshold,
                         merge_scenes=True, target_duration=9.0, max_duration=30.0,
                         device="cuda", batch_size=32):
    video_id = os.path.basename(os.path.normpath(video_folder))
    video_path = os.path.join(video_folder, f"{video_id}.mp4")
    fps, total_frames = get_video_info(video_path)
    print(f"Processing video: {video_path}")
    print(f"FPS: {fps:.2f}, Total frames: {total_frames}, sample_rate: {sample_rate}")

    metadata_path = os.path.join(video_folder, f"{video_id}.json")
    try:
        with open(metadata_path, 'r') as mf:
            video_metadata = json.load(mf)
        print(f"Loaded video metadata")
    except FileNotFoundError:
        video_metadata = {}

    # Create a temporary folder for frame extraction.
    temp_folder = os.path.join(video_folder, "frames_temp")
    frame_files = extract_frames_ffmpeg(video_path, temp_folder, sample_rate=sample_rate)
    if not frame_files:
        print("No frames extracted.")
        return None

    # Load CLIP model.
    print(f"Loading CLIP ViT-B/32 on device={device}")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # Encode all frames in batches.
    print(f"Encoding {len(frame_files)} frames (batch_size={batch_size})")
    embeddings = compute_embeddings_batched(
        frame_files, model, preprocess, device, batch_size=batch_size
    )
    print(f"Computed embeddings for {len(embeddings)} frames.")

    # Detect candidate keyframes and scene boundaries.
    # Indices here are SAMPLED indices (positions in the embeddings list).
    keyframes_sampled, scene_boundaries_sampled = detect_keyframes_and_scene_boundaries(
        embeddings, keyframe_threshold, scene_boundary_threshold
    )
    print(f"Detected {len(keyframes_sampled)} candidate keyframes "
          f"and {len(scene_boundaries_sampled)} scene boundaries.")

    # Convert sampled indices to RAW video frame indices for downstream use.
    keyframes_raw = [s * sample_rate for s in keyframes_sampled]
    scene_boundaries_raw = [s * sample_rate for s in scene_boundaries_sampled]

    # Save candidate keyframe and scene boundary images.
    keyframes_dir = os.path.join(video_folder, "keyframes")
    os.makedirs(keyframes_dir, exist_ok=True)
    keyframe_info = []
    for sampled_idx, raw_idx in zip(keyframes_sampled, keyframes_raw):
        src = frame_files[sampled_idx]
        dst = os.path.join(keyframes_dir, f"keyframe_{raw_idx:06d}.jpg")
        shutil.copy2(src, dst)
        timestamp = raw_idx / fps
        keyframe_info.append({
            "frame_index": raw_idx,
            "timestamp": round(timestamp, 2),
            "image_path": os.path.join(video_folder, "keyframes", f"keyframe_{raw_idx:06d}.jpg")
        })
    keyframes_json_path = os.path.join(keyframes_dir, "keyframe_info.json")
    with open(keyframes_json_path, "w") as f:
        json.dump(keyframe_info, f, indent=2)
    print(f"Candidate keyframe info saved to: {keyframes_json_path}")

    scene_dir = os.path.join(video_folder, "scene_boundaries")
    os.makedirs(scene_dir, exist_ok=True)
    for sampled_idx, raw_idx in zip(scene_boundaries_sampled, scene_boundaries_raw):
        src = frame_files[sampled_idx]
        dst = os.path.join(scene_dir, f"scene_boundary_{raw_idx:06d}.jpg")
        shutil.copy2(src, dst)
    print(f"Scene boundary images saved to: {scene_dir}")

    # Remove temporary folder.
    shutil.rmtree(temp_folder)

    # Natural segmentation from detected boundaries (in raw frame indices).
    segments = segment_video_indices(scene_boundaries_raw, total_frames)
    print(f"Natural segmentation → {len(segments)} scenes")

    # Step 1: Optional — merge short scenes up to target_duration.
    if merge_scenes:
        print(f"Merging short scenes to ~{target_duration}s each")
        segments = adjust_scene_segments_target_duration(segments, fps, target_duration)
        print(f"After merging → {len(segments)} scenes")

    # Step 2: Always — split scenes that exceed max_duration (safety net).
    segments = split_oversized_scenes(segments, fps, max_duration)
    print(f"After oversized-scene split → {len(segments)} scenes")

    # Step 3: Tiny-scene cleanup. Anything still under 2s gets merged with its
    # most-similar neighbor. Segments are now in RAW frame indices, so we map
    # back to sampled indices via raw_frame_to_sampled_idx() to look up embeddings.
    n_embeddings = len(embeddings)

    def embs_for_segment(seg_start, seg_end):
        """Collect embedding tensors covering the raw-frame range [seg_start, seg_end]."""
        out = []
        # Step by sample_rate so we don't dedup-collect the same embedding many times.
        for f in range(seg_start, seg_end + 1, max(1, sample_rate)):
            idx = raw_frame_to_sampled_idx(f, sample_rate, n_embeddings)
            if idx < n_embeddings:
                out.append(embeddings[idx])
        return out

    i = 0
    while i < len(segments):
        start, end = segments[i]
        seg_duration = (end - start + 1) / fps

        if seg_duration < 2.0:
            current_segment_embs = embs_for_segment(start, end)

            if not current_segment_embs:
                if i == len(segments) - 1 and i > 0:
                    prev_start, _ = segments[i - 1]
                    segments[i - 1] = (prev_start, end)
                    segments.pop(i)
                    print(f"Merged final tiny segment ({seg_duration:.2f}s) into previous scene.")
                    break
                elif i < len(segments) - 1:
                    next_start, next_end = segments[i + 1]
                    segments[i] = (start, next_end)
                    segments.pop(i + 1)
                    print(f"Default-merged tiny scene ({seg_duration:.2f}s) with next scene.")
                    continue
                else:
                    # Only one segment and it's tiny; nothing to merge with.
                    i += 1
                    continue

            current_avg_emb = torch.mean(torch.stack(current_segment_embs), dim=0)

            prev_similarity = -1.0
            if i > 0:
                prev_embs = embs_for_segment(segments[i - 1][0], segments[i - 1][1])
                if prev_embs:
                    prev_avg = torch.mean(torch.stack(prev_embs), dim=0)
                    prev_similarity = cosine_similarity(current_avg_emb, prev_avg)

            if i == len(segments) - 1 and i > 0:
                prev_start, _ = segments[i - 1]
                segments[i - 1] = (prev_start, end)
                segments.pop(i)
                print(f"Merged final tiny segment ({seg_duration:.2f}s) into previous scene.")
                break

            next_similarity = -1.0
            if i < len(segments) - 1:
                next_embs = embs_for_segment(segments[i + 1][0], segments[i + 1][1])
                if next_embs:
                    next_avg = torch.mean(torch.stack(next_embs), dim=0)
                    next_similarity = cosine_similarity(current_avg_emb, next_avg)

            if prev_similarity > next_similarity and i > 0:
                prev_start, _ = segments[i - 1]
                segments[i - 1] = (prev_start, end)
                segments.pop(i)
                print(f"Merged scene {i + 1} ({seg_duration:.2f}s) with previous "
                      f"(sim {prev_similarity:.3f}).")
                i -= 1
            elif i < len(segments) - 1:
                next_start, next_end = segments[i + 1]
                segments[i] = (start, next_end)
                segments.pop(i + 1)
                print(f"Merged scene {i + 1} ({seg_duration:.2f}s) with next "
                      f"(sim {next_similarity:.3f}).")
            else:
                # No valid neighbor; keep as-is.
                i += 1
        else:
            i += 1

    print(f"Final scene segmentation: {len(segments)} segments")

    # Create a folder to store scene segments.
    scenes_dir = os.path.join(video_folder, f"{video_id}_scenes")
    os.makedirs(scenes_dir, exist_ok=True)

    # Extract each scene using ffmpeg.
    scene_info = []
    for i, (start_frame, end_frame) in enumerate(segments):
        start_time = start_frame / fps
        end_time = (end_frame / fps) if i < len(segments) - 1 else (total_frames / fps)
        duration = end_time - start_time
        scene_filename = f"scene_{i + 1:03d}.mp4"
        scene_path = os.path.join(scenes_dir, scene_filename)

        print(f"\nScene {i + 1}: frames {start_frame} to {end_frame}, "
              f"time {start_time:.2f}s to {end_time:.2f}s (duration: {duration:.2f}s)")
        extract_video_segment_ffmpeg(video_path, start_time, end_time, scene_path)

        scene_dict = {
            "scene_number": i + 1,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "scene_path": scene_path
        }
        scene_info.append(scene_dict)

    # Save scene segmentation info to JSON.
    scenes_json_path = os.path.join(scenes_dir, "scene_info.json")
    with open(scenes_json_path, "w") as jf:
        json.dump(scene_info, jf, indent=2)
    print(f"\nScene processing complete! JSON info saved to: {scenes_json_path}")

    return keyframes_raw, scene_boundaries_raw, fps, total_frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Threshold Keyframe and Scene Boundary Detection with Video Segmentation using CLIP.")
    parser.add_argument("video_folder", type=str, help="Path to the video folder (e.g., videos/video_id)")
    parser.add_argument("--sample_rate", type=int, default=15,
                        help="Extract every nth frame (default: 15 — gives ~2fps on 30fps video, "
                             "~4fps on 60fps video; dramatically faster than 1).")
    parser.add_argument("--keyframe_threshold", type=float, default=0.95,
                        help="Cosine similarity threshold for candidate keyframes (default: 0.95)")
    parser.add_argument("--scene_boundary_threshold", type=float, default=0.88,
                        help="Cosine similarity threshold for scene boundaries (default: 0.88)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run CLIP (default: cuda)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="CLIP encoding batch size (default: 32). Lower if RAM/VRAM constrained.")
    parser.add_argument("--merge_scenes", action="store_true", default=False,
                        help="Enable scene merging to target duration (default: off)")
    parser.add_argument("--target_duration", type=float, default=9.0,
                        help="Merge short scenes up to this size in seconds (default: 9.0)")
    parser.add_argument("--max_duration", type=float, default=30.0,
                        help="Force-split scenes longer than this many seconds. Acts as a safety net "
                             "when boundary detection misses cuts in long static content. (default: 30.0)")
    args = parser.parse_args()

    result = process_video_folder(
        args.video_folder,
        args.sample_rate,
        args.keyframe_threshold,
        args.scene_boundary_threshold,
        merge_scenes=args.merge_scenes,
        target_duration=args.target_duration,
        max_duration=args.max_duration,
        device=args.device,
        batch_size=args.batch_size,
    )