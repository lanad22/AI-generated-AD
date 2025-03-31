import os
import argparse
import json
import subprocess
import glob
import shutil

import numpy as np
import torch
import clip
from PIL import Image

def get_video_info(video_path):
    """
    Use ffprobe to get the video's FPS and total frame count.
    """
    # Get FPS
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
    
    # Get total frame count
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
    # Build the ffmpeg command. The filter selects every nth frame.
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
        print(f"Error extracting frames: {e.stderr.decode()}")
        return []
    
    frame_files = sorted(glob.glob(os.path.join(output_folder, "frame_*.jpg")))
    print(f"Extracted {len(frame_files)} frames to {output_folder}")
    return frame_files

def extract_keyframes_clip(video_path, device="cuda", sample_rate=1, similarity_threshold=0.85, save_keyframes=True):
    print(f"Extracting keyframes using CLIP from {video_path}")
    fps, total_frames = get_video_info(video_path)
    print(f"Video FPS: {fps:.2f}, Total frames: {total_frames}")
    
    # Create a temporary folder for extracted frames.
    temp_frames_folder = os.path.join(os.path.dirname(video_path), "extracted_frames_temp")
    frame_files = extract_frames_ffmpeg(video_path, temp_frames_folder, sample_rate=sample_rate)
    if not frame_files:
        print("No frames extracted. Exiting keyframe extraction.")
        return [], total_frames, []
    
    # Create a folder for keyframes if saving is enabled.
    if save_keyframes:
        keyframes_folder = os.path.join(os.path.dirname(video_path), "keyframes")
        os.makedirs(keyframes_folder, exist_ok=True)
    else:
        keyframes_folder = None

    # Load CLIP model and preprocessing pipeline.
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    keyframe_indices = []
    keyframes_info = []
    prev_embedding = None

    # Iterate over the extracted frames.
    for i, frame_file in enumerate(frame_files):
        try:
            image = Image.open(frame_file).convert("RGB")
        except Exception as e:
            print(f"Error opening frame {frame_file}: {e}")
            continue
        
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image_input)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        # Map the extracted frame to the original video frame index.
        # Since frames are extracted every 'sample_rate' frames, original_index = i * sample_rate.
        original_index = i * sample_rate
        
        if prev_embedding is not None:
            cosine_sim = torch.nn.functional.cosine_similarity(embedding, prev_embedding).item()
            if cosine_sim < similarity_threshold:
                keyframe_indices.append(original_index)
                timestamp = original_index / fps
                if save_keyframes:
                    dest_file = os.path.join(keyframes_folder, f"keyframe_{original_index:06d}.jpg")
                    shutil.copy2(frame_file, dest_file)
                else:
                    dest_file = None
                keyframes_info.append({
                    "frame_index": original_index,
                    "timestamp": timestamp,
                    "image_path": dest_file
                })
        else:
            # Always treat the first frame as a keyframe.
            keyframe_indices.append(original_index)
            timestamp = original_index / fps
            if save_keyframes:
                dest_file = os.path.join(keyframes_folder, f"keyframe_{original_index:06d}.jpg")
                shutil.copy2(frame_file, dest_file)
            else:
                dest_file = None
            keyframes_info.append({
                "frame_index": original_index,
                "timestamp": timestamp,
                "image_path": dest_file
            })
        
        prev_embedding = embedding

    print(f"Found {len(keyframe_indices)} keyframe indices out of {total_frames} frames")
    
    # Remove the temporary frames folder.
    shutil.rmtree(temp_frames_folder)
    
    return keyframe_indices, total_frames, keyframes_info

def segment_video_indices(keyframe_indices, total_frames):
    segments = []
    for i in range(len(keyframe_indices) - 1):
        start = keyframe_indices[i]
        end = keyframe_indices[i+1] - 1  # End at the frame before the next keyframe.
        segments.append((start, end))
    if keyframe_indices and keyframe_indices[-1] < total_frames:
        segments.append((keyframe_indices[-1], total_frames - 1))
    print(f"Segmented video into {len(segments)} segments (by frame indices)")
    return segments

def merge_short_segments(segments, fps, min_duration=10.0):
    if not segments:
        return segments

    merged = []
    current_seg = segments[0]
    
    for next_seg in segments[1:]:
        current_duration = (current_seg[1] - current_seg[0] + 1) / fps
        if current_duration < min_duration:
            # Merge with the next segment.
            current_seg = (current_seg[0], next_seg[1])
        else:
            merged.append(current_seg)
            current_seg = next_seg

    # Merge the last segment if it is too short.
    current_duration = (current_seg[1] - current_seg[0] + 1) / fps
    if current_duration < min_duration and merged:
        prev_seg = merged.pop()
        current_seg = (prev_seg[0], current_seg[1])
    merged.append(current_seg)
    print(f"Merged segments into {len(merged)} segments (each at least {min_duration}s long)")
    return merged

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
        print(f"Extracted segment: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting segment: {e.stderr.decode()}")

def process_video(video_folder, target_duration, device="cuda", sample_rate=1,
                  similarity_threshold=0.85, min_duration=10.0):
    video_id = os.path.basename(os.path.normpath(video_folder))
    video_path = os.path.join(video_folder, f"{video_id}.mp4")
    scenes_dir = os.path.join(video_folder, f"{video_id}_scenes")
    keyframes_folder = os.path.join(video_folder, "keyframes")
    os.makedirs(scenes_dir, exist_ok=True)
    
    print(f"Processing video: {video_path}")
    
    # Extract keyframes using CLIP with ffmpeg-based frame extraction.
    keyframe_indices, total_frames, keyframes_info = extract_keyframes_clip(
        video_path, device=device, sample_rate=sample_rate, similarity_threshold=similarity_threshold, save_keyframes=True
    )
    if not keyframe_indices or not total_frames:
        print("Failed to extract keyframes/frames. Exiting.")
        return

    # Save keyframe info to JSON.
    keyframes_json_path = os.path.join(keyframes_folder, "keyframes_info.json")
    with open(keyframes_json_path, "w") as f:
        json.dump(keyframes_info, f, indent=2)
    print(f"Keyframes info saved to: {keyframes_json_path}")

    # Retrieve FPS and total duration.
    fps, _ = get_video_info(video_path)
    total_duration = total_frames / fps
    print(f"Video duration: {total_duration:.2f}s, FPS: {fps:.2f}")
    
    # Segment the video based on keyframe indices.
    segments = segment_video_indices(keyframe_indices, total_frames)
    
    # Merge segments that are shorter than the minimum required duration.
    segments = merge_short_segments(segments, fps, min_duration=min_duration)
    
    scene_info = []
    
    # Process and extract each scene.
    for i, (start_frame, end_frame) in enumerate(segments):
        start_time = start_frame / fps
        # For a clear cut, use the end_frame time directly.
        end_time = (end_frame / fps) if i < len(segments) - 1 else (total_frames / fps)
        duration = end_time - start_time
        scene_filename = f"scene_{i+1:03d}.mp4"
        scene_path = os.path.join(scenes_dir, scene_filename)
        
        print(f"\nScene {i+1}: frames {start_frame} to {end_frame}, time {start_time:.2f}s to {end_time:.2f}s (duration: {duration:.2f}s)")
        
        # Extract the scene (with audio) using ffmpeg.
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
    
    # Save scene information to JSON.
    json_path = os.path.join(scenes_dir, "scene_info.json")
    with open(json_path, "w") as jf:
        json.dump(scene_info, jf, indent=2)
    print(f"\nScene processing complete! JSON info saved to: {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect scenes from a video using CLIP-based keyframe segmentation with ffmpeg frame extraction, "
                    "merge scenes shorter than a minimum duration (default: 10s), extract original segments (with audio), "
                    "and save scene and keyframe information to JSON files."
    )
    parser.add_argument("video_folder", type=str,
                        help="Path to the video folder (e.g., videos/video_id). The video file must be named video_id.mp4")
    parser.add_argument("--target_duration", type=float, default=10.0,
                        help="Desired duration (in seconds) for each scene (used for grouping, default: 10s)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run processing (default: cuda)")
    parser.add_argument("--sample_rate", type=int, default=1,
                        help="Extract every nth frame for keyframe detection (default: 1, i.e., every frame)")
    parser.add_argument("--similarity_threshold", type=float, default=0.95,
                        help="Cosine similarity threshold for keyframe detection (default: 0.95)")
    parser.add_argument("--min_duration", type=float, default=10.0,
                        help="Minimum duration (in seconds) for each scene (default: 10s)")
    
    args = parser.parse_args()
    process_video(args.video_folder, args.target_duration, args.device,
                  args.sample_rate, args.similarity_threshold, args.min_duration)
