import os
import argparse
import json
import cv2
import numpy as np
import base64
import shutil
from openai import OpenAI
from dotenv import load_dotenv
import subprocess
import glob
import torch
from PIL import Image

load_dotenv()

def find_scene_for_timestamp(scene_info, timestamp):
    for scene in scene_info:
        if scene["start_time"] <= timestamp < scene["end_time"]:
            return scene
    return None

def find_closest_keyframe(keyframes_dict, target_timestamp):
    eligible = {t: path for t, path in keyframes_dict.items() if t <= target_timestamp}
    
    if not eligible:
        earliest_time = min(keyframes_dict.keys())
        return keyframes_dict[earliest_time], earliest_time
    
    closest_time = max(eligible.keys())
    return eligible[closest_time], closest_time


def extract_frame_at_timestamp(video_path, output_dir, timestamp, scene_start_time):
    os.makedirs(output_dir, exist_ok=True)
    
    rel_timestamp = timestamp - scene_start_time
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_idx = int(rel_timestamp * fps)
    
    if frame_idx < 0 or frame_idx >= frame_count:
        print(f"Error: Timestamp {timestamp}s (frame {frame_idx}) is outside video range")
        video.release()
        return None
    
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = video.read()
    video.release()
    
    if not ret:
        print(f"Error: Could not read frame at timestamp {timestamp}s")
        return None
    
    output_path = os.path.join(output_dir, f"exact_frame_{frame_idx:06d}.jpg")
    cv2.imwrite(output_path, frame)
    print(f"Saved frame at exact timestamp {timestamp:.2f}s (frame {frame_idx})")
    
    return output_path

def get_transcript_for_timestamp(scene, timestamp, video_id, window=2.0):
    transcript_list = scene.get("transcript", [])
    if transcript_list:
        relevant_entries = []
        for entry in transcript_list:
            start = entry.get('start', 0)
            end = entry.get('end', 0)
            if (start <= timestamp <= end) or (abs(start - timestamp) <= window) or (abs(end - timestamp) <= window):
                relevant_entries.append(entry)
        if relevant_entries:
            return "\n".join([f"[{t.get('start', 0):.2f}s - {t.get('end', 0):.2f}s]: {t.get('text', '')}" 
                              for t in relevant_entries])
    
    # No transcript available; load video title and description.
    video_info_path = os.path.join(f"videos/{video_id}", f"{video_id}.json")
    if os.path.exists(video_info_path):
        with open(video_info_path, "r") as f:
            video_details = json.load(f)
        title = video_details.get("title", "No title available")
        description = video_details.get("description", "No description available")
        return f"Title: {title}\nDescription: {description}"
    else:
        return "No transcript or video details available for this moment."

def query_frames_with_api(keyframe_path, exact_frame_path, scene, keyframe_time, exact_time, video_id, query="describe the scene"):
    if not keyframe_path or not os.path.exists(keyframe_path):
        return "Keyframe not found."
    if not exact_frame_path or not os.path.exists(exact_frame_path):
        return "Exact frame not found."
    
    with open(keyframe_path, "rb") as img_file:
        encoded_keyframe = base64.b64encode(img_file.read()).decode('utf-8')
    
    with open(exact_frame_path, "rb") as img_file:
        encoded_exact = base64.b64encode(img_file.read()).decode('utf-8')
    
    transcript = get_transcript_for_timestamp(scene, exact_time, video_id)
    
    scene_info_text = (
        f"SCENE NUMBER: {scene['scene_number']}\n"
        f"SCENE TIMESTAMP RANGE: {scene['start_time']:.2f}s - {scene['end_time']:.2f}s\n"
        f"SCENE DURATION: {scene['duration']:.2f}s\n"
        f"KEYFRAME TIMESTAMP: {keyframe_time:.2f}s\n"
        f"EXACT TIMESTAMP: {exact_time:.2f}s\n\n"
        f"TRANSCRIPT AT TIMESTAMP:\n{transcript}\n\n"
    )
    
    prompt = f"""VIDEO SCENE CONTEXT:
            {scene_info_text}

            USER QUERY: {query}

            IMPORTANT: You are looking at two frames from the video.
            - The first frame is a keyframe captured near timestamp {keyframe_time:.2f}s.
            - The second frame is the exact frame captured at timestamp {exact_time:.2f}s.
            Provide an information and concise description or answer of what is visible in these images in 1 sentence.
            Focus on the most important elements to help the user understand what they want to know about timestamp {exact_time:.2f}s.
            DO NOT mention frames or the timestamp. 
            """
    #print(f"PROMT: {prompt}")
    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    
    try:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_keyframe}", "detail": "high"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_exact}", "detail": "high"}}
        ]
        
        system_message = "You analyze frames extracted from a video and provide answers to user queries based on the provided context."
        
        completion = client.chat.completions.create(
            model="qwen2.5-vl-72b-instruct",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": content}
            ],
            max_tokens=200,
            temperature=0.3,
            timeout=30
        )
        
        response = completion.choices[0].message.content
        return response
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error calling API: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Find the keyframe closest to a given timestamp and the exact frame at that timestamp, then query the API.")
    parser.add_argument("video_id", help="ID of the video (e.g., '_1DDhUnyvwY')")
    parser.add_argument("timestamp", type=float, help="Timestamp in seconds to analyze")
    parser.add_argument("query", default="describe the scene",
                      help="Query to send to the API (default: 'describe the scene')")
    
    args = parser.parse_args()
    
    # Load scene info from JSON.
    scene_info_path = f"videos/{args.video_id}/{args.video_id}_scenes/scene_info.json"
    try:
        with open(scene_info_path, "r") as f:
            scene_info = json.load(f)
    except FileNotFoundError:
        print(f"Error: Scene info file not found at {scene_info_path}")
        return
    
    # Find the scene covering the timestamp.
    scene = find_scene_for_timestamp(scene_info, args.timestamp)
    if not scene:
        print(f"Error: No scene found for timestamp {args.timestamp}")
        return
    
    print(f"Found scene {scene['scene_number']} for timestamp {args.timestamp}s")
    print(f"Scene range: {scene['start_time']:.2f}s - {scene['end_time']:.2f}s (duration: {scene['duration']:.2f}s)")
    
    scene_path = scene.get("scene_path", "")
    if not os.path.exists(scene_path):
        print(f"Error: Scene video not found at {scene_path}")
        return
    
    # Load keyframes info from JSON (assumed always present).
    keyframes_json_path = f"videos/{args.video_id}/keyframes/keyframes_info.json"
    try:
        with open(keyframes_json_path, "r") as f:
            keyframes_info = json.load(f)
    except FileNotFoundError:
        print(f"Error: Keyframes JSON file not found at {keyframes_json_path}")
        return

    # Convert list to dictionary: timestamp -> image_path.
    keyframes_dict = { entry["timestamp"]: entry["image_path"] for entry in keyframes_info }
    print(f"Loaded {len(keyframes_dict)} keyframes from JSON.")
    
    # Extract the exact frame at the requested timestamp.
    exact_frame_dir = f"videos/{args.video_id}/exact_frames"
    print(f"\nExtracting frame at exact timestamp {args.timestamp}s...")
    exact_frame_path = extract_frame_at_timestamp(scene_path, exact_frame_dir, args.timestamp, scene["start_time"])
    
    if keyframes_dict and exact_frame_path:
        keyframe_path, keyframe_time = find_closest_keyframe(keyframes_dict, args.timestamp)
        
        if keyframe_path:
            print(f"\nKeyframe closest to {args.timestamp}s is at {keyframe_time:.2f}s:")
            print(f"  Path: {keyframe_path}")
            
            print(f"\nQuerying API with keyframe (at {keyframe_time:.2f}s) and exact frame (at {args.timestamp:.2f}s)...")
            response = query_frames_with_api(
                keyframe_path,
                exact_frame_path,
                scene,
                keyframe_time,
                args.timestamp,
                args.video_id,
                query=args.query
            )
            
            output_file = f"videos/{args.video_id}/{args.video_id}_{int(args.timestamp)}s.txt"
            with open(output_file, "w") as f:
                f.write(response)
            print(f"\n=== COMBINED RESPONSE ===\n")
            print(response)
            print(f"\nCombined response saved to {output_file}")
        else:
            print("No closest keyframe found.")
    else:
        print("Missing required frames for API query.")
        
    if os.path.exists(exact_frame_dir):
        shutil.rmtree(exact_frame_dir)
        print(f"Cleaned up temporary folder: {exact_frame_dir}")


if __name__ == "__main__":
    main()
