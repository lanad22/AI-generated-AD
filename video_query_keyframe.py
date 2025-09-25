import os
import argparse
import json
import cv2
import shutil
# Removed base64 and openai
from dotenv import load_dotenv
# Added google.generativeai and Pillow
import google.generativeai as genai
from PIL import Image
import subprocess

# Load environment variables from a .env file
load_dotenv()

def find_scene_for_timestamp(scene_info, timestamp):
    """Finds the scene dictionary corresponding to a given timestamp."""
    for scene in scene_info:
        if scene["start_time"] <= timestamp < scene["end_time"]:
            return scene
    return None

def find_closest_keyframe(keyframes_dict, target_timestamp):
    """Finds the keyframe with the timestamp closest to (and not after) the target timestamp."""
    eligible = {t: path for t, path in keyframes_dict.items() if t <= target_timestamp}
    if not eligible:
        earliest_time = min(keyframes_dict.keys())
        return keyframes_dict[earliest_time], earliest_time
    closest_time = max(eligible.keys())
    return eligible[closest_time], closest_time

def extract_frame_at_timestamp(video_path, output_dir, timestamp, scene_start_time):
    """Extracts a single frame from a video at a precise timestamp. Falls back to ffmpeg if cv2 fails."""
    os.makedirs(output_dir, exist_ok=True)
    rel_timestamp = timestamp - scene_start_time
    
    # Try cv2 first
    try:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print(f"CV2: Could not open video file {video_path}, trying ffmpeg...")
            video.release()
            return extract_frame_with_ffmpeg(video_path, output_dir, timestamp, scene_start_time)
        
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = int(rel_timestamp * fps)
        
        if frame_idx < 0 or frame_idx >= frame_count:
            print(f"CV2: Timestamp {timestamp}s (frame {frame_idx}) is outside video range, trying ffmpeg...")
            video.release()
            return extract_frame_with_ffmpeg(video_path, output_dir, timestamp, scene_start_time)
        
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = video.read()
        video.release()
        
        if not ret:
            print(f"CV2: Could not read frame at timestamp {timestamp}s, trying ffmpeg...")
            return extract_frame_with_ffmpeg(video_path, output_dir, timestamp, scene_start_time)
        
        output_path = os.path.join(output_dir, f"exact_frame_{frame_idx:06d}.jpg")
        cv2.imwrite(output_path, frame)
        print(f"CV2: Saved frame at exact timestamp {timestamp:.2f}s (frame {frame_idx})")
        return output_path
        
    except Exception as e:
        print(f"CV2: Exception occurred: {e}, trying ffmpeg...")
        return extract_frame_with_ffmpeg(video_path, output_dir, timestamp, scene_start_time)

def extract_frame_with_ffmpeg(video_path, output_dir, timestamp, scene_start_time):
    """Fallback frame extraction using ffmpeg."""
    
    try:
        rel_timestamp = max(0.0, timestamp - scene_start_time)
        output_path = os.path.join(output_dir, f"exact_frame_ffmpeg_{int(rel_timestamp*1000):06d}.jpg")
        
        # Use ffmpeg to extract frame
        cmd = [
            'ffmpeg', '-y', '-v', 'quiet',
            '-ss', str(rel_timestamp),
            '-i', str(video_path),
            '-frames:v', '1',
            '-q:v', '2',
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and os.path.exists(output_path):
            print(f"FFMPEG: Saved frame at exact timestamp {timestamp:.2f}s")
            return output_path
        else:
            print(f"FFMPEG: Failed to extract frame: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"FFMPEG: Exception occurred: {e}")
        return None


def get_transcript_for_timestamp(scene, timestamp, video_id, window=2.0):
    """Retrieves transcript text around a specific timestamp."""
    transcript_list = scene.get("transcript", [])
    if transcript_list:
        relevant_entries = []
        for entry in transcript_list:
            start, end = entry.get('start', 0), entry.get('end', 0)
            if (start <= timestamp <= end) or (abs(start - timestamp) <= window) or (abs(end - timestamp) <= window):
                relevant_entries.append(entry)
        if relevant_entries:
            return "\n".join([f"[{t.get('start', 0):.2f}s - {t.get('end', 0):.2f}s]: {t.get('text', '')}" for t in relevant_entries])
    video_info_path = os.path.join(f"videos/{video_id}", f"{video_id}.json")
    if os.path.exists(video_info_path):
        with open(video_info_path, "r") as f:
            video_details = json.load(f)
        title = video_details.get("title", "No title available")
        description = video_details.get("description", "No description available")
        return f"Title: {title}\nDescription: {description}"
    return "No transcript or video details available for this moment."

def get_accumulated_audio_clips(scene_info, current_scene_idx):
    """Gathers all audio clips from the beginning of the video up to the current scene."""
    accumulated_clips = []
    for i in range(current_scene_idx + 1):
        scene = scene_info[i]
        if "audio_clips" in scene:
            for clip in scene["audio_clips"]:
                accumulated_clips.append({"scene_number": scene["scene_number"], "start_time": clip.get("start_time", 0), "end_time": clip.get("end_time", 0), "text": clip.get("text", ""), "speakers": clip.get("speakers", [])})
    return accumulated_clips

def format_audio_clips(clips):
    """Formats a list of audio clip dictionaries into a readable string."""
    if not clips:
        return "No audio clips available."
    formatted = []
    for clip in clips:
        speakers = ", ".join(clip["speakers"]) if clip["speakers"] else "Unknown"
        formatted.append(f"[Scene {clip['scene_number']} | {clip['start_time']:.2f}s - {clip['end_time']:.2f}s | Speakers: {speakers}]: {clip['text']}")
    return "\n".join(formatted)

def query_frames_with_api(keyframe_path, exact_frame_path, scene, scene_info, scene_idx, keyframe_time, exact_time, video_id, query="describe the scene"):
    """Encodes frames, constructs a prompt with context, and queries the Google Gemini API."""
    if not keyframe_path or not os.path.exists(keyframe_path):
        return "Keyframe not found."
    if not exact_frame_path or not os.path.exists(exact_frame_path):
        return "Exact frame not found."

    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    except Exception as e:
        return f"Error configuring Gemini API: {e}. Make sure GOOGLE_API_KEY is set."

    transcript = get_transcript_for_timestamp(scene, exact_time, video_id)
    accumulated_audio_clips = get_accumulated_audio_clips(scene_info, scene_idx)
    formatted_audio_clips = format_audio_clips(accumulated_audio_clips)
    
    scene_info_text = (
        f"SCENE NUMBER: {scene['scene_number']}\n"
        f"SCENE TIMESTAMP RANGE: {scene['start_time']:.2f}s - {scene['end_time']:.2f}s\n"
        f"SCENE DURATION: {scene['duration']:.2f}s\n"
        f"KEYFRAME TIMESTAMP: {keyframe_time:.2f}s\n"
        f"EXACT TIMESTAMP: {exact_time:.2f}s\n\n"
        f"TRANSCRIPT AT TIMESTAMP:\n{transcript}\n\n"
        f"ACCUMULATED AUDIO CLIPS (CURRENT AND PREVIOUS SCENES):\n{formatted_audio_clips}\n\n"
    )
    
    if query.lower().strip() == "describe the scene":
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
    else:
        prompt = f"""VIDEO SCENE CONTEXT:
            {scene_info_text}

            USER QUERY: {query}

            IMPORTANT: You are looking at two frames from the video.
            - The first frame is a keyframe captured near timestamp {keyframe_time:.2f}s.
            - The second frame is the exact frame captured at timestamp {exact_time:.2f}s. 
            Use context to provide a contextually rich answer to the {query} in one very concise sentence.
            When the query is a "Why" question, your answer must include the specific cause-and-effect details 
            (identifying who was involved, when the event occurred, and what happened) that directly answer the query.
            If the question is Where, be specific with the setting.
            Do NOT add external details outside of scope.
            Do NOT refer to frame numbers or timestamps.
            """
    
    try:
        keyframe_img = Image.open(keyframe_path)
        exact_frame_img = Image.open(exact_frame_path)

        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        model.system_instruction = "You analyze frames extracted from a video and provide answers to user queries based on the provided context."

        response = model.generate_content(
            [prompt, keyframe_img, exact_frame_img]
        )

        return response.text
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error calling API: {str(e)}"

def main():
    """Main function to parse arguments and run the analysis pipeline."""
    parser = argparse.ArgumentParser(description="Find the keyframe closest to a given timestamp and the exact frame at that timestamp, then query the API.")
    parser.add_argument("video_id", help="ID of the video (e.g., '_1DDhUnyvwY')")
    parser.add_argument("timestamp", type=float, help="Timestamp in seconds to analyze")
    parser.add_argument("query", nargs="?", default="describe the scene",
                      help="Query to send to the API (default: 'describe the scene')")
    
    args = parser.parse_args()
    
    scene_info_path = f"videos/{args.video_id}/{args.video_id}_scenes/scene_info.json"
    try:
        with open(scene_info_path, "r") as f:
            scene_info = json.load(f)
    except FileNotFoundError:
        print(f"Error: Scene info file not found at {scene_info_path}")
        return
    
    scene, scene_idx = None, -1
    for idx, s in enumerate(scene_info):
        if s["start_time"] <= args.timestamp < s["end_time"]:
            scene, scene_idx = s, idx
            break
            
    if not scene:
        print(f"Error: No scene found for timestamp {args.timestamp}")
        return
    
    print(f"Found scene {scene['scene_number']} for timestamp {args.timestamp}s")
    print(f"Scene range: {scene['start_time']:.2f}s - {scene['end_time']:.2f}s (duration: {scene['duration']:.2f}s)")
    
    scene_path = scene.get("scene_path", "")
    if not os.path.exists(scene_path):
        print(f"Error: Scene video not found at {scene_path}")
        return
    
    keyframes_json_path = f"videos/{args.video_id}/keyframes/keyframe_info.json"
    try:
        with open(keyframes_json_path, "r") as f:
            keyframes_info = json.load(f)
    except FileNotFoundError:
        print(f"Error: Keyframes JSON file not found at {keyframes_json_path}")
        return

    keyframes_dict = {entry["timestamp"]: entry["image_path"] for entry in keyframes_info}
    print(f"Loaded {len(keyframes_dict)} keyframes from JSON.")
    
    exact_frame_dir = f"videos/{args.video_id}/exact_frames"
    print(f"\nExtracting frame at exact timestamp {args.timestamp}s...")
    exact_frame_path = extract_frame_at_timestamp(scene_path, exact_frame_dir, args.timestamp, scene["start_time"])
    
    if keyframes_dict and exact_frame_path:
        keyframe_path, keyframe_time = find_closest_keyframe(keyframes_dict, args.timestamp)
        if keyframe_path:
            print(f"\nKeyframe closest to {args.timestamp}s is at {keyframe_time:.2f}s:")
            print(f"  Path: {keyframe_path}")
            
            print(f"\nQuerying API with keyframe (at {keyframe_time:.2f}s) and exact frame (at {args.timestamp:.2f}s)...")
            print(f"Also including accumulated audio clips from scene 1 to scene {scene['scene_number']}...")
            
            response = query_frames_with_api(keyframe_path, exact_frame_path, scene, scene_info, scene_idx, keyframe_time, args.timestamp, args.video_id, query=args.query)
            
            output_file = f"videos/{args.video_id}/{args.video_id}_{int(args.timestamp)}s.txt"
            with open(output_file, "w") as f:
                f.write(response)
            print(f"\n=== API RESPONSE ===\n")
            print(response)
            print(f"\nResponse saved to {output_file}")
        else:
            print("No closest keyframe found.")
    else:
        print("Missing required frames for API query.")
        
    if os.path.exists(exact_frame_dir):
        shutil.rmtree(exact_frame_dir)
        print(f"Cleaned up temporary folder: {exact_frame_dir}")

if __name__ == "__main__":
    main()