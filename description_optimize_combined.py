import json
import tempfile
import subprocess
import os
import argparse
from gtts import gTTS
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def get_tts_duration(text):
    if not text or text.isspace():
        return 0.0
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=True) as temp_file:
        tts = gTTS(text=text, lang='en')
        tts.save(temp_file.name)
        cmd = (f'ffprobe -v error -select_streams a:0 -show_entries format=duration '
               f'-of csv="p=0" "{temp_file.name}"')
        duration = float(subprocess.check_output(cmd, shell=True).decode().strip())
        return duration

def process_scene(scene, client):
    scene_number = scene.get('scene_number', 'unknown')
    scene_start = scene.get("start_time", 0.0)
    scene_end = scene.get("end_time", 0.0)
    available_scene_time = scene_end - scene_start
    
    # Sort clips by start_time.
    audio_clips = scene.get("audio_clips", [])
    text_clips = sorted([clip for clip in audio_clips if clip.get("type") == "Text on Screen"],
                        key=lambda x: x.get("start_time", 0))
    visual_clips = sorted([clip for clip in audio_clips if clip.get("type") == "Visual"],
                          key=lambda x: x.get("start_time", 0))
    
    # Combine text and visual elements and always set type to "Visual"
    text_elements = [clip['text'].strip() for clip in text_clips]
    visual_elements = [clip['text'].strip() for clip in visual_clips]
    
    # Build a prompt for the VLM to merge elements.
    prompt = "TASK: Combine the text on screen and visual elements into ONE coherent description of the scene, " \
             "then SHORTEN it so it FITS within the time limit. " \
             "\n\nVERY IMPORTANT: ALL MEASUREMENTS MUST BE PRESERVED EXACTLY and INCLUDED in the final description. " \
             f"\n\nThe final description MUST fit within {available_scene_time:.2f} seconds of TTS duration - this is a HARD LIMIT." \
             "\n\nBe extremely brief.\n\n"
    
    if text_elements:
        text_list = ", ".join(text_elements)
        prompt += f"Text on Screen: \"{text_list}\"\n"
    
    if visual_elements:
        visual_list = ", ".join(visual_elements)
        prompt += f"Visual Elements: \"{visual_list}\"\n"
    
    prompt += "\nResult:"
    
    try:
        response = client.chat.completions.create(
            model="qwen2.5-72b-instruct",
            messages=[
                {"role": "system", "content": "You are a professional audio describer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150,
        )
        merged_text = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error merging scene {scene_number} using VLM: {e}")
        
    duration = get_tts_duration(merged_text)
    print(f"Scene {scene_number} - Initial duration: {duration:.2f}s, Available time: {available_scene_time:.2f}s")
    
    retry_count = 0
    while duration > available_scene_time and retry_count < 5:
        retry_count += 1
        print(f"Scene {scene_number} - Shortening attempt #{retry_count}: Current duration {duration:.2f}s, Target: {available_scene_time:.2f}s")
        
        shorten_prompt = f"""You are optimizing a visual description for a video.
                            PREVIOUS ATTEMPT: "{merged_text}"
                            
                            ISSUE: This description takes {duration:.2f} seconds to speak, but you only have {available_scene_time:.2f} seconds available.
                            You MUST reduce it by at least {(duration - available_scene_time):.2f} seconds.
                            
                            TASK (Attempt #{retry_count} - URGENT):
                            Create an EXTREMELY SHORT DESCRIPTION that MUST fit within {available_scene_time:.2f} seconds.
                            
                            VERY IMPORTANT: 
                            - ALL MEASUREMENTS MUST BE PRESERVED EXACTLY as written and INCLUDED in the final description
                            - Do not modify any numbers, units, or measurements in any way
                            - Remove ALL details that aren't absolutely necessary
                            - Use as few syllables as possible.
                            
                            OUTPUT FORMAT:
                            Provide only the shortened description, nothing else."""
        try:
            response = client.chat.completions.create(
                model="qwen2.5-72b-instruct",
                messages=[
                    {"role": "system", "content": "You are an expert at creating concise descriptions."},
                    {"role": "user", "content": shorten_prompt}
                ],
                temperature=0.7,
                max_tokens=100,
            )
            merged_text = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error shortening scene {scene_number} description: {e}")
            break
        
        previous_duration = duration
        duration = get_tts_duration(merged_text)
        print(f"Scene {scene_number} - After attempt #{retry_count}: New duration: {duration:.2f}s (reduced by {previous_duration - duration:.2f}s)")
    
    if duration > available_scene_time:
        print(f"\nWARNING: Final description for Scene {scene_number} still exceeds available time ({duration:.2f}s > {available_scene_time:.2f}s)")
    else:
        print(f"\nSUCCESS: Scene {scene_number} description fits within available time ({duration:.2f}s <= {available_scene_time:.2f}s)")
    
    return {
        'scene_number': scene_number,
        'start_time': scene_start,
        'type': "Visual",  
        'text': merged_text,
        'tts_duration': duration
    }

def process_scenes(scenes, client):
    """
    Process every scene by generating a merged description.
    Skip scenes that have no clips.
    """
    processed = []
    for scene in scenes:
        result = process_scene(scene, client)
        if result is not None:  # Only add the scene if it has content
            processed.append(result)
    return processed

def main():
    parser = argparse.ArgumentParser(
        description="Merge and optimize scene descriptions by combining Text on Screen and Visual clips. " \
                    "Ensure the final description does not exceed the scene's total duration. " \
                    "The output JSON will include each scene's type and start_time."
    )
    parser.add_argument("video_folder", help="Path to the video folder (must include scene_info.json)")
    parser.add_argument("--input_file", default="scene_info_deduped.json", help="Input JSON file (default: scene_info_deduped.json)")
    parser.add_argument("--output_file", default="audio_clips_optimized.json", help="Output JSON file (default: audio_clips_optimized.json)")
    
    args = parser.parse_args()

    video_id = os.path.basename(os.path.normpath(args.video_folder))
    scenes_folder = os.path.join(args.video_folder, f"{video_id}_scenes")
    scenes_path = os.path.join(scenes_folder, "scene_info_deduped.json")
    
    if not os.path.exists(scenes_path):
        print(f"Error: scene_info_deduped.json not found in {scenes_folder}")
        
        scenes_path = os.path.join(scenes_folder, "scene_info.json")
        if not os.path.exists(scenes_path):
            print(f"Error: scene_info.json not found in {scenes_folder}")
            return
        else:
            print(f"Found scene_info.json instead")
    
    
    with open(scenes_path, "r") as f:
        scenes = json.load(f)
    
    print("\nSetting up OpenAI client for VLM...")
    api_key = os.getenv("API_KEY")
    base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    if not api_key:
        print("Error: API_KEY environment variable not set")
        return
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # Process all scenes.
    processed_clips = process_scenes(scenes, client)

    print("\n===== PROCESSED SCENE DESCRIPTIONS =====")
    for clip in processed_clips:
        print(f"Scene: {clip.get('scene_number', 'unknown')}, Start Time: {clip.get('start_time', 'N/A')}, Type: {clip.get('type')}")
        print(f"Description: {clip.get('text')}")
        print(f"TTS Duration: {clip.get('tts_duration'):.2f}s\n")
    
    output_file = os.path.join(scenes_folder, args.output_file)
    with open(output_file, "w") as f:
        json.dump(processed_clips, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()