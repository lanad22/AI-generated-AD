import json
import tempfile
import subprocess
import os
import argparse
import time
from gtts import gTTS
from typing import List, Dict

# Model clients
from openai import OpenAI  
import google.generativeai as genai  
from google.generativeai.types import HarmCategory, HarmBlockThreshold  

from dotenv import load_dotenv

load_dotenv()

MODEL_QWEN = "qwen"
MODEL_GEMINI = "gemini"
MODEL_GPT = "gpt"

def get_tts_duration(text):
    if not text or text.isspace():
        return 0.0
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=True) as temp_file:
        try:
            tts = gTTS(text=text, lang='en')
            tts.save(temp_file.name)
            cmd = (f'ffprobe -v error -select_streams a:0 -show_entries format=duration '
                   f'-of csv="p=0" "{temp_file.name}"')
            duration_str = subprocess.check_output(cmd, shell=True, text=True).strip()
            if not duration_str:  
                print(f"Warning: ffprobe returned no duration for text: '{text[:50]}...'")
                return 5.0  
            duration = float(duration_str)
            return duration
        except Exception as e:
            print(f"Error in get_tts_duration for text '{text[:50]}...': {e}. Returning estimated duration.")
            words = len(text.split())
            estimated_duration = words / 2.5  
            return max(1.0, estimated_duration)  

def merge_scene_content(scene, client, model_type, max_retries=10):
    scene_number = scene.get('scene_number', 'unknown')
    scene_start = scene.get("start_time", 0.0)
    scene_end = scene.get("end_time", 0.0)
    available_scene_time = scene_end - scene_start
    

    audio_clips = scene.get("audio_clips", [])
    text_clips = sorted([clip for clip in audio_clips if clip.get("type") == "Text on Screen"],
                        key=lambda x: x.get("start_time", 0))
    visual_clips = sorted([clip for clip in audio_clips if clip.get("type") == "Visual"],
                          key=lambda x: x.get("start_time", 0))
    
    text_elements = [clip['text'].strip() for clip in text_clips]
    visual_elements = [clip['text'].strip() for clip in visual_clips]
    initial_prompt = build_merge_prompt(text_elements, visual_elements, available_scene_time, model_type)
    
    try:
        merged_text = call_model_api(client, model_type, initial_prompt, is_initial=True)
        if not merged_text:
            print(f"Error: No response from {model_type.upper()} for scene {scene_number}")
            return create_fallback_result(scene, text_elements, visual_elements)
    except Exception as e:
        print(f"Error merging scene {scene_number} using {model_type.upper()}: {e}")
        return create_fallback_result(scene, text_elements, visual_elements)
    
    duration = get_tts_duration(merged_text)
    print(f"Scene {scene_number} - Initial duration: {duration:.2f}s, Available time: {available_scene_time:.2f}s")
    
    retry_count = 0
    while duration > available_scene_time and retry_count < max_retries:
        retry_count += 1
        print(f"Scene {scene_number} - Shortening attempt #{retry_count}: Current duration {duration:.2f}s, Target: {available_scene_time:.2f}s")
        
        shorten_prompt = build_shorten_prompt(merged_text, duration, available_scene_time, retry_count, model_type)
        
        try:
            shortened_text = call_model_api(client, model_type, shorten_prompt, is_initial=False)
            if shortened_text:
                merged_text = shortened_text
            else:
                print(f"Warning: No response from {model_type.upper()} for shortening attempt {retry_count}")
                break
        except Exception as e:
            print(f"Error shortening scene {scene_number} description (attempt {retry_count}): {e}")
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

def build_merge_prompt(text_elements, visual_elements, available_time, model_type):
    prompt = ("TASK: Combine the text on screen and visual elements into ONE coherent description of the scene, "
              "then SHORTEN it so it FITS within the time limit."
              "\n\nVERY IMPORTANT: ALL MEASUREMENTS MUST BE PRESERVED EXACTLY and INCLUDED in the final description. "
              f"\n\nThe final description MUST fit within {available_time:.2f} seconds of TTS duration - this is a HARD LIMIT."
              "\n\nBe extremely brief.\n\n")
    
    if text_elements:
        text_list = ", ".join(text_elements)
        prompt += f"Text on Screen: \"{text_list}\"\n"
    
    if visual_elements:
        visual_list = ", ".join(visual_elements)
        prompt += f"Visual Elements: \"{visual_list}\"\n"
    
    prompt += "\nResult:"
    
    return prompt

def build_shorten_prompt(current_text, current_duration, target_duration, attempt_num, model_type):
    return f"""You are optimizing a visual description for a video.
            PREVIOUS ATTEMPT: "{current_text}"
            
            ISSUE: This description takes {current_duration:.2f} seconds to speak, but you only have {target_duration:.2f} seconds available.
            You MUST reduce it by at least {(current_duration - target_duration):.2f} seconds.
            
            TASK (Attempt #{attempt_num} - URGENT):
            Create an EXTREMELY SHORT DESCRIPTION that MUST fit within {target_duration:.2f} seconds.
            
            VERY IMPORTANT: 
            - ALL MEASUREMENTS MUST BE PRESERVED EXACTLY as written and INCLUDED in the final description
            - Do not modify any numbers, units, or measurements in any way
            - Remove ALL details that aren't absolutely necessary
            - Use as few syllables as possible.

            OUTPUT FORMAT:
            Provide only the shortened description, nothing else."""

def call_model_api(client, model_type, prompt, is_initial=True):
    if model_type == MODEL_QWEN:
        try:
            response = client.chat.completions.create(
                model="qwen2.5-72b-instruct",
                messages=[
                    {"role": "system", "content": "You are a professional audio describer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150 if is_initial else 100,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling Qwen API: {e}")
            time.sleep(2)  # Rate limiting backoff
            return None
    
    elif model_type == MODEL_GEMINI:
        system_context = "You are a professional audio describer. Your task is to create precise, concise visual descriptions that FITS the time constrains."
        full_prompt = f"{system_context}\n\n{prompt}"
        try:
            response = client.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=150 if is_initial else 100
                ),
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            return response.text.strip()
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            time.sleep(2)  # Rate limiting backoff
            return None
    
    elif model_type == MODEL_GPT:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional audio describer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150 if is_initial else 100,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling GPT API: {e}")
            time.sleep(2)  # Rate limiting backoff
            return None
    
    return None

def create_fallback_result(scene, text_elements, visual_elements):
    scene_number = scene.get('scene_number', 'unknown')
    scene_start = scene.get("start_time", 0.0)
    
    # Simple concatenation as fallback
    all_elements = text_elements + visual_elements
    fallback_text = " ".join(all_elements) if all_elements else "No description available"
    
    return {
        'scene_number': scene_number,
        'start_time': scene_start,
        'type': "Visual",  
        'text': fallback_text,
        'tts_duration': get_tts_duration(fallback_text)
    }

def initialize_client(model_type):
    if model_type == MODEL_QWEN:
        api_key = os.getenv("API_KEY")
        if not api_key:
            raise ValueError("API_KEY environment variable not set for Qwen")
        
        print("Setting up OpenAI client for DashScope API (Qwen)...")
        return OpenAI(
            api_key=api_key,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
    
    elif model_type == MODEL_GEMINI:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set for Gemini")
        
        print("Initializing Gemini client...")
        genai.configure(api_key=gemini_api_key)
        client = genai.GenerativeModel("gemini-1.5-pro-latest")
        
        # Test Gemini connection
        try:
            print("Testing Gemini connection...")
            test_response = client.generate_content("Say 'Hello' to test the connection.")
            if test_response and test_response.text:
                print(f"Gemini connection successful. Test response: {test_response.text.strip()}")
            else:
                print("Warning: Gemini test returned empty response")
        except Exception as e:
            print(f"Warning: Gemini connection test failed: {e}")
            print("Proceeding anyway, but API calls may fail...")
        
        return client
    
    elif model_type == MODEL_GPT:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set for GPT")
        
        print("Setting up OpenAI client for GPT...")
        client = OpenAI(api_key=openai_api_key)
        
        # Test GPT connection
        try:
            print("Testing GPT connection...")
            test_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say 'Hello' to test the connection."}],
                max_tokens=10
            )
            if test_response and test_response.choices[0].message.content:
                print(f"GPT connection successful. Test response: {test_response.choices[0].message.content.strip()}")
            else:
                print("Warning: GPT test returned empty response")
        except Exception as e:
            print(f"Warning: GPT connection test failed: {e}")
            print("Proceeding anyway, but API calls may fail...")
        
        return client
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def process_scenes(scenes, client, model_type):
    processed = []
    for scene in scenes:
        # Skip scenes that have no clips
        audio_clips = scene.get("audio_clips", [])
        if not audio_clips:
            print(f"Skipping scene {scene.get('scene_number', 'unknown')} - no audio clips")
            continue
            
        result = merge_scene_content(scene, client, model_type)
        if result is not None:
            processed.append(result)
    return processed

def main():
    parser = argparse.ArgumentParser(
        description="Merge and optimize scene descriptions by combining Text on Screen and Visual clips using AI models. "
                    "Supports Qwen, Gemini, and GPT for content optimization."
    )
    parser.add_argument("video_folder", help="Path to the video folder (must include scene_info_qwen.json, scene_info_gemini.json, or scene_info_gpt.json)")
    parser.add_argument("--output_file", help="Output JSON file name", required=True)
    parser.add_argument("--model", type=str, choices=["qwen", "gemini", "gpt"], default="gpt",
                        help="Choose the AI model for content optimization: 'qwen', 'gemini', or 'gpt' (default: qwen)")
    
    args = parser.parse_args()

    video_id = os.path.basename(os.path.normpath(args.video_folder))
    scenes_folder = os.path.join(args.video_folder, f"{video_id}_scenes")
    
    # Determine input file based on model choice
    preferred_input_filename = f"scene_info_{args.model}.json"
    preferred_scenes_path = os.path.join(scenes_folder, preferred_input_filename)
    fallback_scenes_path = os.path.join(scenes_folder, "scene_info.json")
    
    scenes_path = None
    if os.path.exists(preferred_scenes_path):
        scenes_path = preferred_scenes_path
        print(f"Using model-specific input file: {scenes_path}")
    elif os.path.exists(fallback_scenes_path):
        scenes_path = fallback_scenes_path
        print(f"Warning: '{preferred_input_filename}' not found. Using fallback: {scenes_path}")
    else:
        print(f"Error: No suitable scene_info file found in {scenes_folder}. Checked for '{preferred_input_filename}' and '{os.path.basename(fallback_scenes_path)}'.")
        return
    
    with open(scenes_path, "r") as f:
        scenes = json.load(f)
    
    print(f"\nInitializing {args.model.upper()} client...")
    try:
        client = initialize_client(args.model)
    except ValueError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Error initializing {args.model.upper()} client: {e}")
        return
    
    # Process all scenes
    processed_clips = process_scenes(scenes, client, args.model)

    print(f"\n===== PROCESSED SCENE DESCRIPTIONS (using {args.model.upper()}) =====")
    for clip in processed_clips:
        print(f"Scene: {clip.get('scene_number', 'unknown')}, Start Time: {clip.get('start_time', 'N/A')}, Type: {clip.get('type')}")
        print(f"Description: {clip.get('text')}")
        print(f"TTS Duration: {clip.get('tts_duration'):.2f}s\n")
    
    output_file = os.path.join(scenes_folder, args.output_file)
    with open(output_file, "w") as f:
        json.dump(processed_clips, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Total processed scenes: {len(processed_clips)}")
    print(f"Model used: {args.model.upper()}")

if __name__ == "__main__":
    main()