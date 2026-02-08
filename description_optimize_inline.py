import json
import tempfile
import subprocess
import os
import argparse
import time
from typing import List, Dict
import torch
import google as genai
import openai
from google.genai.types import HarmCategory, HarmBlockThreshold
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

from dotenv import load_dotenv
load_dotenv()

MODEL_QWEN = "qwen"
MODEL_GEMINI = "gemini"
MODEL_GPT4 = "gpt"


def get_tts_duration(text: str, speaking_rate: float = 1.25) -> float:
    if not text or text.isspace():
        return 0.0
    
    # Count words
    words = len(text.split())
    
    # Google TTS speaks ~150 words per minute at normal speed
    # At 1.25x: 150 * 1.25 = 187.5 words per minute
    words_per_minute = 150 * speaking_rate
    
    # Convert to seconds
    duration = (words / words_per_minute) * 60
    
    return max(0.5, duration)

def get_scene_clips(scene: Dict) -> List[Dict]:
    clips = []
    for clip_data in scene.get('audio_clips', []):
        text = clip_data.get('text')
        if not text:
            continue
        duration = get_tts_duration(text)  # Simple call
        clips.append({
            'start_time': clip_data.get('start_time', 0),
            'text': text,
            'type': clip_data['type'],
            'scene_number': scene.get('scene_number', 'N/A'),
            'duration': duration,
            'end_time': clip_data.get('start_time', 0) + duration
        })
    clips.sort(key=lambda x: x['start_time'])
    return clips

def separate_clip_types(clips: List[Dict]) -> tuple[List[Dict], List[Dict]]:
    text_clips = [clip for clip in clips if clip['type'] == 'Text on Screen']
    visual_clips = [clip for clip in clips if clip['type'] == 'Visual']
    return text_clips, visual_clips

def find_gaps_around_text_clips(scene: Dict, text_clips: List[Dict], min_gap_duration: float) -> List[Dict]:
    scene_duration = scene.get('duration', 0)
    if not scene_duration and 'start_time' in scene and 'end_time' in scene:
        scene_duration = scene['end_time'] - scene['start_time']
    if not scene_duration:
        print(f"Warning: Scene {scene.get('scene_number')} missing duration for gap calculation.")
        return []

    occupied_segments = [{'start': clip['start_time'], 'end': clip['end_time']} for clip in text_clips]
    if 'transcript' in scene and scene['transcript']:
        for segment in scene['transcript']:
            occupied_segments.append({'start': segment.get('start', 0), 'end': segment.get('end', 0)})

    occupied_segments.sort(key=lambda x: x['start'])

    merged_segments = []
    if occupied_segments:
        current_segment = occupied_segments[0].copy()
        for segment in occupied_segments[1:]:
            if segment['start'] <= current_segment['end']:
                current_segment['end'] = max(current_segment['end'], segment['end'])
            else:
                merged_segments.append(current_segment)
                current_segment = segment.copy()
        merged_segments.append(current_segment)

    eligible_gaps = []
    current_gap_start = 0.0
    for segment in merged_segments:
        if segment['start'] > current_gap_start:
            gap_duration = segment['start'] - current_gap_start
            if gap_duration >= min_gap_duration:
                eligible_gaps.append({'start_time': current_gap_start, 'end_time': segment['start'], 'duration': gap_duration})
        current_gap_start = max(current_gap_start, segment['end'])

    if current_gap_start < scene_duration:
        gap_duration = scene_duration - current_gap_start
        if gap_duration >= min_gap_duration:
            eligible_gaps.append({'start_time': current_gap_start, 'end_time': scene_duration, 'duration': gap_duration})
    
    if not merged_segments and scene_duration >= min_gap_duration:
        eligible_gaps.append({'start_time': 0, 'end_time': scene_duration, 'duration': scene_duration})

    return eligible_gaps


def optimize_with_qwen(optimizer_client: Dict, prompt: str) -> str:
    model = optimizer_client['model']
    processor = optimizer_client['processor']
    
    messages = [
        {"role": "system", "content": "You are an expert at creating concise and accurate visual descriptions for video."},
        {"role": "user", "content": prompt}
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)
    
    output_ids = model.generate(
        **inputs, 
        max_new_tokens=150, 
        temperature=0.7, 
        do_sample=True
    )
    
    input_token_len = inputs.input_ids.shape[1]
    generated_ids = output_ids[:, input_token_len:]
    response_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    
    return response_text.strip()


def optimize_combined_clips(optimizer_client, optimizer_model_type, clips_to_optimize, available_duration, max_retries=5, scene_number_for_logging="N/A"):
    if not clips_to_optimize:
        return None, False

    combined_text = " ".join([clip['text'] for clip in clips_to_optimize])
    optimized_text = ""
    tts_duration = float('inf')

    for attempt in range(max_retries + 1):
        prompt = ""
        if attempt == 0:
            prompt = (f'You are optimizing a set of visual descriptions for a video scene.\n'
                      f'ORIGINAL DESCRIPTIONS: "{combined_text}"\n'
                      f'AVAILABLE TIME: {available_duration:.2f} seconds\n'
                      f'TASK: Combine and condense these descriptions to fit an available speaking duration of {available_duration:.2f} seconds.\n'
                      f'GUIDELINES: Create a coherent, flowing description. Maintain the action order and use concise but natural language. The final description\'s spoken duration MUST be less than or equal to {available_duration:.2f} seconds.\n'
                      f'OUTPUT FORMAT: Provide only the optimized description text, without any explanations, preamble, or markdown.')
        else:
            prompt = (f'Your previous attempt to condense visual descriptions was too long.\n'
                      f'PREVIOUS ATTEMPT (spoken duration {tts_duration:.2f}s): "{optimized_text}"\n'
                      f'ORIGINAL DESCRIPTIONS: "{combined_text}"\n'
                      f'AVAILABLE TIME: {available_duration:.2f} seconds. You need to make it shorter.\n'
                      f'TASK: Create a SIGNIFICANTLY SHORTER version of the visual descriptions that an English speaker can voice in {available_duration:.2f} seconds or less.\n'
                      f'GUIDELINES: Be more aggressive in cutting redundant details. Focus on the most critical visual elements.\n'
                      f'OUTPUT FORMAT: Provide only the new, much shorter description text. No explanations.')

        try:
            if optimizer_model_type == MODEL_QWEN:
                optimized_text = optimize_with_qwen(optimizer_client, prompt)
            
            elif optimizer_model_type == MODEL_GEMINI:
                response = optimizer_client.generate_content(
                    prompt,
                    generation_config={"temperature": 0.7 if attempt == 0 else 1.0, "max_output_tokens": 150},
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    }
                )
                optimized_text = response.text.strip()
            
            elif optimizer_model_type == MODEL_GPT4:
                response = optimizer_client.chat.completions.create(
                    model="gpt-4o", # Replace with "gpt-4.1-..." when available
                    messages=[
                        {"role": "system", "content": "You are an expert at creating concise and accurate visual descriptions for video, condensing them to fit specific time constraints."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7 if attempt == 0 else 1.0,
                    max_tokens=150
                )
                optimized_text = response.choices[0].message.content.strip()

            else:
                print(f"  - Unknown optimizer model type: {optimizer_model_type}")
                return None, False
        except Exception as e:
            print(f"  - Error calling {optimizer_model_type.upper()} (Scene {scene_number_for_logging}, Attempt {attempt+1}): {e}")
            if attempt == max_retries: return None, False
            time.sleep(2)
            continue

        tts_duration = get_tts_duration(optimized_text)
        print(f"  - Scene {scene_number_for_logging}, Opt. Attempt {attempt+1}: Text='{optimized_text[:60]}...', TTS Dura: {tts_duration:.2f}s (Target: {available_duration:.2f}s)")

        if tts_duration <= available_duration and optimized_text:
            print(f"  - Success! Optimized description fits (Scene {scene_number_for_logging}).")
            return {
                'scene_number': clips_to_optimize[0]['scene_number'],
                'text': optimized_text, 'type': 'Visual', 'duration': tts_duration,
                'fits_in_gap': True, 'original_texts': [c['text'] for c in clips_to_optimize]
            }, True

    print(f"  - Max retries reached for optimization (Scene {scene_number_for_logging}).")
    return {
        'scene_number': clips_to_optimize[0]['scene_number'],
        'text': optimized_text if optimized_text else combined_text, 'type': 'Visual',
        'duration': tts_duration if optimized_text else get_tts_duration(combined_text),
        'fits_in_gap': False, 'original_texts': [c['text'] for c in clips_to_optimize]
    }, False

def process_scene(scene: Dict, optimizer_client, optimizer_model_type: str, min_gap_duration: float):
    scene_number = scene.get('scene_number', 'N/A')
    scene_start_abs = scene.get('start_time', 0)
    
    print(f"\n\n===== PROCESSING SCENE {scene_number} (OPTIMIZED PLACEMENT with {optimizer_model_type.upper()}) =====")
    
    clips_from_scene = get_scene_clips(scene)
    text_clips, visual_clips = separate_clip_types(clips_from_scene)
    print(f"\n-- Scene {scene_number}: Found {len(text_clips)} text clips and {len(visual_clips)} visual clips")

    placed_clips = [{'scene_number': c['scene_number'], 'start_time': c['start_time'] + scene_start_abs,
                     'end_time': c['end_time'] + scene_start_abs, 'duration': c['duration'],
                     'type': 'Text on Screen', 'text': c['text']} for c in text_clips]

    eligible_gaps = find_gaps_around_text_clips(scene, text_clips, min_gap_duration)
    print(f"\n-- Scene {scene_number}: Found {len(eligible_gaps)} eligible gaps >= {min_gap_duration}s")

    processed_clip_ids = set()
    current_placement_time = 0  # Track where we are in absolute time
    
    for gap_idx, gap in enumerate(eligible_gaps):
        clips_in_gap_timeframe = [c for c in visual_clips if id(c) not in processed_clip_ids and gap['start_time'] - 1.5 <= c['start_time'] < gap['end_time']]
        clips_in_gap_timeframe.sort(key=lambda x: x['start_time'])
        if not clips_in_gap_timeframe: continue

        print(f"\nProcessing Gap {gap_idx+1} (Duration: {gap['duration']:.2f}s) with {len(clips_in_gap_timeframe)} associated clips.")
        
        # Start placing clips at the beginning of this gap
        gap_start_abs = gap['start_time'] + scene_start_abs
        gap_end_abs = gap['end_time'] + scene_start_abs
        clip_placement_cursor = gap_start_abs
        
        optimized_clip_data, fits = optimize_combined_clips(optimizer_client, optimizer_model_type, clips_in_gap_timeframe, gap['duration'], scene_number_for_logging=scene_number)

        if optimized_clip_data:
            # Place the optimized clip at the current cursor position
            optimized_clip_data['start_time'] = clip_placement_cursor
            optimized_clip_data['end_time'] = clip_placement_cursor + optimized_clip_data['duration']
            
            # Check if it fits in the gap
            if optimized_clip_data['end_time'] > gap_end_abs:
                optimized_clip_data['fits_in_gap'] = False
            
            placed_clips.append(optimized_clip_data)
            
            # Update cursor for next potential clip in this gap
            clip_placement_cursor = optimized_clip_data['end_time']
            
            for clip in clips_in_gap_timeframe:
                processed_clip_ids.add(id(clip))

    remaining_visual_clips = [c for c in visual_clips if id(c) not in processed_clip_ids]
    if remaining_visual_clips:
        print(f"\n-- Scene {scene_number}: Placing {len(remaining_visual_clips)} remaining individual clips.")
        
        # Sort remaining clips by their original start time
        remaining_visual_clips.sort(key=lambda x: x['start_time'])
        
        # Place them sequentially, respecting their original relative timing
        for i, clip in enumerate(remaining_visual_clips):
            clip_start_abs = clip['start_time'] + scene_start_abs
            
            # Check if this overlaps with any already placed clip
            overlaps = False
            for placed in placed_clips:
                if not (clip_start_abs >= placed['end_time'] or 
                       (clip_start_abs + clip['duration']) <= placed['start_time']):
                    overlaps = True
                    break
            
            # If it overlaps, find the next available slot
            if overlaps:
                # Find the earliest time after all placed clips in this region
                relevant_clips = [p for p in placed_clips if p['start_time'] <= clip_start_abs + 10]  # within 10s window
                if relevant_clips:
                    latest_end = max(p['end_time'] for p in relevant_clips)
                    clip_start_abs = max(clip_start_abs, latest_end + 0.1)  # small buffer
            
            placed_clips.append({
                'scene_number': scene_number, 
                'start_time': clip_start_abs,
                'end_time': clip_start_abs + clip['duration'], 
                'duration': clip['duration'],
                'type': 'Visual', 
                'text': clip['text'], 
                'fits_in_gap': False
            })

    placed_clips.sort(key=lambda x: x['start_time'])
    
    # Final check: verify no duplicate start times
    start_times = [clip['start_time'] for clip in placed_clips]
    if len(start_times) != len(set(start_times)):
        print(f"WARNING: Scene {scene_number} has duplicate start times after processing!")
        for i, clip in enumerate(placed_clips):
            print(f"  Clip {i}: {clip['start_time']:.2f}s - {clip['text'][:50]}")
    
    return placed_clips


def main():
    parser = argparse.ArgumentParser(description="Process and optimize audio descriptions for video scenes.")
    parser.add_argument("video_folder", help="Path to the video folder")
    parser.add_argument("--output", help="Output JSON file name", required=True)
    parser.add_argument("--optimizer_model", type=str, choices=[MODEL_GEMINI, MODEL_QWEN, MODEL_GPT4], default=MODEL_GPT4,
                        help="Choose the model for optimizing descriptions: 'gemini', 'qwen', or 'gpt'.")
    parser.add_argument("--min_gap", type=float, default=2.0, help="Minimum gap duration in seconds to consider for placing descriptions.")
    args = parser.parse_args()

    video_id = os.path.basename(os.path.normpath(args.video_folder))
    scenes_folder = os.path.join(args.video_folder, f"{video_id}_scenes")
    
    preferred_input_filename = "scene_info.json" # Default
    if args.optimizer_model == MODEL_QWEN:
        preferred_input_filename = "scene_info_qwen.json"
    elif args.optimizer_model == MODEL_GEMINI:
        preferred_input_filename = "scene_info_gemini.json"
    elif args.optimizer_model == MODEL_GPT4:
        preferred_input_filename = "scene_info_gpt.json"
        
    preferred_scenes_path = os.path.join(scenes_folder, preferred_input_filename)
    fallback_scenes_path = os.path.join(scenes_folder, "scene_info.json")
    
    scenes_path = ""
    if os.path.exists(preferred_scenes_path):
        scenes_path = preferred_scenes_path
    elif os.path.exists(fallback_scenes_path):
        scenes_path = fallback_scenes_path
    else:
        print(f"Error: No suitable scene_info file found in {scenes_folder}.")
        return

    print(f"Using input scene file: {scenes_path}")
    with open(scenes_path, "r", encoding="utf-8") as f:
        scenes = json.load(f)

    optimizer_client = None
    if args.optimizer_model == MODEL_QWEN:
        print("Initializing LOCAL Qwen model with 4-bit quantization...")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-72B-Instruct", 
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2", 
            device_map="auto", 
            quantization_config=quantization_config,
            cache_dir="../.cache")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")
        optimizer_client = {'model': model, 'processor': processor}
    elif args.optimizer_model == MODEL_GEMINI:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            print("Error: GEMINI_API_KEY environment variable not set.")
            return
        print("Initializing Gemini API client...")
        genai.configure(api_key=gemini_api_key)
        optimizer_client = genai.GenerativeModel("gemini-1.5-pro-latest")
    elif args.optimizer_model == MODEL_GPT4:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("Error: OPENAI_API_KEY environment variable not set.")
            return
        print("Initializing OpenAI API client...")
        optimizer_client = openai.OpenAI(api_key=openai_api_key)

    if not optimizer_client:
        print("Error: Optimizer client could not be initialized.")
        return

    all_clips = []
    for scene in scenes:
        scene_clips = process_scene(scene, optimizer_client, args.optimizer_model, args.min_gap)
        all_clips.extend(scene_clips)

    output_file_path = os.path.join(scenes_folder, args.output)
    with open(output_file_path, 'w', encoding="utf-8") as f:
        json.dump(all_clips, f, indent=2)

    print(f"\nResults saved to: {output_file_path}")
    print(f"Total audio clips generated: {len(all_clips)}")

if __name__ == "__main__":
    main()