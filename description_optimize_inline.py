import json
import tempfile
import subprocess
import os
import argparse
from gtts import gTTS
from typing import List, Dict

# Model clients will be initialized based on choice
from openai import OpenAI # For Qwen
import google.generativeai as genai # For Gemini
from google.generativeai.types import HarmCategory, HarmBlockThreshold # For Gemini safety settings

from dotenv import load_dotenv

load_dotenv()

# --- Model Type Constants ---
MODEL_QWEN = "qwen"
MODEL_GEMINI = "gemini"

def get_tts_duration(text):
    """Calculate the duration of text when converted to speech."""
    if not text or text.isspace():
        return 0.0
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=True) as temp_file:
        try:
            tts = gTTS(text=text, lang='en')
            tts.save(temp_file.name)
            cmd = (f'ffprobe -v error -select_streams a:0 -show_entries format=duration '
                   f'-of csv="p=0" "{temp_file.name}"')
            duration_str = subprocess.check_output(cmd, shell=True, text=True).strip()
            if not duration_str: # Handle empty output from ffprobe
                print(f"Warning: ffprobe returned no duration for text: '{text[:50]}...'")
                return 5.0 # Assign a fallback duration to avoid errors, can be adjusted
            duration = float(duration_str)
            return duration
        except Exception as e:
            print(f"Error in get_tts_duration for text '{text[:50]}...': {e}. Returning estimated duration.")
            # Fallback: estimate duration based on word count if gTTS/ffprobe fails
            words = len(text.split())
            estimated_duration = words / 2.5  # Approx 2.5 words per second
            return max(1.0, estimated_duration) # Ensure at least 1s

def get_scene_clips(scene):
    clips = []
    for clip_data in scene.get('audio_clips', []): # Renamed to clip_data to avoid conflict
        text = clip_data.get('text')
        if not text: # Skip clips with no text
            continue
        duration = get_tts_duration(text)
        clips.append({
            'start_time': clip_data.get('start_time', 0),
            'text': text,
            'type': clip_data['type'],
            'scene_number': scene.get('scene_number', 'N/A'), # Use get with default
            'duration': duration,
            'end_time': clip_data.get('start_time', 0) + duration
        })
    clips.sort(key=lambda x: x['start_time'])
    return clips

def separate_clip_types(clips):
    text_clips = [clip for clip in clips if clip['type'] == 'Text on Screen']
    visual_clips = [clip for clip in clips if clip['type'] == 'Visual']
    return text_clips, visual_clips

def identify_transcript_gaps(scene, min_gap_duration=2.0):
    scene_start_time = scene.get('start_time', 0) # Absolute start time of the scene
    scene_duration_from_metadata = scene.get('duration', scene.get('end_time', scene_start_time) - scene_start_time)

    if not scene_duration_from_metadata and 'end_time' in scene:
        scene_duration_from_metadata = scene['end_time'] - scene_start_time
    elif not scene_duration_from_metadata:
        print(f"Warning: Scene {scene.get('scene_number')} has no duration. Cannot identify gaps accurately.")
        return []


    gaps = []
    if 'transcript' in scene and scene['transcript']:
        segments = sorted(scene['transcript'], key=lambda x: x.get('start', 0))
        current_time = 0.0 # Relative to scene start

        # Gap at beginning
        if segments and segments[0].get('start', 0) > current_time:
            gap_duration = segments[0]['start'] - current_time
            if gap_duration >= min_gap_duration:
                gaps.append({
                    'start_time': current_time,
                    'end_time': segments[0]['start'],
                    'duration': gap_duration
                })
        current_time = segments[0].get('start', 0) if segments else 0 # Ensure current_time advances even if no start gap

        # Gaps between segments
        for segment in segments:
            seg_start = segment.get('start', current_time)
            seg_end = segment.get('end', seg_start)

            if seg_start > current_time: # A gap exists before this segment
                gap_duration = seg_start - current_time
                if gap_duration >= min_gap_duration:
                    gaps.append({
                        'start_time': current_time,
                        'end_time': seg_start,
                        'duration': gap_duration
                    })
            current_time = max(current_time, seg_end) # Advance time to the end of current segment

        # Gap at end
        if current_time < scene_duration_from_metadata:
            gap_duration = scene_duration_from_metadata - current_time
            if gap_duration >= min_gap_duration:
                gaps.append({
                    'start_time': current_time,
                    'end_time': scene_duration_from_metadata,
                    'duration': gap_duration
                })
        return gaps
    else: # Scene has no transcript
        if scene_duration_from_metadata >= min_gap_duration:
            return [{
                'start_time': 0,
                'end_time': scene_duration_from_metadata,
                'duration': scene_duration_from_metadata
            }]
        else:
            return []

def find_gaps_around_text_clips(scene, text_clips, min_gap_duration=2.0):
    scene_duration = scene.get('duration', 0)
    if not scene_duration and 'start_time' in scene and 'end_time' in scene:
        scene_duration = scene['end_time'] - scene['start_time']
    if not scene_duration: # Fallback if still no duration
        print(f"Warning: Scene {scene.get('scene_number')} missing duration for gap calculation.")
        return []


    occupied_segments = []
    for clip in text_clips: # These are relative to scene start
        occupied_segments.append({
            'start': clip['start_time'],
            'end': clip['end_time'], # end_time here is start_time + tts_duration
            'type': 'text'
        })

    if 'transcript' in scene and scene['transcript']:
        for segment in scene['transcript']: # transcript start/end are relative to scene start
            occupied_segments.append({
                'start': segment.get('start', 0),
                'end': segment.get('end', 0),
                'type': 'transcript'
            })

    occupied_segments.sort(key=lambda x: x['start'])

    merged_segments = []
    if occupied_segments:
        current_segment = occupied_segments[0].copy() # Use a copy
        for segment in occupied_segments[1:]:
            if segment['start'] <= current_segment['end']:
                current_segment['end'] = max(current_segment['end'], segment['end'])
            else:
                merged_segments.append(current_segment)
                current_segment = segment.copy() # Use a copy
        merged_segments.append(current_segment)

    eligible_gaps = []
    current_gap_start = 0.0
    for segment in merged_segments:
        if segment['start'] > current_gap_start:
            gap_duration = segment['start'] - current_gap_start
            if gap_duration >= min_gap_duration:
                eligible_gaps.append({
                    'start_time': current_gap_start,
                    'end_time': segment['start'],
                    'duration': gap_duration
                })
        current_gap_start = max(current_gap_start, segment['end'])

    if current_gap_start < scene_duration:
        gap_duration = scene_duration - current_gap_start
        if gap_duration >= min_gap_duration:
            eligible_gaps.append({
                'start_time': current_gap_start,
                'end_time': scene_duration,
                'duration': gap_duration
            })
    
    if not merged_segments and scene_duration >= min_gap_duration: # Entire scene is a gap
        eligible_gaps.append({
            'start_time': 0,
            'end_time': scene_duration,
            'duration': scene_duration
        })

    return eligible_gaps

def process_scene_direct_placement(scene, min_gap_duration=2.0):
    scene_number = scene.get('scene_number', 'N/A')
    scene_start_abs = scene.get('start_time', 0) # Absolute start time of the scene
    scene_duration = scene.get('duration', 0)
    if not scene_duration and 'end_time' in scene:
        scene_duration = scene['end_time'] - scene_start_abs

    print(f"\n\n===== PROCESSING SCENE {scene_number} (DIRECT PLACEMENT) =====")
    print(f"Scene duration (relative): {scene_duration:.2f}s (Absolute start: {scene_start_abs:.2f}s)")

    clips_from_scene = get_scene_clips(scene) # Durations calculated here
    text_clips, visual_clips = separate_clip_types(clips_from_scene)

    print(f"\n-- Scene {scene_number}: Found {len(text_clips)} text clips and {len(visual_clips)} visual clips")

    placed_clips = []
    for text_clip in text_clips:
        placed_clips.append({
            'scene_number': text_clip['scene_number'],
            'start_time': text_clip['start_time'] + scene_start_abs, # Absolute time
            'end_time': text_clip['end_time'] + scene_start_abs,   # Absolute time
            'duration': text_clip['duration'],
            'type': 'Text on Screen',
            'text': text_clip['text']
        })

    eligible_gaps = find_gaps_around_text_clips(scene, text_clips, min_gap_duration)
    print(f"\n-- Scene {scene_number}: Found {len(eligible_gaps)} eligible gaps of at least {min_gap_duration}s (times relative to scene start)")
    for i, gap in enumerate(eligible_gaps):
        print(f"   Gap {i+1}: {gap['start_time']:.2f}s - {gap['end_time']:.2f}s (Duration: {gap['duration']:.2f}s)")

    for i, visual_clip in enumerate(visual_clips):
        # visual_clip start_time is relative to scene start
        clip_relative_start = visual_clip['start_time']
        clip_relative_end = visual_clip['end_time'] # This is start_time + duration
        fits_in_gap = False
        for gap in eligible_gaps:
            # A clip fits if its entire duration is within a gap's timeframe.
            # Allow a small tolerance (e.g., 0.1s) for floating point issues if needed, but strict is better.
            if gap['start_time'] <= clip_relative_start and clip_relative_end <= gap['end_time']:
                fits_in_gap = True
                break
        placed_clips.append({
            'scene_number': scene_number,
            'start_time': visual_clip['start_time'] + scene_start_abs, # Absolute time
            'end_time': visual_clip['end_time'] + scene_start_abs,     # Absolute time
            'duration': visual_clip['duration'],
            'type': 'Visual',
            'text': visual_clip['text'],
            'fits_in_gap': fits_in_gap # Based on relative times
        })

    placed_clips.sort(key=lambda x: x['start_time'])
    # ... (summary printing can remain similar) ...
    return placed_clips


def optimize_combined_clips(optimizer_client, optimizer_model_type, clips_to_optimize, available_duration, max_retries=3, scene_number_for_logging="N/A"):
    if not clips_to_optimize:
        return None, False

    combined_text = " ".join([clip['text'] for clip in clips_to_optimize])
    optimized_text = ""
    tts_duration = float('inf')

    for attempt in range(max_retries + 1): # +1 for the initial attempt
        if optimizer_model_type == MODEL_QWEN:
            if attempt == 0:
                prompt = f"""You are optimizing a set of visual descriptions for a video scene.
                            ORIGINAL DESCRIPTIONS: "{combined_text}"
                            AVAILABLE TIME: {available_duration:.2f} seconds
                            TASK: Combine and condense these descriptions to fit an available speaking duration of {available_duration:.2f} seconds.
                            GUIDELINES: Create a coherent, flowing description. Maintain the action order and use concise but natural language. The final description's spoken duration MUST be less than or equal to {available_duration:.2f} seconds.
                            OUTPUT FORMAT: Provide only the optimized description text, without any explanations, preamble, or markdown.
                            """
            else: # Retry prompt for Qwen
                prompt = f"""Your previous attempt to condense visual descriptions was too long.
                            PREVIOUS ATTEMPT (spoken duration {tts_duration:.2f}s): "{optimized_text}"
                            ORIGINAL DESCRIPTIONS: "{combined_text}"
                            AVAILABLE TIME: {available_duration:.2f} seconds. You need to make it shorter.
                            TASK: Create a SIGNIFICANTLY SHORTER version of the visual descriptions that an English speaker can voice in {available_duration:.2f} seconds or less.
                            GUIDELINES: Be more aggressive in cutting redundant details. Focus on the most critical visual elements.
                            OUTPUT FORMAT: Provide only the new, much shorter description text. No explanations.
                            """
            try:
                response = optimizer_client.chat.completions.create(
                    model="qwen2.5-72b-instruct", 
                    messages=[
                        {"role": "system", "content": "You are an expert at creating concise and accurate visual descriptions for video."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7 if attempt == 0 else 1.0, 
                    max_tokens=150 if attempt == 0 else 100
                )
                optimized_text = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"  - Error calling Qwen API (Scene {scene_number_for_logging}, Attempt {attempt+1}): {e}")
                if attempt == max_retries: return None, False
                time.sleep(2) 
                continue

        elif optimizer_model_type == MODEL_GEMINI:
            gemini_model_name = "gemini-1.5-pro-latest" 
            if attempt == 0:
                prompt = f"""Condense the following visual descriptions into a single, coherent narrative that can be spoken in {available_duration:.2f} seconds or less.
                            Original Descriptions (combine these):
                            "{combined_text}"

                            Available speaking time: {available_duration:.2f} seconds.

                            Instructions:
                            - Maintain the original order of actions and essential information.
                            - Use concise, natural English.
                            - The final description's spoken duration MUST NOT exceed {available_duration:.2f} seconds.
                            - Output ONLY the condensed description text. Do not include any other explanatory text, preamble, or markdown formatting.
                            """
            else: 
                prompt = f"""Your previous condensed description was too long.
                            Previous attempt (spoken duration {tts_duration:.2f}s): "{optimized_text}"
                            Original Descriptions (to be condensed): "{combined_text}"
                            Available speaking time: {available_duration:.2f} seconds. The new version MUST be shorter.

                            Instructions:
                            - Be more aggressive in shortening the text. Focus only on the most critical visual elements.
                            - The final description's spoken duration MUST NOT exceed {available_duration:.2f} seconds.
                            - Output ONLY the new, much shorter condensed description text. No explanations or markdown.
                            """
            try:
                response = optimizer_client.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7 if attempt == 0 else 1.0,
                        max_output_tokens=150 if attempt == 0 else 100
                    ),
                    # Configure safety settings for Gemini if needed, though less critical for text summarization
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    }
                )
                optimized_text = response.text.strip()
            except Exception as e:
                print(f"  - Error calling Gemini API (Scene {scene_number_for_logging}, Attempt {attempt+1}): {e}")
                if attempt == max_retries: return None, False
                time.sleep(2)
                continue

        else:
            print(f"  - Unknown optimizer model type: {optimizer_model_type}")
            return None, False

        tts_duration = get_tts_duration(optimized_text)
        print(f"  - Scene {scene_number_for_logging}, Opt. Attempt {attempt+1}: Text='{optimized_text[:60]}...', TTS Dura: {tts_duration:.2f}s (Target: {available_duration:.2f}s)")

        if tts_duration <= available_duration and optimized_text: # Ensure text is not empty
            fits_in_gap = True
            print(f"  - Success! Optimized description fits (Scene {scene_number_for_logging}).")
            optimized_clip_data = {
                'scene_number': clips_to_optimize[0]['scene_number'],
                'text': optimized_text,
                'type': 'Visual',
                'duration': tts_duration,
                'fits_in_gap': True,
                'original_texts': [c['text'] for c in clips_to_optimize]
            }
            return optimized_clip_data, True
        
        if attempt == max_retries:
            print(f"  - Max retries reached for optimization (Scene {scene_number_for_logging}). Final TTS: {tts_duration:.2f}s.")
            # Return the last attempt even if it doesn't fit, but mark it
            optimized_clip_data = {
                'scene_number': clips_to_optimize[0]['scene_number'],
                'text': optimized_text if optimized_text else combined_text, # Fallback to combined if opt text is empty
                'type': 'Visual',
                'duration': tts_duration if optimized_text else get_tts_duration(combined_text),
                'fits_in_gap': False,
                'original_texts': [c['text'] for c in clips_to_optimize]
            }
            return optimized_clip_data, False

    return None, False # Should be covered by loop logic


def process_scene(scene, optimizer_client, optimizer_model_type, min_gap_duration=2.0):
    scene_number = scene.get('scene_number', 'N/A')
    scene_start_abs = scene.get('start_time', 0)
    # scene_end_abs = scene.get('end_time', 0)
    scene_duration_relative = scene.get('duration', 0)
    if not scene_duration_relative and 'end_time' in scene:
         scene_duration_relative = scene['end_time'] - scene_start_abs
    
    print(f"\n\n===== PROCESSING SCENE {scene_number} (OPTIMIZED PLACEMENT with {optimizer_model_type.upper()}) =====")
    print(f"Scene duration (relative): {scene_duration_relative:.2f}s (Absolute start: {scene_start_abs:.2f}s)")

    clips_from_scene = get_scene_clips(scene)
    text_clips, visual_clips = separate_clip_types(clips_from_scene)

    print(f"\n-- Scene {scene_number}: Found {len(text_clips)} text clips and {len(visual_clips)} visual clips")

    placed_clips = []
    for text_clip in text_clips: # Text clips are placed with absolute times
        placed_clips.append({
            'scene_number': text_clip['scene_number'],
            'start_time': text_clip['start_time'] + scene_start_abs,
            'end_time': text_clip['end_time'] + scene_start_abs,
            'duration': text_clip['duration'],
            'type': 'Text on Screen',
            'text': text_clip['text']
        })

    eligible_gaps = find_gaps_around_text_clips(scene, text_clips, min_gap_duration)
    print(f"\n-- Scene {scene_number}: Found {len(eligible_gaps)} eligible gaps of at least {min_gap_duration}s (times relative to scene start)")
    for i, gap in enumerate(eligible_gaps):
        print(f"   Gap {i+1}: {gap['start_time']:.2f}s - {gap['end_time']:.2f}s (Duration: {gap['duration']:.2f}s)")

    processed_clip_ids = set() # To track original visual clips that have been processed

    for gap_idx, gap in enumerate(eligible_gaps):
        # Clips whose original relative start time falls within the gap's broader timeframe (allowing some lead-in)
        # visual_clips have start_time relative to scene start
        clips_in_gap_timeframe = [
            clip for clip in visual_clips
            if id(clip) not in processed_clip_ids and \
               (gap['start_time'] - 1.5 <= clip['start_time'] < gap['end_time'] or \
                gap['start_time'] < clip['end_time'] <= gap['end_time'] or \
                (clip['start_time'] < gap['start_time'] and clip['end_time'] > gap['end_time'])) # Overlaps entire gap
        ]
        clips_in_gap_timeframe.sort(key=lambda x: x['start_time'])

        if not clips_in_gap_timeframe:
            print(f"  Gap {gap_idx+1}: No unprocessed visual clips associated with this gap's timeframe. Skipping.")
            continue

        gap_actual_duration = gap['duration']
        print(f"\nProcessing Gap {gap_idx+1} (Duration: {gap_actual_duration:.2f}s) with {len(clips_in_gap_timeframe)} associated visual clips.")

        # Simple strategy: try to combine all clips for this gap
        optimized_clip_data, fits = optimize_combined_clips(
            optimizer_client, optimizer_model_type, clips_in_gap_timeframe, gap_actual_duration, scene_number_for_logging=scene_number
        )

        if optimized_clip_data:
            optimized_clip_data['start_time'] = gap['start_time'] + scene_start_abs # Place at gap start
            optimized_clip_data['end_time'] = optimized_clip_data['start_time'] + optimized_clip_data['duration']
            # Ensure it doesn't spill out of the original gap due to TTS duration rounding
            if optimized_clip_data['end_time'] > (gap['end_time'] + scene_start_abs):
                print(f"  Warning (Scene {scene_number}): Optimized clip duration {optimized_clip_data['duration']:.2f}s made it spill out of gap {gap_idx+1}. Marking as not fitting.")
                optimized_clip_data['fits_in_gap'] = False
            
            placed_clips.append(optimized_clip_data)
            for clip in clips_in_gap_timeframe:
                processed_clip_ids.add(id(clip))
        else:
            print(f"  Optimization failed for clips in Gap {gap_idx+1} (Scene {scene_number}). Placing original clips individually.")
            for clip in clips_in_gap_timeframe:
                if id(clip) not in processed_clip_ids: # Should always be true here but for safety
                    placed_clips.append({
                        'scene_number': scene_number,
                        'start_time': clip['start_time'] + scene_start_abs,
                        'end_time': clip['end_time'] + scene_start_abs,
                        'duration': clip['duration'],
                        'type': 'Visual',
                        'text': clip['text'],
                        'fits_in_gap': False # Marked as not fitting if placed individually after opt. failure
                    })
                    processed_clip_ids.add(id(clip))


    # Add any remaining visual clips that weren't associated with any gap or failed optimization
    remaining_visual_clips = [clip for clip in visual_clips if id(clip) not in processed_clip_ids]
    if remaining_visual_clips:
        print(f"\n-- Scene {scene_number}: Placing {len(remaining_visual_clips)} remaining visual clips individually (not part of optimized groups).")
        for clip in remaining_visual_clips:
            placed_clips.append({
                'scene_number': scene_number,
                'start_time': clip['start_time'] + scene_start_abs,
                'end_time': clip['end_time'] + scene_start_abs,
                'duration': clip['duration'],
                'type': 'Visual',
                'text': clip['text'],
                'fits_in_gap': False # These were not part of a successful gap optimization
            })

    placed_clips.sort(key=lambda x: x['start_time'])
    fit_clips = sum(1 for clip in placed_clips if clip.get('type') == 'Visual' and clip.get('fits_in_gap') == True)
    unfit_clips = sum(1 for clip in placed_clips if clip.get('type') == 'Visual' and clip.get('fits_in_gap') == False)
    
    print(f"\n-- Scene {scene_number} Summary:")
    print(f"   Visual clips: {len(visual_clips)} total")
    print(f"   Clips that fit in gaps: {fit_clips}")
    print(f"   Clips that don't fit in gaps: {unfit_clips}")
    print(f"   Total placed clips: {len(placed_clips)} (including {len(text_clips)} text clips)")
    
    return placed_clips

def main():
    parser = argparse.ArgumentParser(description="Process audio clips for video, optionally optimizing descriptions.")
    parser.add_argument("video_folder", help="Path to the video folder")
    parser.add_argument("--output", help="Output file name", required=True)
    parser.add_argument("--skip_optimization", action="store_true",
                        help="Skip optimization and place all clips at their original timestamps (direct placement).")
    parser.add_argument("--optimizer_model", type=str, choices=["gemini", "qwen"], required=True,
                        help="Choose the model for optimizing descriptions: 'gemini' or 'qwen'.")
    parser.add_argument("--min_gap", type=float, default=2.0, help="Minimum gap duration in seconds to consider for placing visual descriptions.")

    args = parser.parse_args()
    video_id = os.path.basename(os.path.normpath(args.video_folder))
    scenes_folder = os.path.join(args.video_folder, f"{video_id}_scenes")
    video_metadata_path = os.path.join(args.video_folder, f"{video_id}.json")

    if not os.path.exists(video_metadata_path):
        print(f"Error: Metadata file {video_metadata_path} not found.")
        return

    scenes_path = None
    preferred_input_filename = f"scene_info_{args.optimizer_model}.json"
    preferred_scenes_path = os.path.join(scenes_folder, preferred_input_filename)
    fallback_scenes_path_original = os.path.join(scenes_folder, "scene_info.json")

    if os.path.exists(preferred_scenes_path):
        scenes_path = preferred_scenes_path
        print(f"Using preferred input scene file: {scenes_path}")
    elif os.path.exists(fallback_scenes_path_original):
        scenes_path = fallback_scenes_path_original
        print(f"Warning: '{preferred_input_filename}' not found. Using fallback: {scenes_path}")
    else:
        print(f"Error: No suitable scene_info file found in {scenes_folder}. Checked for '{preferred_input_filename}' and '{os.path.basename(fallback_scenes_path_original)}'.")
        return

    with open(scenes_path, "r", encoding="utf-8") as f:
        scenes = json.load(f)

    all_clips = []
    optimizer_client = None

    if args.skip_optimization:
        print("\nUsing DIRECT PLACEMENT mode (skipping description optimization).")
        for scene in scenes:
            scene_clips = process_scene_direct_placement(scene, args.min_gap)
            all_clips.extend(scene_clips)
    else:
        print(f"\nUsing OPTIMIZED PLACEMENT mode with {args.optimizer_model.upper()} optimizer.")
        # The internal constants MODEL_QWEN and MODEL_GEMINI are still useful here for comparison
        if args.optimizer_model == MODEL_QWEN: # Compares with the string "qwen"
            api_key = os.getenv("API_KEY")
            if not api_key:
                print("Error: API_KEY environment variable not set (for Qwen optimizer).")
                return
            print("Setting up OpenAI client for DashScope API (Qwen)...")
            optimizer_client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            )
        elif args.optimizer_model == MODEL_GEMINI: # Compares with the string "gemini"
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not gemini_api_key:
                print("Error: GEMINI_API_KEY environment variable not set (for Gemini optimizer).")
                return
            print("Initializing Gemini client...")
            genai.configure(api_key=gemini_api_key)
            optimizer_client = genai.GenerativeModel("gemini-1.5-pro-latest")

        if not optimizer_client:
            print("Error: Optimizer client could not be initialized.")
            return

        for scene in scenes:
            scene_clips = process_scene(scene, optimizer_client, args.optimizer_model, args.min_gap)
            all_clips.extend(scene_clips)

    all_clips.sort(key=lambda x: x['start_time'])

    output_file_path = os.path.join(scenes_folder, args.output)
    with open(output_file_path, 'w', encoding="utf-8") as f:
        json.dump(all_clips, f, indent=2)

    print(f"\nResults saved to: {output_file_path}")
    print(f"Total audio clips generated: {len(all_clips)}")

if __name__ == "__main__":
    main()