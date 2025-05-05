import json
import tempfile
import subprocess
import os
import argparse
from gtts import gTTS
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def get_tts_duration(text):
    """Calculate the duration of text when converted to speech."""
    if not text or text.isspace():
        return 0.0
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=True) as temp_file:
        tts = gTTS(text=text, lang='en')
        tts.save(temp_file.name)
        cmd = (f'ffprobe -v error -select_streams a:0 -show_entries format=duration '
               f'-of csv="p=0" "{temp_file.name}"')
        duration = float(subprocess.check_output(cmd, shell=True).decode().strip())
        return duration

def get_scene_clips(scene):
    clips = []
    
    for clip in scene.get('audio_clips', []):
        # Calculate TTS duration for each clip
        duration = get_tts_duration(clip['text'])
        
        clips.append({
            'start_time': clip.get('start_time', 0),
            'text': clip['text'],
            'type': clip['type'],
            'scene_number': scene['scene_number'],
            'duration': duration,
            'end_time': clip.get('start_time', 0) + duration
        })
    
    # Sort by start time
    clips.sort(key=lambda x: x['start_time'])
    
    return clips

def separate_clip_types(clips):
    text_clips = [clip for clip in clips if clip['type'] == 'Text on Screen']
    visual_clips = [clip for clip in clips if clip['type'] == 'Visual']
    
    return text_clips, visual_clips

def identify_transcript_gaps(scene, min_gap_duration=2.0):
    scene_duration = scene['end_time'] - scene['start_time']
    
    if 'transcript' in scene and scene['transcript']:
        segments = sorted(scene['transcript'], key=lambda x: x.get('start', 0))
        
        gaps = []
        
        # Gap at beginning
        if segments[0]['start'] > 0:
            gap_duration = segments[0]['start']
            if gap_duration >= min_gap_duration:
                gaps.append({
                    'start_time': 0, 
                    'end_time': segments[0]['start'],
                    'duration': gap_duration
                })
        
        # Gaps between segments
        for i in range(len(segments) - 1):
            current_end = segments[i]['end']
            next_start = segments[i+1]['start']
            
            if next_start > current_end:
                gap_duration = next_start - current_end
                if gap_duration >= min_gap_duration:
                    gaps.append({
                        'start_time': current_end,
                        'end_time': next_start,
                        'duration': gap_duration
                    })
        
        # Gap at end
        if segments[-1]['end'] < scene_duration:
            gap_duration = scene_duration - segments[-1]['end']
            if gap_duration >= min_gap_duration:
                gaps.append({
                    'start_time': segments[-1]['end'],
                    'end_time': scene_duration,
                    'duration': gap_duration
                })
                
        return gaps
    else:
        # Scene has no transcript, treat entire scene as a gap if it's long enough
        if scene_duration >= min_gap_duration:
            return [{
                'start_time': 0,  # Relative to scene start
                'end_time': scene_duration,
                'duration': scene_duration
            }]
        else:
            return []

def find_gaps_around_text_clips(scene, text_clips, min_gap_duration=2.0):
    scene_duration = scene['duration']
    
    transcript_gaps = identify_transcript_gaps(scene, min_gap_duration)
    
    if not text_clips:
        return transcript_gaps
    
    sorted_text_clips = sorted(text_clips, key=lambda x: x['start_time'])
    
    occupied_segments = []
    for clip in sorted_text_clips:
        occupied_segments.append({
            'start': clip['start_time'],
            'end': clip['end_time'],
            'type': 'text'
        })
    
    if 'transcript' in scene and scene['transcript']:
        for segment in scene['transcript']:
            occupied_segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'type': 'transcript'
            })
    
    occupied_segments.sort(key=lambda x: x['start'])

    merged_segments = []
    if occupied_segments:
        current_segment = occupied_segments[0]
        for segment in occupied_segments[1:]:
            if segment['start'] <= current_segment['end']:
                current_segment['end'] = max(current_segment['end'], segment['end'])
            else:
                merged_segments.append(current_segment)
                current_segment = segment
        merged_segments.append(current_segment)
    

    eligible_gaps = []
    if merged_segments and merged_segments[0]['start'] > 0:
        duration = merged_segments[0]['start']
        if duration >= min_gap_duration:
            eligible_gaps.append({
                'start_time': 0,
                'end_time': merged_segments[0]['start'],
                'duration': duration
            })
    
    for i in range(len(merged_segments) - 1):
        start = merged_segments[i]['end']
        end = merged_segments[i+1]['start']
        duration = end - start
        
        if duration >= min_gap_duration:
            eligible_gaps.append({
                'start_time': start,
                'end_time': end,
                'duration': duration
            })
    
    # Gap at the end
    if merged_segments and merged_segments[-1]['end'] < scene_duration:
        duration = scene_duration - merged_segments[-1]['end']
        if duration >= min_gap_duration:
            eligible_gaps.append({
                'start_time': merged_segments[-1]['end'],
                'end_time': scene_duration,
                'duration': duration
            })
    
    if not merged_segments and scene_duration >= min_gap_duration:
        eligible_gaps.append({
            'start_time': 0,
            'end_time': scene_duration,
            'duration': scene_duration
        })
    
    return eligible_gaps

def process_scene_direct_placement(scene, min_gap_duration=2.0):
    """Process a scene by directly placing clips without any optimization but still checking gap fit"""
    scene_number = scene['scene_number']
    scene_start = scene['start_time']
    scene_end = scene['end_time']
    scene_duration = scene['duration']
    
    print(f"\n\n===== PROCESSING SCENE {scene_number} (DIRECT PLACEMENT) =====")
    print(f"Scene duration: {scene_duration:.2f}s ({scene_start:.2f}s - {scene_end:.2f}s)")

    clips = get_scene_clips(scene)
    text_clips, visual_clips = separate_clip_types(clips)
    
    print(f"\n-- Scene {scene_number}: Found {len(text_clips)} text clips and {len(visual_clips)} visual clips")
    
    placed_clips = []
    
    for text_clip in text_clips:
        placed_clips.append({
            'scene_number': text_clip['scene_number'],
            'start_time': text_clip['start_time'] + scene_start,
            'end_time': text_clip['end_time'] + scene_start,
            'duration': text_clip['duration'],
            'type': 'Text on Screen',
            'text': text_clip['text']
        })
    
    eligible_gaps = find_gaps_around_text_clips(scene, text_clips, min_gap_duration)
    
    print(f"\n-- Scene {scene_number}: Found {len(eligible_gaps)} eligible gaps of at least {min_gap_duration}s")
    for i, gap in enumerate(eligible_gaps):
        print(f"   Gap {i+1}: {gap['start_time']:.2f}s - {gap['end_time']:.2f}s (Duration: {gap['duration']:.2f}s)")
    
    # Process visual clips, checking if they fit in gaps
    for i, clip in enumerate(visual_clips):
        print(f"  Placing visual clip {i+1}/{len(visual_clips)}")
        
        # Check if this clip falls within any gap
        clip_start = clip['start_time']
        clip_end = clip['start_time'] + clip['duration']
        fits_in_gap = False
        
        for gap in eligible_gaps:
            gap_start = gap['start_time']
            gap_end = gap['end_time']
            
            # Check if clip starts within gap and ends within gap
            if gap_start - 1.5 <= clip_start < gap_end and clip_end <= gap_end:
                fits_in_gap = True
                break
        
        placed_clips.append({
            'scene_number': scene_number,
            'start_time': clip['start_time'] + scene_start,
            'end_time': clip['start_time'] + clip['duration'] + scene_start,
            'duration': clip['duration'],
            'type': 'Visual', 
            'text': clip['text'],
            'fits_in_gap': fits_in_gap
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

def optimize_combined_clips(client, clips, available_duration, max_retries=5):
    
    combined_text = " ".join([clip['text'] for clip in clips])
    prompt = f"""You are optimizing a set of visual descriptions for a video.
                ORIGINAL DESCRIPTIONS: "{combined_text}"
                
                AVAILABLE TIME: {available_duration:.2f} seconds
                
                TASK:
                Combine and condense these descriptions to fit within {available_duration:.2f} seconds.
                
                GUIDELINES:
                - Create a coherent, flowing description
                - Maintain the action order and use concise but natural language.
                - Ensure the final description MUST be spoken within {available_duration:.2f} seconds
                
                OUTPUT FORMAT:
                Provide only the optimized description text, without explanations."""
    
    try:
        response = client.chat.completions.create(
            model="qwen2.5-72b-instruct",
            messages=[
                {"role": "system", "content": "You are an expert at creating concise visual descriptions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        optimized_text = response.choices[0].message.content.strip()
        tts_duration = get_tts_duration(optimized_text)
    
        retry_count = 0
        while tts_duration > available_duration and retry_count < max_retries:
            print(f"  - Combined description too long ({tts_duration:.2f}s for {available_duration:.2f}s available). Retry {retry_count+1}...")
        
            retry_prompt = f"""You are optimizing multiple visual descriptions for a video.
                            PREVIOUS ATTEMPT: "{optimized_text}"
                            
                            This description takes {tts_duration:.2f} seconds to speak, but you only have {available_duration:.2f} seconds available.
                            You need to reduce it by {(tts_duration - available_duration):.2f} seconds.
                            
                            TASK:
                            Create a SHORTER version that MUST fit within {available_duration:.2f} seconds.
                            
                            GUIDELINES:
                            - Keep the most critical visual elements and eliminate redundant details.
                            - Use concise but natural language. 
                            
                            OUTPUT FORMAT:
                            Provide only the shortened description, nothing else."""
            
            retry_response = client.chat.completions.create(
                model="qwen2.5-72b-instruct",
                messages=[
                    {"role": "system", "content": "You are an expert at creating extremely concise descriptions."},
                    {"role": "user", "content": retry_prompt}
                ],
                temperature=1.2,
                max_tokens=100  
            )
            
            optimized_text = retry_response.choices[0].message.content.strip()
            tts_duration = get_tts_duration(optimized_text)
            retry_count += 1
           
            if tts_duration <= available_duration:
                print(f"  - Success! Retry produced a fitting description ({tts_duration:.2f}s).")
                break
        
        fits_in_gap = tts_duration <= available_duration
        status = "fits" if fits_in_gap else "doesn't fit"
        print(f"  - Final description {status} ({tts_duration:.2f}s vs {available_duration:.2f}s available)")
        
        optimized_clip = {
            'scene_number': clips[0]['scene_number'],
            'text': optimized_text,
            'type': 'Visual',
            'duration': tts_duration,
            'fits_in_gap': fits_in_gap,
            'original_texts': [c['text'] for c in clips]
        }
        
        return optimized_clip, fits_in_gap
        
    except Exception as e:
        print(f"Error optimizing combined descriptions: {e}")
        return None, False

def process_scene(scene, client, min_gap_duration=2.0):
    scene_number = scene['scene_number']
    scene_start = scene['start_time']
    scene_end = scene['end_time']
    scene_duration = scene['duration']
    
    print(f"\n\n===== PROCESSING SCENE {scene_number} =====")
    print(f"Scene duration: {scene_duration:.2f}s ({scene_start:.2f}s - {scene_end:.2f}s)")

    clips = get_scene_clips(scene)
    text_clips, visual_clips = separate_clip_types(clips)
    
    print(f"\n-- Scene {scene_number}: Found {len(text_clips)} text clips and {len(visual_clips)} visual clips")
    
    placed_clips = []
    
    for text_clip in text_clips:
        placed_clips.append({
            'scene_number': text_clip['scene_number'],
            'start_time': text_clip['start_time'] + scene_start,
            'end_time': text_clip['end_time'] + scene_start,
            'duration': text_clip['duration'],
            'type': 'Text on Screen',
            'text': text_clip['text']
        })
    
    eligible_gaps = find_gaps_around_text_clips(scene, text_clips, min_gap_duration)
    
    print(f"\n-- Scene {scene_number}: Found {len(eligible_gaps)} eligible gaps of at least {min_gap_duration}s")
    for i, gap in enumerate(eligible_gaps):
        print(f"   Gap {i+1}: {gap['start_time']:.2f}s - {gap['end_time']:.2f}s (Duration: {gap['duration']:.2f}s)")
    
    processed_clips = set()
    
    for gap_idx, gap in enumerate(eligible_gaps):
        gap_clips = [clip for clip in visual_clips 
             if gap['start_time'] - 1.5 <= clip['start_time'] < gap['end_time']]
        
        if not gap_clips:
            print(f"  Gap {gap_idx+1}: No visual clips in this timeframe, skipping")
            continue
        
        gap_clips.sort(key=lambda x: x['start_time'])
        
        gap_duration = gap['duration']
        print(f"\nProcessing gap {gap_idx+1}: {gap['start_time']:.2f}s - {gap['end_time']:.2f}s (Duration: {gap_duration:.2f}s)")
        print(f"Found {len(gap_clips)} clips in this gap's timeframe")
        
        if gap_duration < 15.0:
            print(f"Gap < 15s: Combining all {len(gap_clips)} clips and optimizing together")
            
            if gap_clips:
                clips_with_scene = []
                for clip in gap_clips:
                    clip_copy = clip.copy()
                    clip_copy['scene_number'] = scene_number
                    clips_with_scene.append(clip_copy)
                
                optimized_clip, fits_in_gap = optimize_combined_clips(client, clips_with_scene, gap_duration)
                
                if optimized_clip and fits_in_gap:
                    optimized_clip['start_time'] = gap['start_time'] + scene_start
                    optimized_clip['end_time'] = optimized_clip['start_time'] + optimized_clip['duration']
                    placed_clips.append(optimized_clip)
                    
                    # Mark all clips as processed
                    for clip in gap_clips:
                        processed_clips.add(id(clip))
                else:
                    print(f"  Combined description doesn't fit, using original clips with original timestamps")
                    # Place all individual clips with their original timestamps
                    for clip in gap_clips:
                        placed_clips.append({
                            'scene_number': scene_number,
                            'start_time': clip['start_time'] + scene_start,
                            'end_time': clip['start_time'] + clip['duration'] + scene_start,
                            'duration': clip['duration'],
                            'type': 'Visual',
                            'text': clip['text'],
                            'fits_in_gap': False
                        })
                        processed_clips.add(id(clip))
        else:
            if len(gap_clips) >= 2:
                mid_index = len(gap_clips) // 2
                mid_clip = gap_clips[mid_index]
                mid_timestamp = mid_clip['start_time']
                
                print(f"Gap >= 15s: Using middle clip at index {mid_index} (timestamp {mid_timestamp:.2f}s) as dividing point")
                
                first_half_clips = gap_clips[:mid_index]
                second_half_clips = gap_clips[mid_index:] if mid_index < len(gap_clips) else []
                
                print(f"First half: {len(first_half_clips)} clips, Second half: {len(second_half_clips)} clips")
                
                # First half: from start of gap to mid clip timestamp
                first_half_duration = mid_timestamp - gap['start_time']
                
                # Second half: from mid clip timestamp to end of gap
                second_half_duration = gap['end_time'] - mid_timestamp
                
                print(f"First half duration: {first_half_duration:.2f}s, Second half duration: {second_half_duration:.2f}s")
                
                if first_half_clips:
                    if first_half_duration >= min_gap_duration:
                        print(f"\nOptimizing first half clips to fit in {first_half_duration:.2f}s")
                        
                        clips_with_scene = []
                        for clip in first_half_clips:
                            clip_copy = clip.copy()
                            clip_copy['scene_number'] = scene_number
                            clips_with_scene.append(clip_copy)
                        
                        optimized_clip, fits_in_gap = optimize_combined_clips(client, clips_with_scene, first_half_duration)
                        
                        if optimized_clip and fits_in_gap:
                            optimized_clip['start_time'] = gap['start_time'] + scene_start
                            optimized_clip['end_time'] = optimized_clip['start_time'] + optimized_clip['duration']
                            optimized_clip['half'] = 'first'
                            placed_clips.append(optimized_clip)
                            
                            for clip in first_half_clips:
                                processed_clips.add(id(clip))
                        else:
                            print(f"  Could not optimize first half clips, using original placement")
                            for clip in first_half_clips:
                                placed_clips.append({
                                    'scene_number': scene_number,
                                    'start_time': clip['start_time'] + scene_start,
                                    'end_time': clip['start_time'] + clip['duration'] + scene_start,
                                    'duration': clip['duration'],
                                    'type': 'Visual',
                                    'text': clip['text'],
                                    'fits_in_gap': False
                                })
                                processed_clips.add(id(clip))
                    else:
                        print(f"  First half duration too short ({first_half_duration:.2f}s), using original placement")
                        for clip in first_half_clips:
                            placed_clips.append({
                                'scene_number': scene_number,
                                'start_time': clip['start_time'] + scene_start,
                                'end_time': clip['start_time'] + clip['duration'] + scene_start,
                                'duration': clip['duration'],
                                'type': 'Visual',
                                'text': clip['text'],
                                'fits_in_gap': False
                            })
                            processed_clips.add(id(clip))
                
                if second_half_clips:
                    if second_half_duration >= min_gap_duration:
                        print(f"\nOptimizing second half clips to fit in {second_half_duration:.2f}s")
                        
                        clips_with_scene = []
                        for clip in second_half_clips:
                            clip_copy = clip.copy()
                            clip_copy['scene_number'] = scene_number
                            clips_with_scene.append(clip_copy)
                        
                        optimized_clip, fits_in_gap = optimize_combined_clips(client, clips_with_scene, second_half_duration)
                        
                        if optimized_clip and fits_in_gap:
                            optimized_clip['start_time'] = mid_timestamp + scene_start
                            optimized_clip['end_time'] = optimized_clip['start_time'] + optimized_clip['duration']
                            placed_clips.append(optimized_clip)

                            for clip in second_half_clips:
                                processed_clips.add(id(clip))
                        else:
                            print(f"  Could not optimize second half clips, using original placement")
                            for clip in second_half_clips:
                                placed_clips.append({
                                    'scene_number': scene_number,
                                    'start_time': clip['start_time'] + scene_start,
                                    'end_time': clip['start_time'] + clip['duration'] + scene_start,
                                    'duration': clip['duration'],
                                    'type': 'Visual',
                                    'text': clip['text'],
                                    'fits_in_gap': False
                                })
                                processed_clips.add(id(clip))
                    else:
                        print(f"  Second half duration too short ({second_half_duration:.2f}s), using original placement")
                        for clip in second_half_clips:
                            placed_clips.append({
                                'scene_number': scene_number,
                                'start_time': clip['start_time'] + scene_start,
                                'end_time': clip['start_time'] + clip['duration'] + scene_start,
                                'duration': clip['duration'],
                                'type': 'Visual',
                                'text': clip['text'],
                                'fits_in_gap': False,
                            })
                            processed_clips.add(id(clip))
            else:
                print(f"Gap >= 15s but only one clip found, optimizing it directly")
                
                clip = gap_clips[0]
                
                # Create optimized clip directly
                optimized_clip, fits_in_gap = optimize_combined_clips(client, [clip], gap_duration)
                
                if optimized_clip and fits_in_gap:
                    optimized_clip['start_time'] = gap['start_time'] + scene_start
                    optimized_clip['end_time'] = optimized_clip['start_time'] + optimized_clip['duration']
                    optimized_clip['gap_idx'] = gap_idx
                    placed_clips.append(optimized_clip)
                    processed_clips.add(id(clip))
                else:
                    print(f"  Could not optimize single clip, using original placement")
                    # Place clip with its original timestamp
                    placed_clips.append({
                        'scene_number': scene_number,
                        'start_time': clip['start_time'] + scene_start,
                        'end_time': clip['start_time'] + clip['duration'] + scene_start,
                        'duration': clip['duration'],
                        'type': 'Visual',
                        'text': clip['text'],
                        'fits_in_gap': False
                    })
                    processed_clips.add(id(clip))
    
    # Add any remaining unprocessed visual clips with fits_in_gap = False
    unprocessed_clips = [clip for clip in visual_clips if id(clip) not in processed_clips]
    
    if unprocessed_clips:
        print(f"\nAdding {len(unprocessed_clips)} unprocessed visual clips with fits_in_gap = False")
        for clip in unprocessed_clips:
            placed_clips.append({
                'scene_number': scene_number,
                'start_time': clip['start_time'] + scene_start,
                'end_time': clip['start_time'] + clip['duration'] + scene_start,
                'duration': clip['duration'],
                'type': 'Visual',
                'text': clip['text'],
                'fits_in_gap': False
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
    parser = argparse.ArgumentParser(description="Process audio clips for video")
    parser.add_argument("video_folder", help="Path to the video folder")
    parser.add_argument("--output", help="Output file name", default="audio_clips_optimized.json")
    parser.add_argument("--skip-optimization", action="store_true", 
                        help="Skip optimization and place all clips at their original timestamps")
    
    args = parser.parse_args()
    video_id = os.path.basename(os.path.normpath(args.video_folder))  
    scenes_folder = os.path.join(args.video_folder, f"{video_id}_scenes")
    video_metadata_path = os.path.join(args.video_folder, f"{video_id}.json")
    
    with open(video_metadata_path, "r") as f:
        video_metadata = json.load(f)
    
    video_category = video_metadata.get("category", "Other")
    print(f"CATEGORY: {video_category}")
    
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
    
    all_clips = []
    
    if args.skip_optimization:
        print("\nUsing DIRECT PLACEMENT mode (skipping optimization)")
        for scene in scenes:
            scene_clips = process_scene_direct_placement(scene)
            all_clips.extend(scene_clips)
    else:
        print("\nUsing OPTIMIZED PLACEMENT mode")
        # Need OpenAI client for optimization
        api_key = os.getenv("API_KEY")
        base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        
        if not api_key:
            print("Error: API_KEY environment variable not set")
            return
            
        print("\nSetting up OpenAI client for DashScope API...")
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        
        for scene in scenes:
            scene_clips = process_scene(scene, client)
            all_clips.extend(scene_clips)

    all_clips.sort(key=lambda x: x['start_time'])
    
    # Save results
    output_file = os.path.join(scenes_folder, args.output) 
    with open(output_file, 'w') as f:
        json.dump(all_clips, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print(f"Final output: {len(all_clips)} unique clips")

if __name__ == "__main__":
    main()