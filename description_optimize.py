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
    """Generate TTS for the given text and return its duration (in seconds)."""
    if not text or text.isspace():
        return 0.0
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=True) as temp_file:
        tts = gTTS(text=text, lang='en')
        tts.save(temp_file.name)
        cmd = (f'ffprobe -v error -select_streams a:0 -show_entries format=duration '
               f'-of csv="p=0" "{temp_file.name}"')
        duration = float(subprocess.check_output(cmd, shell=True).decode().strip())
        return duration

def compute_global_gaps(scenes, min_duration=1.5):
    """Compute time intervals without transcripts that are at least min_duration seconds long."""
    gaps = []
    
    for scene in scenes:
        scene_start = scene['start_time']
        scene_end = scene['end_time']
        transcripts = scene['transcript']
        
        if not transcripts:  # If no transcripts, whole scene is a gap
            if scene_end - scene_start >= min_duration:
                gaps.append({'start_time': scene_start, 'end_time': scene_end, 'scene_number': scene['scene_number']})
        else:
            transcripts.sort(key=lambda x: x['start'])
            
            # Gap at beginning
            if transcripts[0]['start'] > 0:
                gap_duration = transcripts[0]['start']
                if gap_duration >= min_duration:
                    gaps.append({'start_time': scene_start, 'end_time': scene_start + transcripts[0]['start'], 
                                'scene_number': scene['scene_number']})
            
            # Gaps between transcripts
            for i in range(len(transcripts) - 1):
                current_end = scene_start + transcripts[i]['end']
                next_start = scene_start + transcripts[i + 1]['start']
                if next_start > current_end:
                    gap_duration = next_start - current_end
                    if gap_duration >= min_duration:
                        gaps.append({'start_time': current_end, 'end_time': next_start, 
                                    'scene_number': scene['scene_number']})
            
            # Gap at end
            if scene_start + transcripts[-1]['end'] < scene_end:
                gap_duration = scene_end - (scene_start + transcripts[-1]['end'])
                if gap_duration >= min_duration:
                    gaps.append({'start_time': scene_start + transcripts[-1]['end'], 'end_time': scene_end, 
                                'scene_number': scene['scene_number']})
    
    return merge_gaps(gaps)

def merge_gaps(gaps, min_duration=1.5):
    """Merge adjacent or overlapping gaps and filter out gaps shorter than min_duration."""
    if not gaps:
        return []
        
    gaps.sort(key=lambda x: x['start_time'])
    merged_gaps = []
    current_gap = gaps[0]
    
    for gap in gaps[1:]:
        if gap['start_time'] <= current_gap['end_time']:
            current_gap['end_time'] = max(current_gap['end_time'], gap['end_time'])
            if gap['scene_number'] != current_gap['scene_number']:
                current_gap['scene_number'] = f"{current_gap['scene_number']}-{gap['scene_number']}"
        else:
            # Only add the gap if it's longer than min_duration
            if current_gap['end_time'] - current_gap['start_time'] >= min_duration:
                merged_gaps.append(current_gap)
            current_gap = gap
    
    # Check the last gap
    if current_gap['end_time'] - current_gap['start_time'] >= min_duration:
        merged_gaps.append(current_gap)
        
    return merged_gaps

def compute_global_audio_clips(scenes):
    """Compute global audio clips with absolute timestamps."""
    audio_clips = []
    
    for scene in scenes:
        scene_start = scene['start_time']
        for clip in scene['audio_clips']:
            audio_clips.append({
                'start_time': scene_start + clip['start_time'],
                'text': clip['text'],
                'type': clip['type'],
                'scene_number': scene['scene_number']
            })
    
    audio_clips.sort(key=lambda x: x['start_time'])
    
    i = 0
    while i < len(audio_clips) - 1:
        current = audio_clips[i]
        next_clip = audio_clips[i + 1]

        if (current['start_time'] == next_clip['start_time'] and
            current['type'] == 'Text on Screen' and next_clip['type'] == 'Visual'):
            current['tts_duration'] = get_tts_duration(current['text'])
            next_clip['start_time'] = current['start_time'] + current['tts_duration']
        
        i += 1
    
    audio_clips.sort(key=lambda x: x['start_time'])
    
    for i,clip in enumerate(audio_clips):
        clip['id'] = i
    return audio_clips

def compute_subgaps_and_candidates(gaps, audio_clips):
    """Compute subgaps after reserving text clips and find visual candidates."""
    subgaps = []
    candidates_id = set()
    candidates_by_subgap = {}
    
    for i, gap in enumerate(gaps):
        gap_start, gap_end = gap['start_time'], gap['end_time']
        
        # Find text clips in this gap
        text_clips = [clip for clip in audio_clips 
                     if clip['type'] == 'Text on Screen' and gap_start <= clip['start_time'] < gap_end]
        text_clips.sort(key=lambda x: x['start_time'])
        
        # Add TTS duration to text clips
        for clip in text_clips:
            clip['tts_duration'] = get_tts_duration(clip['text'])
        
        if not text_clips:  # No text clips - entire gap is a subgap
            subgap_id = f"subgap_{i}_0"
            subgap = {'id': subgap_id, 'start_time': gap_start, 'end_time': gap_end, 'parent_gap_index': i}
            subgaps.append(subgap)
            
            # Find visual candidates
            candidates_by_subgap[subgap_id] = []               
            for clip in audio_clips:
                if clip['type'] == 'Visual' and subgap['start_time'] <= clip['start_time'] < subgap['end_time']:
                    candidates_by_subgap[subgap_id].append(clip)
                    candidates_id.add(clip['id'])
        else:
            # Create subgaps around text clips
            current_time = gap_start
            
            for j, text_clip in enumerate(text_clips):
                # Subgap before text clip if there's space
                if text_clip['start_time'] > current_time:
                    subgap_id = f"subgap_{i}_{j}_before"
                    subgap = {'id': subgap_id, 'start_time': current_time, 
                             'end_time': text_clip['start_time'], 'parent_gap_index': i}
                    subgaps.append(subgap)
                    
                    # Find visual candidates for this subgap
                    candidates_by_subgap[subgap_id] = []               
                    for clip in audio_clips:
                        if clip['type'] == 'Visual' and subgap['start_time'] <= clip['start_time'] < subgap['end_time']:
                            candidates_by_subgap[subgap_id].append(clip)
                            candidates_id.add(clip['id'])
                
                # Update current time after text clip
                current_time = text_clip['start_time'] + text_clip['tts_duration']
            
            # Final subgap after last text clip if there's space
            if current_time < gap_end:
                subgap_id = f"subgap_{i}_final"
                subgap = {'id': subgap_id, 'start_time': current_time, 'end_time': gap_end, 'parent_gap_index': i}
                subgaps.append(subgap)
                
                candidates_by_subgap[subgap_id] = []               
                for clip in audio_clips:
                    if clip['type'] == 'Visual' and subgap['start_time'] <= clip['start_time'] < subgap['end_time']:
                        candidates_by_subgap[subgap_id].append(clip)
                        candidates_id.add(clip['id'])
    
    return subgaps, candidates_by_subgap, candidates_id

def optimize_descriptions(client, subgap, candidates, max_retries=3):
    """Optimize visual descriptions to fit within a subgap with automatic retries if needed."""
    # If no candidates, return None
    if not candidates:
        return None
    # Get gap information
    available_duration = subgap['end_time'] - subgap['start_time']
    
    # Construct candidates as a string
    candidates_str = ""
    scene_number = candidates[0]['scene_number']
    for i, candidate in enumerate(candidates):
        candidates_str += f"{i+1}. Start time: {candidate['start_time']:.2f}s - \"{candidate['text']}\"\n"
    
    # Create initial prompt
    prompt = f"""You are optimizing visual descriptions for a video.
                GAP INFORMATION:
                - Gap duration: {available_duration:.2f} seconds (available time for audio)
                - Gap start time: {subgap['start_time']:.2f}s
                - Gap end time: {subgap['end_time']:.2f}s

                CANDIDATE DESCRIPTIONS:
                {candidates_str}

                TASK:
                1. Create a contextually rich description that fits within {available_duration:.2f} seconds.
                2. Guidelines:
                - Keep as much detail as possible while remaining within the time limit
                - Preserve the most visually striking and significant elements
                - Use evocative, precise language that paints a clear mental image
                - If needed, condense redundant information while retaining key details
                3. IMPORTANT RULES:
                - A description cannot have a start time earlier than its original start time
                - The final description MUST be short enough to be spoken within {available_duration:.2f} seconds
                - Maximize descriptive detail within the time constraint

                OUTPUT FORMAT:
                Provide only the final optimized description text, without any explanations."""
    
    # Initial attempt
    try:
        response = client.chat.completions.create(
            model="qwen2.5-72b-instruct",
            messages=[
                {"role": "system", "content": "You are an expert audio describer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        # Get the optimized text
        optimized_text = response.choices[0].message.content.strip()
        
        # Calculate TTS duration
        tts_duration = get_tts_duration(optimized_text)
        
        # Create result
        optimized_clip = {
            'scene_number': scene_number,
            'start_time': subgap['start_time'],
            'duration': tts_duration,
            'end_time': subgap['start_time'] + tts_duration,
            'type': 'Visual',
            'text': optimized_text,
            'fits_in_gap': tts_duration <= available_duration + 0.5
        }
        
        # If it doesn't fit and we have retries left, try again
        retry_count = 0
        while not optimized_clip['fits_in_gap'] and retry_count < max_retries:
            print(f"  - Description too long ({optimized_clip['duration']:.2f}s for {available_duration:.2f}s gap). Retry {retry_count+1}...")
            
            # Create retry prompt with feedback about previous attempt
            retry_prompt = f"""You are optimizing visual descriptions for a video.
                            GAP INFORMATION:
                            - Gap duration: {available_duration:.2f} seconds (available time for audio)
                            - Your previous description took {optimized_clip['duration']:.2f} seconds, which is longer then available time.
                            - You need to reduce it by {(optimized_clip['duration'] - available_duration):.2f} seconds.

                            PREVIOUS ATTEMPT (TOO LONG):
                            "{optimized_clip['text']}"

                            TASK:
                            Create a SHORTER version that MUST fits within {available_duration:.2f} seconds.
                            
                            IMPORTANT: 
                            - Do NOT over-shorten - only remove enough words to fit the time constraint.
                            - Maintain the core visual elements and overall meaning.
                            - Focus on removing less essential adjectives, redundant phrases, or secondary details.

                            OUTPUT FORMAT:
                            Provide only the shortened description, nothing else."""
            
            # Make retry attempt
            retry_response = client.chat.completions.create(
                model="qwen2.5-72b-instruct",
                messages=[
                    {"role": "system", "content": "You are an expert at creating extremely concise descriptions."},
                    {"role": "user", "content": retry_prompt}
                ],
                temperature=0.9,
                max_tokens=100  # Even fewer tokens for retries
            )
            
            # Get retry text
            retry_text = retry_response.choices[0].message.content.strip()
            
            # Calculate new duration
            retry_duration = get_tts_duration(retry_text)
            
            # Update the clip
            optimized_clip = {
                'scene_number': scene_number,
                'start_time': subgap['start_time'],
                'duration': retry_duration,
                'end_time': subgap['start_time'] + retry_duration,
                'type': 'Visual',
                'text': retry_text,
                'fits_in_gap': retry_duration <= available_duration + 0.5
            }
            
            # Update retry count
            retry_count += 1
            
            # If it fits, we're done
            if optimized_clip['fits_in_gap']:
                print(f"  - Success! Retry produced a fitting description ({retry_duration:.2f}s).")
                break
        
        return optimized_clip
        
    except Exception as e:
        print(f"Error optimizing description: {e}")
        return None
    
def combine_clips(audio_clips, optimized_clips_by_subgap, candidates_id):
    """Combine original text clips with optimized visual clips and sort by start time."""
    combined_clips = []
    
    # Add non-candidate clips from original audio clips
    for clip in audio_clips:
        if clip['id'] not in candidates_id:
            duration = get_tts_duration(clip['text'])
            
            combined_clips.append({
                'scene_number': clip['scene_number'],
                'start_time': clip['start_time'],
                'duration': duration,
                'end_time': clip['start_time'] + duration,
                'type': clip['type'],
                'text': clip['text'],
            })
    
    # Add all optimized clips
    for _, clip in optimized_clips_by_subgap.items():
        # Add clip directly since each subgap has one optimized clip
        combined_clips.append({
            'scene_number': clip['scene_number'],
            'start_time': clip['start_time'],
            'duration': clip['duration'],
            'end_time': clip['end_time'],
            'type': clip['type'],
            'text': clip['text'],
        })
    
    # Sort all clips by start time
    combined_clips.sort(key=lambda x: x['start_time'])
    
    return combined_clips

def main():
    parser = argparse.ArgumentParser(description="Globally optimize audio descriptions per gap using Qwen2.5")
    parser.add_argument("video_folder", help="Path to the video folder (must include scene_info.json)")
    
    args = parser.parse_args()
    
    video_id = os.path.basename(os.path.normpath(args.video_folder))
    scenes_folder = os.path.join(args.video_folder, f"{video_id}_scenes")
    scenes_json_path = os.path.join(scenes_folder, "scene_info.json")

    if not os.path.exists(scenes_json_path):
        print(f"Error: scene_info.json not found in {args.video_folder}")
        return
    
    with open(scenes_json_path, "r") as f:
        scenes = json.load(f)
        
    
    # Compute global gaps and audio clips
    gaps = compute_global_gaps(scenes)
    audio_clips = compute_global_audio_clips(scenes)

    # Compute subgaps and candidates
    subgaps, candidates_by_subgap, candidates_id = compute_subgaps_and_candidates(gaps, audio_clips)
    #print(f'AUDIO CLIPS: {audio_clips}\n\n')
    #print(f'CANDIDATES ID: {candidates_id}\n\n')
    #print(f'CANDIDATES BY SUBGAP: {candidates_by_subgap}\n\n')
    
    # Print results
    
    print("\n===== GAPS WITHOUT TRANSCRIPT (> 1.5s) =====")
    for i, gap in enumerate(gaps):
        print(f"Gap {i}: {gap['start_time']:.2f}s - {gap['end_time']:.2f}s (Duration: {gap['end_time'] - gap['start_time']:.2f}s, Scene: {gap['scene_number']})")
    
    print("\n===== ALL AUDIO CLIPS (ABSOLUTE TIMESTAMPS) =====")
    for i, clip in enumerate(audio_clips):
        if 'tts_duration' in clip and clip['type'] == 'Text on Screen':
            print(f"Clip {i}: {clip['start_time']:.2f}s - {clip['start_time'] + clip['tts_duration']:.2f}s - Type: {clip['type']}, Scene: {clip['scene_number']}")
        else:
            print(f"Clip {i}: {clip['start_time']:.2f}s - Type: {clip['type']}, Scene: {clip['scene_number']}")
        print(f"       Text: {clip['text']}")
    
    print("\n===== SUBGAPS CREATED AFTER RESERVING TEXT CLIPS =====")
    for i, subgap in enumerate(subgaps):
        print(f"Subgap {i}: {subgap['start_time']:.2f}s - {subgap['end_time']:.2f}s (Duration: {subgap['end_time'] - subgap['start_time']:.2f}s)")
        print(f"       Parent gap: {subgap['parent_gap_index']}")
    
    print("\n===== VISUAL CANDIDATES FOR EACH SUBGAP =====")
    for subgap_id, candidates in candidates_by_subgap.items():
        subgap_index = next((i for i, sg in enumerate(subgaps) if sg['id'] == subgap_id), None)
        print(f"For Subgap {subgap_index} - {len(candidates)} visual candidates:")
        for candidate in candidates:
            print(f"   Visual {candidate['id']}: {candidate['start_time']:.2f}s - Scene: {candidate['scene_number']}")
            print(f"           {candidate['text']}")
    
    # OPTIMIZE AUDIO CLIPS
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
    
    # Process each subgap
    print("\n===== OPTIMIZED CLIPS BY GAP =====")
    
    # Use a different data structure to store optimized clips by subgap ID
    optimized_clips_by_subgap = {}

    for i, subgap in enumerate(subgaps):
        subgap_id = subgap['id']
        candidates = candidates_by_subgap.get(subgap_id, [])
        
        if not candidates:
            continue
        
        optimized_clip = optimize_descriptions(client, subgap, candidates)
        
        if optimized_clip:
            optimized_clips_by_subgap[subgap_id] = optimized_clip

    # Print optimized clips by subgap
    for i, subgap in enumerate(subgaps):
        subgap_id = subgap['id']
        
        # Skip subgaps without optimized clips
        if subgap_id not in optimized_clips_by_subgap:
            continue
        
        clip = optimized_clips_by_subgap[subgap_id]
        
        print(f"\nSUBGAP {i} ({subgap_id}): {subgap['start_time']:.2f}s - {subgap['end_time']:.2f}s")
        print("OPTIMIZED CLIP:")
        print("{")
        print(f"  start_time: {clip['start_time']:.2f}s,")
        print(f"  duration: {clip['duration']:.2f}s,")
        print(f"  end_time: {clip['end_time']:.2f}s,")
        print(f"  type: {clip['type']},")
        print(f"  text: \"{clip['text']}\"")
        if not clip['fits_in_gap']:
            print(f"  WARNING: CLIP DOES NOT FIT IN GAP!")
        print("}")
    
    final_audio_clips = combine_clips(audio_clips,optimized_clips_by_subgap,candidates_id)
    
    # Save results
    output_file = os.path.join(scenes_folder, "audio_clips_optimized.json") 
    with open(output_file, 'w') as f:
        json.dump(final_audio_clips, f, indent=2)

    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()