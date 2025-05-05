import json
import tempfile
import subprocess
import os
import re
import argparse
from gtts import gTTS
from typing import List, Dict
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

def evaluate_clip_necessity(client, clip, transcript_data, previous_descriptions):
    scene_number = clip.get('scene_number', 0)
    scene_transcript = [t for t in transcript_data if t.get('scene_number') == scene_number]
    
    # Get current scene transcript
    transcript_text = ""
    for segment in scene_transcript:
        transcript_text += f"{segment.get('text', '')} "
    
    # Get cumulative transcript up to the current scene
    cumulative_transcript = ""
    for segment in transcript_data:
        if segment.get('scene_number') <= scene_number:
            cumulative_transcript += f"{segment.get('text', '')} "
    
    previous_desc_text = ""
    for desc in previous_descriptions:
        previous_desc_text += f"[Scene {desc.get('scene_number')}] {desc.get('type')}: {desc.get('text')} "
    
    prompt = f"""
            You are an accessibility expert selecting ONE visual description per scene to convert to audio description for blind and low-vision users.

            ### CONTEXT
            - IMPORTANT: You must select ONLY ONE description per scene - the most important one
            - Audio descriptions interrupt the natural flow of content and should be MINIMAL
            - The video's spoken audio (transcript) is the primary source of information
            - Audio descriptions should be used SPARINGLY - only for truly critical visual information

            ### INPUT
            CURRENT SCENE TRANSCRIPT:
            {transcript_text}

            CUMULATIVE TRANSCRIPT SO FAR:
            {cumulative_transcript}
            
            CUMULATIVE DESCRIPTION SO FAR:
            {previous_desc_text}

            VISUAL DESCRIPTIONS TO EVALUATE:
            {clip['text']}

            ### EVALUATION CRITERIA
            Include this visual description (necessary = true) if it meets **any** one of these essential conditions:
            - Important Visual Information: Conveys visual details not in the audio (new actions, expressions, settings).
            - Unspoken Actions & Key Events: Describes important silent actions or events (e.g. a character’s meaningful gesture, a key object movement).
            - Scene Context & Characters: Identifies who or where when audio alone is ambiguous (e.g. new character entry, location change).
            - Novelty & Variation: Introduces a distinct visual element or scene detail that has not been described before (e.g. a flowing stream, a perched butterfly).
            - Scene Changes & Time Jumps: Notes unannounced transitions (e.g. “cut to: a hospital corridor, later that night”).

            ### OUTPUT FORMAT
            Return a JSON array where ONLY ONE item at most has "necessary": true:
            - "id": Index (0-based) of the description.
            - "necessary": true or false (only ONE can be true).
            - "reason": "Clear explanation of why this was selected as the most important description or why none were necessary"
            """

    try:
        response = client.chat.completions.create(
            model="qwen2.5-72b-instruct",
            messages=[
                {"role": "system", "content": "You are expert audio describer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        result = response.choices[0].message.content.strip()
        print(f"MODEL RESPONSE: {result}")  # Print the raw model response
        
        try:
            matches = re.search(r'\{.*\}', result, re.DOTALL)
            if matches:
                analysis = json.loads(matches.group(0))
                return analysis.get('necessary', False), analysis.get('reason', "No reason provided")
            else:
                print(f"Could not find JSON in response for clip in scene {scene_number}.")
                return False, "Failed to parse model response"
        except json.JSONDecodeError:
            print(f"Failed to parse JSON for clip in scene {scene_number}.")
            return False, "Failed to parse model response"
            
    except Exception as e:
        print(f"Error evaluating necessity: {e}")
        return False, f"Error: {str(e)}"

def optimize_description(client, clip):
    if not clip:
        return None
    
    prompt = f"""
            TASK: Create an extremely concise version of this visual description for an audio description track.

            ORIGINAL DESCRIPTION:
            {clip['text']}

            GUIDELINES:
            - Focus ONLY on the most essential visual elements
            - Make it significantly more concise while keeping the most critical information
            - Use natural, conversational language that won't disrupt the overall experience
            - Use clear, vivid language suitable for audio description
            - Maintain a flowing sentence structure that sounds good when read aloud
            - Start with the most important element of the scene
            - Be extremely concise - every word must earn its place

            OUTPUT:
            Provide only the optimized description."""
    
    try:
        response = client.chat.completions.create(
            model="qwen2.5-72b-instruct",
            messages=[
                {"role": "system", "content": "You are an expert audio describer."},
                {"role": "user", "content": prompt}
            ],
            temperature=1.0,
            max_tokens=100
        )
        
        optimized_text = response.choices[0].message.content.strip()
        
        # Calculate TTS duration
        tts_duration = get_tts_duration(optimized_text)
        
        optimized_clip = clip.copy()
        optimized_clip['text'] = optimized_text
        optimized_clip['duration'] = tts_duration
        optimized_clip['end_time'] = clip['start_time'] + tts_duration
        optimized_clip['original_text'] = clip['text']
        
        return optimized_clip
        
    except Exception as e:
        print(f"Error optimizing clip: {e}")
        return clip  

def main():
    parser = argparse.ArgumentParser(description="Analyze and optimize visual descriptions for accessibility")
    parser.add_argument("video_folder", help="Path to the video folder containing scene_info.json and audio_clips_optimized.json")
    parser.add_argument("--no-analyze-necessity", action="store_true", 
                        help="Skip analyzing whether descriptions are necessary (default is to analyze)")
    
    args = parser.parse_args()
    
    video_id = os.path.basename(os.path.normpath(args.video_folder))  
    scenes_folder = os.path.join(args.video_folder, f"{video_id}_scenes")
    scene_info_path = os.path.join(scenes_folder, "scene_info.json")
    audio_clips_path = os.path.join(scenes_folder, "audio_clips_optimized.json")
    
    if not os.path.exists(scene_info_path):
        print(f"Error: {scene_info_path} not found")
        return
    
    if not os.path.exists(audio_clips_path):
        print(f"Error: {audio_clips_path} not found")
        return
    
    with open(scene_info_path, "r") as f:
        scene_info = json.load(f)
    
    with open(audio_clips_path, "r") as f:
        audio_clips = json.load(f)
    
    transcript_data = []
    for scene in scene_info:
        scene_number = scene.get('scene_number', 0)
        for segment in scene.get('transcript', []):
            transcript_segment = segment.copy()
            transcript_segment['scene_number'] = scene_number
            transcript_data.append(transcript_segment)
    
    print(f"Loaded transcript with {len(transcript_data)} segments")
    print(f"Loaded {len(audio_clips)} descriptions from audio_clips_optimized.json")
    
    # Setup the OpenAI client for DashScope API
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
    
    # Filter non-gap visual descriptions
    non_gap_visuals = [desc for desc in audio_clips 
                      if desc.get('fits_in_gap', True) is False 
                      and desc.get('type') == 'Visual']
    
    print(f"\nFound {len(non_gap_visuals)} Visual descriptions where fits_in_gap is false")
    
    if not non_gap_visuals:
        print("No Visual descriptions with fits_in_gap=false to process.")
        return
    
    audio_clips.sort(key=lambda x: (x.get('scene_number', 0), x.get('start_time', 0)))
    
    final_clips = []
    clips_kept = 0
    clips_removed = 0
    
    previous_descriptions = []
    
    for clip in audio_clips:
        if clip.get('type') != 'Visual' or clip.get('fits_in_gap', True):
            final_clips.append(clip)
            previous_descriptions.append(clip)
            continue
        
        if not args.no_analyze_necessity:
            print(f"\n===== EVALUATING CLIP IN SCENE {clip['scene_number']} =====")
            print(f"Description: \"{clip['text']}\"")
            
            is_necessary, reason = evaluate_clip_necessity(
                client, clip, transcript_data, previous_descriptions)
            
            if is_necessary:
                clips_kept += 1
                # Optimize the necessary clip
                optimized_clip = optimize_description(client, clip)
                if optimized_clip:
                    print(f"Original ({len(clip['text'])} chars): {clip['text']}")
                    print(f"Optimized ({len(optimized_clip['text'])} chars): {optimized_clip['text']}")
                    final_clips.append(optimized_clip)
                    previous_descriptions.append(optimized_clip)
                else:
                    final_clips.append(clip)
                    previous_descriptions.append(clip)
            else:
                clips_removed += 1
                # Not adding to final_clips if unnecessary
        else:
            print(f"\nOptimizing clip in scene {clip['scene_number']}:")
            optimized_clip = optimize_description(client, clip)
            if optimized_clip:
                print(f"Original ({len(clip['text'])} chars): {clip['text']}")
                print(f"Optimized ({len(optimized_clip['text'])} chars): {optimized_clip['text']}")
                final_clips.append(optimized_clip)
                previous_descriptions.append(optimized_clip)
            else:
                final_clips.append(clip)
                previous_descriptions.append(clip)
    

    final_clips.sort(key=lambda x: (x.get('scene_number', 0), x.get('start_time', 0)))

    with open(audio_clips_path, 'w') as f:
        json.dump(final_clips, f, indent=2)

    print(f"\nResults saved back to: {audio_clips_path}")
    print(f"Final output: {len(final_clips)} clips total")
    
    if not args.no_analyze_necessity:
        print(f"Non-gap visual descriptions: {len(non_gap_visuals)} evaluated")
        print(f"- {clips_kept} kept and optimized")
        print(f"- {clips_removed} removed")

if __name__ == "__main__":
    main()