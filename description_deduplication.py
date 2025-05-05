import json
import os
import re
import argparse
from difflib import SequenceMatcher
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def normalize_text(text):
    """Normalize text for comparison by removing extra spaces, lowercasing, etc."""
    if not text:
        return ""
    # Remove extra whitespace, lowercase, and remove punctuation
    import re
    text = re.sub(r'\s+', ' ', text.lower().strip())
    text = re.sub(r'[^\w\s]', '', text)
    return text

def are_texts_similar(text1, text2, similarity_threshold=0.80):
    """Check if two text strings are similar enough to be considered duplicates."""
    norm_text1 = normalize_text(text1)
    norm_text2 = normalize_text(text2)
    
    similarity = SequenceMatcher(None, norm_text1, norm_text2).ratio()
    return similarity >= similarity_threshold

def has_significant_gap(scene):
    # Get scene duration information
    scene_start = scene.get('start_time', 0)
    scene_end = scene.get('end_time', 0)
    scene_duration = scene.get('duration', scene_end - scene_start)
    
    # Get transcript segments
    transcripts = scene.get('transcript', [])
    
    # If no transcripts, the entire scene is a gap
    if not transcripts:
        # Check if scene duration is significant (> 1.5 seconds)
        return scene_duration > 2.0
    
    # Sort transcripts by start time to ensure proper sequencing
    transcripts = sorted(transcripts, key=lambda x: x.get('start', 0))
    
    # Check for gap at the beginning of the scene
    first_transcript_start = transcripts[0].get('start', 0)
    if first_transcript_start > 2.0:
        return True
    
    # Check for gaps between transcripts
    for i in range(len(transcripts) - 1):
        current_end = transcripts[i].get('end', 0)
        next_start = transcripts[i + 1].get('start', 0)
        
        if next_start - current_end > 2.0:
            return True
    
    # Check for gap at the end of the scene
    last_transcript_end = transcripts[-1].get('end', 0)
    if scene_duration - last_transcript_end > 2.0:
        return True
    
    # No significant gaps found
    return False

def deduplicate_text_clips(text_clips_with_info):
    """Deduplicate text clips by comparing with clips from up to 3 scenes before."""
    if len(text_clips_with_info) <= 1:
        return text_clips_with_info, []  # No duplicates possible with 0 or 1 clips
    
    print(f"\nDeduplicating {len(text_clips_with_info)} text clips between scenes (up to 3 scenes back)...")
    
    text_clips_to_remove = []
    
    # Sort clips by scene index
    sorted_clips = sorted(text_clips_with_info, key=lambda x: x[1])
    
    # Group clips by scene
    scene_clips = {}
    for clip, scene_idx, clip_idx in sorted_clips:
        if scene_idx not in scene_clips:
            scene_clips[scene_idx] = []
        scene_clips[scene_idx].append((clip, scene_idx, clip_idx))
    
    # Get list of scene indices in order
    scene_indices = sorted(scene_clips.keys())
    
    print("\n===== TEXT CLIPS SIMILARITY ANALYSIS (UP TO 3 SCENES BACK) =====")
    
    # Compare clips between current scene and up to 3 previous scenes
    for i in range(len(scene_indices)):
        current_scene_idx = scene_indices[i]
        current_scene_clips = scene_clips[current_scene_idx]
        
        # If current scene has no clips, skip
        if not current_scene_clips:
            continue
        
        # Look back up to 3 scenes
        for j in range(1, 4):
            if i - j < 0:  # Can't go back further than the first scene
                continue
                
            prev_scene_idx = scene_indices[i - j]
            prev_scene_clips = scene_clips[prev_scene_idx]
            
            # If previous scene has no clips, skip
            if not prev_scene_clips:
                continue
                
            # Compare each text clip in current scene with each in previous scene
            for curr_clip, curr_scene_idx, curr_clip_idx in current_scene_clips:
                for prev_clip, prev_scene_idx, prev_clip_idx in prev_scene_clips:
                    if are_texts_similar(curr_clip['text'], prev_clip['text']):
                        # Choose which one to remove (usually the one from the current scene since we want to keep earlier content)
                        clip_to_remove = (curr_clip, curr_scene_idx, curr_clip_idx)
                        original_idx = text_clips_with_info.index(clip_to_remove)
                        
                        # Add to removal list with global index
                        text_clips_to_remove.append((original_idx, curr_clip, curr_scene_idx, curr_clip_idx))
                        
                        # Log what we're keeping and what we're removing
                        print(f"\nFound similar text between Scene {prev_scene_idx+1} and Scene {curr_scene_idx+1} ({j} scenes apart)")
                        print(f"KEEPING: \"{prev_clip['text'][:100]}...\" from Scene {prev_scene_idx+1}")
                        print(f"REMOVING: \"{curr_clip['text'][:100]}...\" from Scene {curr_scene_idx+1}")
                        print(f"    Reason: Similar to text in previous scene ({j} scenes back)")
                        
                        # Once we've found a match and decided to remove, break out of the inner loop
                        break
                
                # If this current clip is marked for removal, break out of comparing it with more previous clips
                if any(idx == text_clips_with_info.index((curr_clip, curr_scene_idx, curr_clip_idx)) for idx, _, _, _ in text_clips_to_remove):
                    break
    
    # Filter out the clips to remove
    kept_indices = set(range(len(text_clips_with_info))) - set(idx for idx, _, _, _ in text_clips_to_remove)
    kept_clips = [text_clips_with_info[i] for i in kept_indices]
    
    print(f"\nText deduplication complete. Removed {len(text_clips_to_remove)} clips.")
    return kept_clips, text_clips_to_remove

def deduplicate_visual_clips(client, visual_clips_with_info, scenes):
    """Deduplicate visual clips by comparing with clips from up to 3 scenes before."""
    if len(visual_clips_with_info) <= 1:
        return visual_clips_with_info, []  # No duplicates possible with 0 or 1 clips
    
    print(f"\nDeduplicating {len(visual_clips_with_info)} visual clips between scenes (up to 3 scenes back)...")
    
    visual_clips_to_remove = []
    
    # Sort clips by scene index
    sorted_clips = sorted(visual_clips_with_info, key=lambda x: x[1])
    
    # Group clips by scene
    scene_clips = {}
    for clip, scene_idx, clip_idx in sorted_clips:
        if scene_idx not in scene_clips:
            scene_clips[scene_idx] = []
        scene_clips[scene_idx].append((clip, scene_idx, clip_idx))
    
    # Get list of scene indices in order
    scene_indices = sorted(scene_clips.keys())
    
    print("\n===== VISUAL CLIPS SIMILARITY ANALYSIS (UP TO 3 SCENES BACK) =====")
    
    # Compare clips between current scene and up to 3 previous scenes
    for i in range(len(scene_indices)):
        current_scene_idx = scene_indices[i]
        current_scene_clips = scene_clips[current_scene_idx]
        
        # If current scene has no clips, skip
        if not current_scene_clips:
            continue
        
        # Check if this scene has a significant gap before comparing
        if current_scene_idx < len(scenes) and has_significant_gap(scenes[current_scene_idx]):
            print(f"Skipping comparison for Scene {current_scene_idx+1} - significant gap detected")
            continue
        
        # For each of the previous 3 scenes
        previous_scenes_to_compare = []
        for j in range(1, 4):
            if i - j < 0:  # Can't go back further than the first scene
                continue
                
            prev_scene_idx = scene_indices[i - j]
            
            # Skip if previous scene has a significant gap
            if prev_scene_idx < len(scenes) and has_significant_gap(scenes[prev_scene_idx]):
                print(f"Skipping comparison with Scene {prev_scene_idx+1} - significant gap detected")
                continue
                
            prev_scene_clips = scene_clips[prev_scene_idx]
            
            # If previous scene has no clips, skip
            if not prev_scene_clips:
                continue
                
            previous_scenes_to_compare.append((prev_scene_idx, prev_scene_clips))
        
        # If no previous scenes to compare with, continue to next scene
        if not previous_scenes_to_compare:
            continue
            
        print(f"Comparing visual clips from Scene {current_scene_idx+1} with clips from {len(previous_scenes_to_compare)} previous scenes")
        
        # Prepare data for model comparison
        comparison_data = []
        id_map = {}  # Maps local_id back to (clip, scene_idx, clip_idx)
        
        # Add all current scene clips
        for local_id, (clip, scene_idx, clip_idx) in enumerate(current_scene_clips):
            comparison_data.append({
                "id": local_id,
                "text": clip['text'],
                "type": clip['type'],
                "scene": scene_idx + 1,
                "start_time": clip.get('start_time', 0),
                "clip_index_in_scene": clip_idx
            })
            id_map[local_id] = (clip, scene_idx, clip_idx)
        
        # Add all clips from previous scenes (continue the ID numbering)
        next_id = len(current_scene_clips)
        for prev_scene_idx, prev_scene_clips in previous_scenes_to_compare:
            for local_id_offset, (clip, scene_idx, clip_idx) in enumerate(prev_scene_clips):
                local_id = next_id + local_id_offset
                comparison_data.append({
                    "id": local_id,
                    "text": clip['text'],
                    "type": clip['type'],
                    "scene": scene_idx + 1,
                    "start_time": clip.get('start_time', 0),
                    "clip_index_in_scene": clip_idx
                })
                id_map[local_id] = (clip, scene_idx, clip_idx)
            next_id += len(prev_scene_clips)
        
        # Create the prompt for visual clips comparison
        current_scene_clip_ids = list(range(0, len(current_scene_clips)))
        previous_scenes_clip_ids = list(range(len(current_scene_clips), len(comparison_data)))
        
        prompt = f"""You are analyzing visual description clips to identify redundant content between scenes.

        TASK:
        Review visual descriptions from Scene {current_scene_idx+1} and up to 3 previous scenes, and identify ONLY pairs of clips that describe the EXACT SAME visual content or object.

        CLIPS:
        {json.dumps(comparison_data, indent=2)}

        INSTRUCTIONS:
        1. Clips with IDs {current_scene_clip_ids[0]} to {current_scene_clip_ids[-1]} are from the current Scene {current_scene_idx+1}.
        2. Clips with IDs {previous_scenes_clip_ids[0]} to {previous_scenes_clip_ids[-1]} are from up to 3 previous scenes.
        3. Compare clips between the current scene and previous scenes to find TRULY redundant descriptions of the EXACT SAME visual content.
        4. CRITICAL: Only mark clips as redundant if they describe the EXACT SAME object or visual element.
        5. DO NOT mark clips as redundant if they:
           - Describe similar but different objects (e.g., "a red car" vs "another red car")
           - Describe the same type of object but in different contexts or locations
           - Describe different aspects of a similar scene
           - Contain unique details not present in the other description
        6. For each pair of truly redundant clips (describing the exact same object), decide which ONE to keep - typically the more detailed and informative one.
        7. Return a JSON array with objects representing each pair of clips that describe the EXACT SAME object.

        OUTPUT FORMAT:
        Provide a JSON array of objects with "keep_id", "remove_id", and "reason" fields:
        [
          {{
            "keep_id": 12,
            "remove_id": 2,
            "reason": "Both clips describe exactly the same tree in the foreground with identical details about its trunk and branches - keeping the more detailed description."
          }}
        ]
        """
        
        try:
            response = client.chat.completions.create(
                model="qwen2.5-72b-instruct",
                messages=[
                    {"role": "system", "content": "You are a specialist in identifying truly redundant content in visual descriptions. You only mark descriptions as redundant if they describe the EXACT SAME object or visual element with high confidence. You prefer to keep unique content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,  # Lower temperature for more consistent, conservative matching
                max_tokens=1000
            )
            
            # Get the model's response
            result = response.choices[0].message.content.strip()
            
            # Parse the response
            try:
                # Extract JSON array from response
                matches = re.search(r'\[.*\]', result, re.DOTALL)
                if matches:
                    dedup_results = json.loads(matches.group(0))
                    
                    # Process each pair
                    for pair in dedup_results:
                        keep_id = pair.get("keep_id")
                        remove_id = pair.get("remove_id")
                        reason = pair.get("reason", "No reason provided")
                        
                        if keep_id is None or remove_id is None:
                            continue
                        
                        # Validate IDs are in range
                        if not (0 <= keep_id < len(id_map)) or not (0 <= remove_id < len(id_map)):
                            print(f"Warning: Invalid ID in pair {keep_id}/{remove_id}, skipping")
                            continue
                        
                        # Get the clips details
                        keeper_clip, keeper_scene_idx, keeper_clip_idx = id_map[keep_id]
                        remove_clip, remove_scene_idx, remove_clip_idx = id_map[remove_id]
                        
                        # Verify that we're comparing current scene with previous scene
                        if not ((remove_scene_idx == current_scene_idx and keeper_scene_idx != current_scene_idx) or 
                               (keeper_scene_idx == current_scene_idx and remove_scene_idx != current_scene_idx)):
                            print(f"Warning: Not comparing current scene with previous scene, skipping")
                            continue
                        
                        # Find the original global index
                        original_idx = visual_clips_with_info.index((remove_clip, remove_scene_idx, remove_clip_idx))
                        
                        # Add to removal list with global index
                        visual_clips_to_remove.append((original_idx, remove_clip, remove_scene_idx, remove_clip_idx))
                        
                        # Log the decision
                        keeper_text = keeper_clip['text'][:100]
                        remove_text = remove_clip['text'][:100]
                        
                        print(f"\nFound redundant visual content between Scene {keeper_scene_idx+1} and Scene {remove_scene_idx+1}")
                        print(f"KEEPING: \"{keeper_text}...\" from Scene {keeper_scene_idx+1}")
                        print(f"REMOVING: \"{remove_text}...\" from Scene {remove_scene_idx+1}")
                        print(f"    Reason: {reason}")
                
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Failed to parse model response for scene {current_scene_idx+1}: {e}")
                
        except Exception as e:
            print(f"Error during visual clip comparison for scene {current_scene_idx+1}: {e}")
    
    # Filter out the clips to remove
    kept_indices = set(range(len(visual_clips_with_info))) - set(idx for idx, _, _, _ in visual_clips_to_remove)
    kept_clips = [visual_clips_with_info[i] for i in kept_indices]
    
    print(f"\nVisual deduplication complete. Removed {len(visual_clips_to_remove)} clips.")
    return kept_clips, visual_clips_to_remove

def process_video_folder(client, video_folder):
    """Process a video folder to globally deduplicate audio clips."""
    video_id = os.path.basename(os.path.normpath(video_folder))
    scenes_folder = os.path.join(video_folder, f"{video_id}_scenes")
    scenes_json_path = os.path.join(scenes_folder, "scene_info.json")
    
    if not os.path.exists(scenes_json_path):
        print(f"Error: scene_info.json not found in {scenes_folder}")
        return
    
    print(f"Processing {scenes_json_path}...")
    with open(scenes_json_path, "r") as f:
        scenes = json.load(f)
    
    total_scenes = len(scenes)
    
    # Global collections for all clips
    all_text_clips = []  # (clip, scene_index, clip_index)
    all_visual_clips = []  # (clip, scene_index, clip_index)
    
    # Collect all clips
    for scene_idx, scene in enumerate(scenes):
        scene_number = scene.get('scene_number', scene_idx+1)
        print(f"Collecting clips from Scene {scene_number}...")
        
        audio_clips = scene.get('audio_clips', [])
        if not audio_clips:
            continue
            
        # Collect text clips from all scenes
        for clip_idx, clip in enumerate(audio_clips):
            if clip.get('type') == 'Text on Screen':
                all_text_clips.append((clip, scene_idx, clip_idx))
            
            elif clip.get('type') == 'Visual': 
                all_visual_clips.append((clip, scene_idx, clip_idx))

    print(f"\nCollected {len(all_text_clips)} text clips from all scenes")
    print(f"Collected {len(all_visual_clips)} visual clips from scenes")
    
    # Global deduplication
    kept_text_clips, removed_text_clips = deduplicate_text_clips(all_text_clips)
    kept_visual_clips, removed_visual_clips = deduplicate_visual_clips(client, all_visual_clips, scenes)
    
    # Prepare a map of clips to remove
    clips_to_remove = {}  # {scene_idx: {clip_idx: True}}
    
    for _, _, scene_idx, clip_idx in removed_text_clips + removed_visual_clips:
        if scene_idx not in clips_to_remove:
            clips_to_remove[scene_idx] = {}
        clips_to_remove[scene_idx][clip_idx] = True
    
    # Update scenes by removing the redundant clips
    for scene_idx, scene in enumerate(scenes):
        if scene_idx in clips_to_remove:
            # Get original clips
            original_clips = scene.get('audio_clips', [])
            # Filter out removed clips
            updated_clips = [
                clip for clip_idx, clip in enumerate(original_clips)
                if clip_idx not in clips_to_remove[scene_idx]
            ]
            # Update scene
            scene['audio_clips'] = updated_clips
            print(f"Updated Scene {scene.get('scene_number', scene_idx+1)}: Removed {len(original_clips) - len(updated_clips)} clips")
    
    # Prepare full report of removed clips
    all_removed_clips = []
    for _, clip, scene_idx, _ in removed_text_clips + removed_visual_clips:
        clip_copy = clip.copy()
        clip_copy['scene_number'] = scenes[scene_idx].get('scene_number', scene_idx+1)
        all_removed_clips.append(clip_copy)
    
    # Save deduplicated scenes
    output_path = os.path.join(scenes_folder, "scene_info_deduped.json")
    with open(output_path, "w") as f:
        json.dump(scenes, f, indent=2)
    
    print(f"\n===== GLOBAL DEDUPLICATION SUMMARY =====")
    print(f"Total scenes processed: {total_scenes}")
    print(f"Original text clips: {len(all_text_clips)}")
    print(f"Kept text clips: {len(kept_text_clips)}")
    print(f"Removed text clips: {len(removed_text_clips)}")
    print(f"Original visual clips: {len(all_visual_clips)}")
    print(f"Kept visual clips: {len(kept_visual_clips)}")
    print(f"Removed visual clips: {len(removed_visual_clips)}")
    print(f"Total removed clips: {len(removed_text_clips) + len(removed_visual_clips)}")
    print(f"\nDeduplicated scenes saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Globally deduplicate audio descriptions")
    parser.add_argument("video_folder", help="Path to the video folder containing scene_info.json")
    
    args = parser.parse_args()
    
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
    
    # Process the video folder
    process_video_folder(client, args.video_folder)

if __name__ == "__main__":
    main()