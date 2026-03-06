import json
import os
import argparse

def prepare_dialogue(scenes):
    dialogue = []
    sequence_counter = 1
    last_dialogue_end = None
    continuing_dialogue = False
    
    for scene in scenes:
        scene_starttime = scene.get("start_time", 0)
        transcript = scene.get("transcript", [])
        
        for line in transcript:
            start = scene_starttime + line.get("start", 0)
            end = scene_starttime + line.get("end", 0)
            text = line.get("text", "")
            
            gap_threshold = 0.1
            
            if last_dialogue_end is not None and abs(start - last_dialogue_end) < gap_threshold:
                if dialogue and continuing_dialogue:
                    dialogue[-1]["end_time"] = end - 0.1
                    dialogue[-1]["duration"] = round(dialogue[-1]["end_time"] - dialogue[-1]["start_time"], 2)
                    continuing_dialogue = False
                    last_dialogue_end = end
                    continue
            
            duration = round(end - start, 2)
            dialogue.append({
                "start_time": start,
                "end_time": end - 0.1,
                "duration": duration,
                "audio_text": text,
                "sequence_num": sequence_counter
            })
            sequence_counter += 1
            
            if end >= scene.get("end_time", 0) - gap_threshold:
                continuing_dialogue = True
            else:
                continuing_dialogue = False
                
            last_dialogue_end = end
            
    return dialogue

def check_interference(clip_start, clip_end, existing_intervals):
    for interval in existing_intervals:
        interval_start = interval["start_time"]
        if "end_time" in interval and interval["end_time"]:
            interval_end = interval["end_time"]
        else:
            interval_end = interval["tts_duration"] + interval_start
        
        if max(clip_start, interval_start) < min(clip_end, interval_end):
            return True
    return False

def prepare_audio_clips(scene_number, all_clips, dialogue_timestamps):
    prepared_clips = []
    scene_clips = [clip for clip in all_clips if clip.get("scene_number") == scene_number]
    scene_clips.sort(key=lambda x: x["start_time"])

    for i, clip in enumerate(scene_clips):
        clip_start = clip["start_time"]
        if "end_time" in clip and clip["end_time"]:
            clip_end = clip["end_time"]
        else:
            clip_end = clip["tts_duration"] + clip_start

        track_type = "inline"
        if check_interference(clip_start, clip_end, dialogue_timestamps):
            track_type = "extended"
        
        other_clips_in_scene = [
            ac for j, ac in enumerate(scene_clips) if i != j
        ]
        
        if track_type == "inline" and check_interference(clip_start, clip_end, other_clips_in_scene):
            track_type = "extended"
        
        if track_type == "extended":
            clip_end = clip_start

        prepared_clips.append({
            "scene_number": clip["scene_number"],
            "text": clip["text"],
            "type": clip["type"],
            "start_time": clip_start,
            "end_time": clip_end,
            "track_type": track_type 
        })
    return prepared_clips

def compile_final_data(video_id, model_choice):
    base_dir = os.path.join("videos", video_id)
    scene_dir = os.path.join(base_dir, f"{video_id}_scenes")
    scene_info_path = os.path.join(scene_dir, "scene_info.json")
    metadata_path = os.path.join(base_dir, f"{video_id}.json")
    output_path = ""

    if model_choice == "gemini":
        ai_user_id = "6845e4375506faa0752b8d62"
        audio_clips_filename = "audio_clips_optimized_gemini.json"
        output_path = os.path.join(base_dir, "final_data_gemini.json")
    elif model_choice == "qwen":
        ai_user_id = "68798f57c48a173631902319"
        audio_clips_filename = "audio_clips_optimized_qwen.json"
        output_path = os.path.join(base_dir, "final_data_qwen.json")
        
    elif model_choice == "gpt":
        ai_user_id = "650506db3ff1c2140ea10ece"
        audio_clips_filename = "audio_clips_optimized_gpt.json"
        output_path = os.path.join(base_dir, "final_data_gpt.json")
        
    else:
        raise ValueError("Invalid model choice provided.")

    audio_clips_path = os.path.join(scene_dir, audio_clips_filename)
    for path in [scene_info_path, audio_clips_path, metadata_path]:
        if not os.path.exists(path):
            print(f"Error: Required file not found at {path}")
            return

    with open(scene_info_path, "r") as f:
        scenes = json.load(f)

    with open(audio_clips_path, "r") as f:
        all_audio_clips = json.load(f)
    print(f"Successfully loaded audio clips from: {audio_clips_path}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    dialogue_timestamps = prepare_dialogue(scenes)
    
    audio_clips = []
    for scene in scenes:
        audio_clips.extend(prepare_audio_clips(scene.get("scene_number"), all_audio_clips, dialogue_timestamps))

    final_data = {
        "dialogue_timestamps": dialogue_timestamps,
        "audio_clips": audio_clips,
        "youtube_id": video_id,
        "video_name": metadata.get("title", ""),
        "video_length": metadata.get("video_length", 0),
        "aiUserId": ai_user_id 
    }

    # Save to final_data.json
    with open(output_path, "w", encoding='utf-8') as out_f:
        json.dump(final_data, out_f, indent=2, ensure_ascii=False)
    print(f"Saved final_data.json to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile dialogue timestamps and audio clips into final_data.json")
    parser.add_argument("video_id", help="YouTube video ID (e.g., dQw4w9WgXcQ)")
    parser.add_argument("--model", type=str, choices=["gemini", "qwen", "gpt"], default="gpt",
                        help="The model used to generate the audio clips, which determines the input file and AI User ID.")
    args = parser.parse_args()

    compile_final_data(args.video_id, args.model)