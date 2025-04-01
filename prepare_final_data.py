import json
import os
import argparse

def prepare_dialogue(scene):
    """Prepare dialogue_timestamp entries from transcript using global timestamps."""
    dialogue = []
    scene_starttime = scene.get("start_time", 0)
    transcript = scene.get("transcript", [])

    for idx, line in enumerate(transcript):
        # Add the global offset to get absolute timestamps.
        start = scene_starttime + line.get("start", 0)
        end = scene_starttime + line.get("end", 0)
        duration = round(end - start, 2)

        dialogue.append({
            "sequence_num": idx + 1,
            "start_time": start,
            "end_time": end,
            "duration": duration
        })

    return dialogue

def prepare_audio_clips(scene_number, all_clips):
    """Filter and format audio clips for a given scene_number"""
    return [
        {
            "scene_number": clip["scene_number"],
            "text": clip["text"],
            "type": clip["type"],
            "start_time": clip["start_time"],   
        }
        for clip in all_clips
        if clip.get("scene_number") == scene_number
    ]

def compile_final_data(video_id):
    # Paths
    base_dir = os.path.join("videos", video_id)
    scene_dir = os.path.join(base_dir, f"{video_id}_scenes")
    scene_info_path = os.path.join(scene_dir, "scene_info.json")
    audio_clips_path = os.path.join(scene_dir, "audio_clips_optimized.json")
    metadata_path = os.path.join(base_dir, f"{video_id}.json")
    output_path = os.path.join(base_dir, "final_data.json")

    # Load files
    with open(scene_info_path, "r") as f:
        scenes = json.load(f)

    with open(audio_clips_path, "r") as f:
        all_audio_clips = json.load(f)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Collect dialogue_timestamps and audio_clips
    dialogue_timestamps = []
    audio_clips = []

    for scene in scenes:
        scene_number = scene.get("scene_number")
        dialogue_timestamps.extend(prepare_dialogue(scene))
        audio_clips.extend(prepare_audio_clips(scene_number, all_audio_clips))

    # Compile final data
    final_data = {
        "dialogue_timestamps": dialogue_timestamps,
        "audio_clips": audio_clips,
        "youtube_id": video_id,
        "video_name": metadata.get("title", ""),
        "video_length": metadata.get("video_length", 0),
        "aiUserId": "650506db3ff1c2140ea10ece" 
    }

    # Save to final_data.json
    with open(output_path, "w") as out_f:
        json.dump(final_data, out_f, indent=2, ensure_ascii=False)
    print(f"Saved final_data.json to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile dialogue timestamps and audio clips into final_data.json")
    parser.add_argument("video_id", help="YouTube video ID (e.g., dQw4w9WgXcQ)")
    args = parser.parse_args()

    compile_final_data(args.video_id)
