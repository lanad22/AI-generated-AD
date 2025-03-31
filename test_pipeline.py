import os
import subprocess
import torch
import logging
import sys
import json

logger = logging.getLogger("narration_bot")
logging.basicConfig(level=logging.DEBUG)

def check_youtube_downloaded(video_id: str) -> bool:
    output_dir = os.path.join("videos", video_id)
    video_path = os.path.join(output_dir, f"{video_id}.mp4")
    captions_path = os.path.join(output_dir, f"{video_id}.json")
    result = os.path.exists(video_path) and os.path.exists(captions_path)
    logger.debug(f"Check youtube_downloaded for {video_id}: {result}")
    return result

def check_keyframe_scene_detector(video_id: str) -> bool:
    scene_info_path = os.path.join("videos", video_id, f"{video_id}_scenes", "scene_info.json")
    result = os.path.exists(scene_info_path)
    logger.debug(f"Check keyframe_scene_detector for {video_id}: {result}")
    return result

def check_transcribe_scene(video_id: str) -> bool:
    scene_info_path = os.path.join("videos", video_id, f"{video_id}_scenes", "scene_info.json")
    if not os.path.exists(scene_info_path):
        logger.debug(f"transcribe_scene check: {scene_info_path} does not exist")
        return False
    try:
        with open(scene_info_path, "r") as f:
            scenes = json.load(f)
        # Verify at least one scene has a non-empty transcript.
        for scene in scenes:
            if "transcript" in scene:
                logger.debug(f"transcribe_scene check: Found transcript in scene {scene.get('scene_number')}")
                return True
        logger.debug("transcribe_scene check: No transcript found in any scene")
        return False
    except Exception as e:
        logger.error(f"Error reading {scene_info_path}: {e}")
        return False

def check_video_caption_api(video_id: str) -> bool:
    scene_info_path = os.path.join("videos", video_id, f"{video_id}_scenes", "scene_info.json")
    if not os.path.exists(scene_info_path):
        logger.debug(f"video_caption_api check: {scene_info_path} does not exist")
        return False
    try:
        with open(scene_info_path, "r") as f:
            scenes = json.load(f)
        # We expect each scene to include the "audio_clips" field.
        for scene in scenes:
            if "audio_clips" not in scene:
                logger.debug(f"video_caption_api check: Scene {scene.get('scene_number')} is missing 'audio_clips'")
                return False
        logger.debug("video_caption_api check: All scenes have 'audio_clips'")
        return True
    except Exception as e:
        logger.error(f"Error reading {scene_info_path}: {e}")
        return False

def check_description_optimize(video_id: str) -> bool:
    scene_dir = os.path.join("videos", video_id, f"{video_id}_scenes")
    audio_clips_opt_path = os.path.join(scene_dir, "audio_clips_optimized.json")
    result = os.path.exists(audio_clips_opt_path)
    logger.debug(f"Check description_optimize for {video_id}: {result}")
    return result

def check_final_data(video_id: str) -> bool:
    final_data_path = os.path.join("videos", video_id, "final_data.json")
    result = os.path.exists(final_data_path)
    logger.debug(f"Check final_data for {video_id}: {result}")
    return result

def run_pipeline(video_id: str) -> bool:
    # Check if CUDA is available.
    if torch.cuda.is_available():
        logger.info("CUDA is available. Using CUDA for processing.")
        device_flag = ""
    else:
        logger.info("CUDA is not available. Using CPU for processing.")
        device_flag = "--device cpu"

    # Define pipeline steps with check functions.
    pipeline_steps = [
        {
            "command": f"python youtube_downloader.py {video_id}",
            "check": lambda: check_youtube_downloaded(video_id)
        },
        {
            "command": f"python keyframe_scene_detector.py videos/{video_id} {device_flag}",
            "check": lambda: check_keyframe_scene_detector(video_id)
        },
        {
            "command": f"python transcribe_scenes.py videos/{video_id} {device_flag}",
            "check": lambda: check_transcribe_scene(video_id)
        },
        {
            "command": f"python video_caption_api.py videos/{video_id}",
            "check": lambda: check_video_caption_api(video_id)
        },
        {
            "command": f"python description_optimize.py videos/{video_id}",
            "check": lambda: check_description_optimize(video_id)
        },
        {
            "command": f"python prepare_final_data.py {video_id}",
            "check": lambda: check_final_data(video_id)
        }
    ]

    for step in pipeline_steps:
        cmd = step["command"]
        if step["check"]():
            logger.info(f"Skipping command (already done): {cmd}")
            continue

        logger.debug(f"Running command: {cmd}")
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                stdout=sys.stdout,  
                stderr=sys.stderr,
                text=True
            )
        except Exception as e:
            logger.error(f"Exception when running command {cmd}: {str(e)}")
            return False
        
        if result.returncode != 0:
            logger.error(f"Command failed: {cmd}\nReturn code: {result.returncode}")
            return False
        
    if not check_final_data(video_id):
        logger.error(f"final_data.json was not created for video {video_id}.")
        return False

    logger.debug("Pipeline completed successfully and final_data.json exists.")
    return True

# Example usage:
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run video processing pipeline with resume capability.")
    parser.add_argument("video_id", help="YouTube video ID to process (e.g., dQw4w9WgXcQ)")
    args = parser.parse_args()

    if run_pipeline(args.video_id):
        print("Pipeline executed successfully.")
    else:
        print("Pipeline execution failed. Check logs for details.")
