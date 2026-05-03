import os
import subprocess
import shutil
import torch
import logging
import sys
import json
import glob
import argparse
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("narration_bot")
logging.basicConfig(level=logging.DEBUG)

PYTHON = sys.executable
CLEANUP_AFTER_PROCESSING = os.getenv("CLEANUP_AFTER_PROCESSING", "false").lower() == "true"

DEFAULT_MODEL = "gemini"


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
        for scene in scenes:
            if "transcript" in scene:
                logger.debug(f"transcribe_scene check: Found transcript in scene {scene.get('scene_number')}")
                return True
        logger.debug("transcribe_scene check: No transcript found in any scene")
        return False
    except Exception as e:
        logger.error(f"Error reading {scene_info_path}: {e}")
        return False


def check_video_caption(video_id: str, model: str) -> bool:
    scene_info_path = os.path.join("videos", video_id, f"{video_id}_scenes", f"scene_info_{model}.json")
    if not os.path.exists(scene_info_path):
        logger.debug(f"video_caption check: {scene_info_path} does not exist")
        return False
    try:
        with open(scene_info_path, "r") as f:
            scenes = json.load(f)
        for scene in scenes:
            if "audio_clips" not in scene or scene["audio_clips"] == []:
                logger.debug(f"video_caption check: Scene {scene.get('scene_number')} is missing 'audio_clips'")
                return False
        logger.debug("video_caption check: All scenes have 'audio_clips'")
        return True
    except Exception as e:
        logger.error(f"Error reading {scene_info_path}: {e}")
        return False


def check_clip_analyze(video_id: str, model: str) -> bool:
    """clip_analyze.py outputs scene_info_{model}_filtered.json."""
    filtered_path = os.path.join("videos", video_id, f"{video_id}_scenes", f"scene_info_{model}_filtered.json")
    result = os.path.exists(filtered_path)
    logger.debug(f"Check clip_analyze for {video_id}: {result}")
    return result


def check_description_optimize_inline(video_id: str, model: str) -> bool:
    audio_clips_path = os.path.join(
        "videos", video_id, f"{video_id}_scenes", f"audio_clips_optimized_{model}.json"
    )
    result = os.path.exists(audio_clips_path)
    logger.debug(f"Check description_optimize_inline for {video_id}: {result}")
    return result


def check_final_data(video_id: str, model: str = None):
    """If model is given, check the model-specific final_data file. Otherwise glob."""
    base_dir = os.path.join("videos", video_id)
    if model:
        target = os.path.join(base_dir, f"final_data_{model}.json")
        if os.path.exists(target):
            logger.debug(f"Check final_data for {video_id}: {target}")
            return target
        logger.debug(f"No final_data_{model}.json for {video_id}")
        return None

    # Generic glob fallback for callers that don't care about model.
    pattern = os.path.join(base_dir, "final_data*.json")
    matches = glob.glob(pattern)
    if not matches:
        logger.debug(f"No final_data found for {video_id}")
        return None
    chosen = matches[0]
    logger.debug(f"Check final_data for {video_id}: {chosen}")
    return chosen


def run_pipeline(video_id: str, model: str) -> bool:
    if torch.cuda.is_available():
        logger.info("CUDA is available. Using CUDA for processing.")
        device_flag = ""
    else:
        logger.info("CUDA is not available. Using CPU for processing.")
        device_flag = "--device cpu"

    output_filename = f"audio_clips_optimized_{model}.json"

    # Workflow:
    # 1. fetch_video                → download video + metadata
    # 2. keyframe_scene_detector    → split into scenes
    # 3. transcribe_scenes          → per-scene transcript
    # 4. video_caption              → AI-generated descriptions per scene
    # 5. clip_analyze               → filter for accuracy + necessity
    # 6. description_optimize_inline → place + merge on the timeline
    # 7. prepare_final_data         → produce final_data_{model}.json
    pipeline_steps = [
        {
            "command": f"{PYTHON} fetch_video.py {video_id}",
            "check": lambda: check_youtube_downloaded(video_id),
        },
        {
            "command": f"{PYTHON} keyframe_scene_detector.py videos/{video_id} --merge_scenes --target_duration 9.0 --max_duration 15.0 {device_flag}",
            "check": lambda: check_keyframe_scene_detector(video_id),
        },
        {
            "command": f"{PYTHON} transcribe_scenes.py videos/{video_id} {device_flag}",
            "check": lambda: check_transcribe_scene(video_id),
        },
        {
            "command": f"{PYTHON} video_caption.py videos/{video_id} --model {model}",
            "check": lambda: check_video_caption(video_id, model),
        },
        {
            "command": f"{PYTHON} clip_analyze.py videos/{video_id} --model {model}",
            "check": lambda: check_clip_analyze(video_id, model),
        },
        {
            "command": (
                f"{PYTHON} description_optimize_inline.py videos/{video_id} "
                f"--optimizer_model {model} --output {output_filename}"
            ),
            "check": lambda: check_description_optimize_inline(video_id, model),
        },
        {
            "command": f"{PYTHON} prepare_final_data.py {video_id} --model {model}",
            "check": lambda: check_final_data(video_id, model) is not None,
        },
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
                text=True,
            )
        except Exception as e:
            logger.error(f"Exception when running command {cmd}: {str(e)}")
            return False

        if result.returncode != 0:
            logger.error(f"Command failed: {cmd}\nReturn code: {result.returncode}")
            return False

    if not check_final_data(video_id, model):
        logger.error(f"final_data_{model}.json was not created for video {video_id}.")
        return False

    logger.info(f"Pipeline completed successfully and final_data_{model}.json exists.")

    logger.info(f"Uploading results for {video_id} to S3...")
    upload_cmd = f"{PYTHON} upload_results.py {video_id}"
    try:
        result = subprocess.run(upload_cmd, shell=True, stdout=sys.stdout, stderr=sys.stderr, text=True)
        if result.returncode != 0:
            logger.warning(f"Results upload failed for {video_id}, but pipeline succeeded")
    except Exception as e:
        logger.warning(f"Results upload exception for {video_id}: {e}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run video processing pipeline with resume capability.")
    parser.add_argument("--video_id", required=True, help="YouTube video ID to process (e.g., dQw4w9WgXcQ)")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=["gemini", "gpt", "qwen"],
        help=f"Captioner/filter/optimizer model to use (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    if run_pipeline(args.video_id, args.model):
        print("Pipeline executed successfully.")
    else:
        print("Pipeline execution failed. Check logs for details.")
        sys.exit(1)