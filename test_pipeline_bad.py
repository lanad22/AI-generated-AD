"""
INTENTIONALLY-BAD variant of test_pipeline.py.

Produces a deliberately worse audio-description track for A/B comparison against
the normal pipeline. Differences from test_pipeline.py:

  1. Captioning uses video_caption_bad.py (all AD quality/accuracy guidance
     stripped from the prompt) instead of video_caption.py.
  2. clip_analyze.py (accuracy + necessity filtering) is SKIPPED.
  3. description_optimize_inline.py (gap placement, merging, compression) is
     SKIPPED and replaced by flatten_clips_bad.py, which only converts clips to
     absolute time so a final_data file can still be built.

Every output carries `bad` in its name so the good pipeline's files
(scene_info_{model}.json, audio_clips_optimized_{model}.json,
final_data_{model}.json, ...) are never overwritten. Intermediates use the
scene_info_{model}_bad.json form; the final, forwarded file is final_data_bad.json.

The shared, model-independent early steps (fetch_video, keyframe_scene_detector,
transcribe_scenes) reuse the existing outputs when present.
"""

import os
import subprocess
import torch
import logging
import sys
import argparse
from dotenv import load_dotenv

# Reuse the shared, model-independent stage checks from the good pipeline.
from test_pipeline import (
    check_youtube_downloaded,
    check_keyframe_scene_detector,
    check_transcribe_scene,
)

load_dotenv()

logger = logging.getLogger("narration_bot")
logging.basicConfig(level=logging.DEBUG)

PYTHON = sys.executable
DEFAULT_MODEL = "gemini"

# Dedicated AI user id for the intentionally-bad description track. Baked into
# final_data_{model}_bad.json so the backend forwards it under this identity.
BAD_AI_USER_ID = "693f3ef0864bfcde60dd9b52"


def _scenes_dir(video_id: str) -> str:
    return os.path.join("videos", video_id, f"{video_id}_scenes")


def check_video_caption_bad(video_id: str, model: str) -> bool:
    path = os.path.join(_scenes_dir(video_id), f"scene_info_{model}_bad.json")
    result = os.path.exists(path)
    logger.debug(f"Check video_caption_bad for {video_id}: {result}")
    return result


def check_clip_dedup_bad(video_id: str, model: str) -> bool:
    path = os.path.join(_scenes_dir(video_id), f"scene_info_{model}_bad_deduped.json")
    result = os.path.exists(path)
    logger.debug(f"Check clip_dedup (bad) for {video_id}: {result}")
    return result


def check_flatten_bad(video_id: str, model: str) -> bool:
    path = os.path.join(_scenes_dir(video_id), f"audio_clips_optimized_{model}_bad.json")
    result = os.path.exists(path)
    logger.debug(f"Check flatten (bad) for {video_id}: {result}")
    return result


def check_final_data_bad(video_id: str, model: str) -> bool:
    # The bad track's final file is model-independent: final_data_bad.json.
    path = os.path.join("videos", video_id, "final_data_bad.json")
    result = os.path.exists(path)
    logger.debug(f"Check final_data (bad) for {video_id}: {result}")
    return result


def run_pipeline(video_id: str, model: str) -> bool:
    if torch.cuda.is_available():
        logger.info("CUDA is available. Using CUDA for processing.")
        device_flag = ""
    else:
        logger.info("CUDA is not available. Using CPU for processing.")
        device_flag = "--device cpu"

    scenes_dir = _scenes_dir(video_id)
    caption_out = os.path.join(scenes_dir, f"scene_info_{model}_bad.json")
    deduped_out = os.path.join(scenes_dir, f"scene_info_{model}_bad_deduped.json")
    flat_out = os.path.join(scenes_dir, f"audio_clips_optimized_{model}_bad.json")
    final_out = os.path.join("videos", video_id, "final_data_bad.json")

    # Workflow (BAD variant):
    # 1. fetch_video                → download video + metadata          [shared]
    # 2. keyframe_scene_detector    → split into scenes                  [shared]
    # 3. transcribe_scenes          → per-scene transcript               [shared]
    # 4. video_caption_bad          → LOW-QUALITY descriptions per scene
    # 5. clip_dedup                 → resolve same-start-time redundancy
    # -- clip_analyze                 SKIPPED (no accuracy/necessity filter)
    # -- description_optimize_inline  SKIPPED (no placement/merge/compress)
    # 6. flatten_clips_bad          → flat, absolute-time clip list (bridge)
    # 7. prepare_final_data         → produce final_data_bad.json (BAD_AI_USER_ID)
    pipeline_steps = [
        {
            "command": f"{PYTHON} fetch_video.py -- {video_id}",
            "check": lambda: check_youtube_downloaded(video_id),
        },
        {
            "command": f"{PYTHON} keyframe_scene_detector.py videos/{video_id} --merge_scenes --target_duration 9.0 --max_duration 25.0 {device_flag}",
            "check": lambda: check_keyframe_scene_detector(video_id),
        },
        {
            "command": f"{PYTHON} transcribe_scenes.py videos/{video_id} {device_flag}",
            "check": lambda: check_transcribe_scene(video_id),
        },
        {
            "command": f"{PYTHON} video_caption_bad.py videos/{video_id} --model {model}",
            "check": lambda: check_video_caption_bad(video_id, model),
        },
        {
            "command": (
                f"{PYTHON} clip_dedup.py videos/{video_id} --model {model} "
                f"--input {caption_out} --output {deduped_out}"
            ),
            "check": lambda: check_clip_dedup_bad(video_id, model),
        },
        {
            "command": (
                f"{PYTHON} flatten_clips_bad.py {deduped_out} --output {flat_out}"
            ),
            "check": lambda: check_flatten_bad(video_id, model),
        },
        {
            "command": (
                f"{PYTHON} prepare_final_data.py --model {model} "
                f"--audio_clips_path {flat_out} --output_path {final_out} "
                f"--ai_user_id {BAD_AI_USER_ID} -- {video_id}"
            ),
            "check": lambda: check_final_data_bad(video_id, model),
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

    if not check_final_data_bad(video_id, model):
        logger.error(f"final_data_bad.json was not created for video {video_id}.")
        return False

    logger.info(f"Bad pipeline completed successfully. {final_out} exists.")
    logger.info("NOTE: S3 upload is intentionally skipped for the bad pipeline "
                "so experimental results are not published.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the INTENTIONALLY-BAD video processing pipeline (bad-prefixed outputs)."
    )
    parser.add_argument("--video_id", required=True, help="YouTube video ID to process (e.g., dQw4w9WgXcQ)")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=["gemini", "gpt", "qwen"],
        help=f"Captioner/dedup model to use (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    if run_pipeline(args.video_id, args.model):
        print("Bad pipeline executed successfully.")
    else:
        print("Bad pipeline execution failed. Check logs for details.")
        sys.exit(1)
