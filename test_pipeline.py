"""
Main video processing pipeline: download (or fetch from S3) -> keyframes -> transcribe -> caption -> optimize -> final_data.
Invoked by server.py for AI description generation, or run directly via CLI.
"""
import os
import subprocess
import torch
import logging
import sys
import json
import glob

from dotenv import load_dotenv
load_dotenv()

try:
    from config import S3_VIDEO_BUCKET as DEFAULT_S3_BUCKET
except ImportError:
    DEFAULT_S3_BUCKET = os.getenv("S3_VIDEO_BUCKET", "youdescribe-downloaded-youtube-videos")

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

def check_video_caption(video_id: str) -> bool:
    scene_info_path = os.path.join("videos", video_id, f"{video_id}_scenes", "scene_info_gpt.json")
    if not os.path.exists(scene_info_path):
        logger.debug(f"video_caption check: {scene_info_path} does not exist")
        return False
    try:
        with open(scene_info_path, "r") as f:
            scenes = json.load(f)
        # We expect each scene to include the "audio_clips" field.
        for scene in scenes:
            if "audio_clips" not in scene or scene["audio_clips"] == []:
                logger.debug(f"video_caption check: Scene {scene.get('scene_number')} is missing 'audio_clips'")
                return False
        logger.debug("video_caption check: All scenes have 'audio_clips'")
        return True
    except Exception as e:
        logger.error(f"Error reading {scene_info_path}: {e}")
        return False

def check_description_optimize(video_id: str) -> bool:
    scene_dir = os.path.join("videos", video_id, f"{video_id}_scenes")
    audio_clips_opt_path = os.path.join(scene_dir, "audio_clips_optimized_gpt.json")
    result = os.path.exists(audio_clips_opt_path)
    logger.debug(f"Check description_optimize for {video_id}: {result}")
    return result

def check_description_optimize_combined(video_id: str) -> bool:
    scene_dir = os.path.join("videos", video_id, f"{video_id}_scenes")
    audio_clips_combined_path = os.path.join(scene_dir, "audio_clips_optimized_gpt.json")
    result = os.path.exists(audio_clips_combined_path)
    logger.debug(f"Check description_optimize_combined for {video_id}: {result}")
    return result

def check_final_data(video_id: str) -> bool:
    base_dir = os.path.join("videos", video_id)
    final_data_path = os.path.join(base_dir, "final_data*.json")
    matches = glob.glob(final_data_path)
    if not matches:
        logger.debug(f"No final_data found for {video_id}")
        return None
    chosen = matches[0]
    
    logger.debug(f"Check final_data for {video_id}: {chosen}")
    return chosen

def get_video_category(video_id: str) -> str:
    metadata_path = os.path.join("videos", video_id, f"{video_id}.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                # YouTube API typically stores category in 'categoryId' or 'category'
                category_id = metadata.get('categoryId', '')
                category_name = metadata.get('category', '').lower()
                
                # Map common category IDs to names (YouTube's category mapping)
                category_mapping = {
                    '26': 'howto & style',
                    '27': 'education',
                    '22': 'people & blogs',
                    '24': 'entertainment'
                }
                
                if category_id in category_mapping:
                    return category_mapping[category_id]
                elif category_name:
                    return category_name
                    
        except Exception as e:
            logger.warning(f"Error reading video metadata: {e}")
    
    return 'unknown'

def analyze_text_on_screen_density(video_id: str) -> bool:
    scene_info_path = os.path.join("videos", video_id, f"{video_id}_scenes", "scene_info_gpt.json")
    
    if not os.path.exists(scene_info_path):
        logger.warning(f"scene_info_gpt.json not found for video {video_id}")
        return False
    
    try:
        with open(scene_info_path, 'r') as f:
            scenes = json.load(f)
            
        total_clips = 0
        text_on_screen_clips = 0
        unique_text_entries = set()
        
        for scene in scenes:
            audio_clips = scene.get('audio_clips', [])
            
            for clip in audio_clips:
                total_clips += 1
                
                if clip.get('type') == 'Text on Screen':
                    text_on_screen_clips += 1
                    text_content = clip.get('text', '').strip()
                    
                    if text_content:
                        # Track unique text entries to avoid counting duplicates
                        unique_text_entries.add(text_content)
        
        if total_clips == 0:
            return False
        
        # Calculate metrics
        text_on_screen_ratio = text_on_screen_clips / total_clips
        unique_text_count = len(unique_text_entries)
        
        logger.info(f"Video {video_id} text analysis:")
        logger.info(f"  - Text on screen ratio: {text_on_screen_ratio:.2%} ({text_on_screen_clips}/{total_clips})")
        logger.info(f"  - Unique text entries: {unique_text_count}")
        is_high_density = text_on_screen_ratio > 0.3 and unique_text_count >= 5
        
        logger.info(f"  - High text density: {is_high_density}")
        return is_high_density
        
    except Exception as e:
        logger.warning(f"Error analyzing text density for {video_id}: {e}")
        return False

def should_use_combined_optimization(video_id: str) -> bool:
    category = get_video_category(video_id).lower()
    is_howto_style = any(keyword in category for keyword in ['howto', 'style'])
    
    has_high_text_density = analyze_text_on_screen_density(video_id)
    
    logger.info(f"Video {video_id} optimization decision:")
    logger.info(f"  - Category: {category}")
    logger.info(f"  - Is howto/style/education: {is_howto_style}")
    logger.info(f"  - Has high text density: {has_high_text_density}")
    
    should_use_combined = is_howto_style and has_high_text_density
    logger.info(f"  - Use combined optimization: {should_use_combined}")
    
    return should_use_combined

def get_description_optimization_step(video_id: str):
    if should_use_combined_optimization(video_id):
        logger.info(f"Using combined optimization for video {video_id}")
        return {
            "command": f"python description_optimize_combined.py videos/{video_id}".strip(),
            "check": lambda: check_description_optimize_combined(video_id)
        }
    else:
        logger.info(f"Using standard optimization for video {video_id}")
        return [
            {
                "command": f"python description_optimize_inline.py videos/{video_id} --output audio_clips_optimized_gpt.json".strip(),
                "check": lambda: False  # Always run inline first
            },
            {
                "command": f"python description_optimize_extended.py videos/{video_id}".strip(),
                "check": lambda: check_description_optimize(video_id)
            }
        ]

def fetch_from_s3_if_available(video_id: str, s3_video_path: str = None, s3_metadata_path: str = None) -> bool:
    """Try to fetch the video from S3. Bucket comes from config (DEFAULT_S3_BUCKET). Returns True if successful or already local."""
    if check_youtube_downloaded(video_id):
        logger.info(f"Video {video_id} already exists locally. Skipping download.")
        return True

    try:
        from s3_fetcher import S3Fetcher
        fetcher = S3Fetcher(bucket_name=DEFAULT_S3_BUCKET)
        success = fetcher.fetch_video_package(
            video_id,
            output_dir="videos",
            s3_video_path=s3_video_path,
            s3_metadata_path=s3_metadata_path,
        )
        if success:
            logger.info(f"Video {video_id} fetched from S3 successfully.")
            return True
        else:
            logger.warning(f"S3 fetch failed for {video_id}, will try YouTube download.")
            return False
    except ImportError:
        logger.info("S3 fetcher not available (boto3 not installed). Using YouTube download.")
        return False
    except Exception as e:
        logger.warning(f"S3 fetch error for {video_id}: {e}. Will try YouTube download.")
        return False


def cleanup_scene_clips(video_id: str, base_dir: str = "videos"):
    """
    Delete scene clip .mp4 files after captioning is complete.
    These are only needed during transcription and captioning steps.
    JSON files in the scene dir are preserved (scene_info, audio_clips, etc.).
    Also removes legacy scene_boundaries/ dir if present from older pipeline runs.
    """
    import shutil as _shutil

    scene_dir = os.path.join(base_dir, video_id, f"{video_id}_scenes")
    if os.path.isdir(scene_dir):
        clip_files = glob.glob(os.path.join(scene_dir, "*.mp4"))
        total_size = sum(os.path.getsize(f) for f in clip_files)
        for f in clip_files:
            os.remove(f)
        if clip_files:
            logger.info(f"Cleaned up {len(clip_files)} scene clips ({total_size / (1024*1024):.2f} MB freed)")

    # Remove legacy scene_boundaries/ dir (no longer generated, but may exist from old runs)
    boundaries_dir = os.path.join(base_dir, video_id, "scene_boundaries")
    if os.path.isdir(boundaries_dir):
        _shutil.rmtree(boundaries_dir)
        logger.info(f"Cleaned up legacy scene_boundaries/ directory")


def run_pipeline(video_id: str, s3_video_path: str = None, s3_metadata_path: str = None) -> bool:
    # S3 bucket is always from config (DEFAULT_S3_BUCKET); only paths are passed per run.
    # Check if CUDA is available.
    if torch.cuda.is_available():
        logger.info("CUDA is available. Using CUDA for processing.")
        device_flag = ""
    else:
        logger.info("CUDA is not available. Using CPU for processing.")
        device_flag = "--device cpu"

    # Step 0: Try to fetch from S3 first, then fall back to YouTube download
    s3_fetched = fetch_from_s3_if_available(video_id, s3_video_path, s3_metadata_path)

    #define pipeline
    base_pipeline_steps = [
        {
            "command": f"python youtube_downloader.py {video_id}",
            "check": lambda: check_youtube_downloaded(video_id) or s3_fetched
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
            "command": f"python video_caption.py videos/{video_id}",
            "check": lambda: check_video_caption(video_id)
        }
    ]

    # Add description optimization steps (could be 1 or 2 steps)
    optimization_steps = get_description_optimization_step(video_id)
    base_pipeline_steps.extend(optimization_steps)

    # Add final step
    base_pipeline_steps.append({
        "command": f"python prepare_final_data.py {video_id} --model gpt",
        "check": lambda: check_final_data(video_id)
    })

    pipeline_steps = base_pipeline_steps
    for i, step in enumerate(pipeline_steps):
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

        # After video_caption step (step index 3), clean up scene clip .mp4 files
        # They are no longer needed — only JSON outputs matter from here on.
        if i == 3:
            cleanup_scene_clips(video_id)

    if not check_final_data(video_id):
        logger.error(f"final_data.json was not created for video {video_id}.")
        return False

    logger.debug("Pipeline completed successfully and final_data.json exists.")

    # Post-processing: upload results to S3 and optionally clean up local files
    cleanup_enabled = os.getenv("CLEANUP_AFTER_PROCESSING", "false").lower() == "true"
    try:
        from s3_fetcher import S3Fetcher
        fetcher = S3Fetcher(bucket_name=DEFAULT_S3_BUCKET)
        uploaded = fetcher.upload_pipeline_results(video_id)
        if uploaded:
            logger.info(f"Pipeline results uploaded to S3: {list(uploaded.keys())}")
            if cleanup_enabled:
                fetcher.cleanup_local_files(video_id)
            else:
                logger.info("Local cleanup disabled (set CLEANUP_AFTER_PROCESSING=true to enable)")
        else:
            logger.warning("No results uploaded to S3. Keeping local files.")
    except ImportError:
        logger.info("S3 upload not available (boto3 not installed). Results kept locally.")
    except Exception as e:
        logger.warning(f"S3 result upload failed: {e}. Results kept locally.")

    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run video processing pipeline with resume capability.")
    parser.add_argument("--video_id", required=True, help="YouTube video ID to process (e.g., dQw4w9WgXcQ)")
    parser.add_argument("--s3_video_path", default=None, help="S3 key for the video file (bucket from config)")
    parser.add_argument("--s3_metadata_path", default=None, help="S3 key for the metadata JSON")
    args = parser.parse_args()

    try:
        if run_pipeline(args.video_id, s3_video_path=args.s3_video_path, s3_metadata_path=args.s3_metadata_path):
            print("Pipeline executed successfully.")
        else:
            print("Pipeline execution failed. Check logs for details.", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Pipeline error: {e}", file=sys.stderr)
        logger.exception("Pipeline failed with exception")
        sys.exit(2)