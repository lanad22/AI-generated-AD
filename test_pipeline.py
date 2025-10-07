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
    scene_info_path = os.path.join("videos", video_id, f"{video_id}_scenes", "scene_info_gemini.json")
    if not os.path.exists(scene_info_path):
        logger.debug(f"video_caption_api check: {scene_info_path} does not exist")
        return False
    try:
        with open(scene_info_path, "r") as f:
            scenes = json.load(f)
        # We expect each scene to include the "audio_clips" field.
        for scene in scenes:
            if "audio_clips" not in scene or scene["audio_clips"] == []:
                logger.debug(f"video_caption_api check: Scene {scene.get('scene_number')} is missing 'audio_clips'")
                return False
        logger.debug("video_caption_api check: All scenes have 'audio_clips'")
        return True
    except Exception as e:
        logger.error(f"Error reading {scene_info_path}: {e}")
        return False

def check_description_optimize(video_id: str) -> bool:
    scene_dir = os.path.join("videos", video_id, f"{video_id}_scenes")
    audio_clips_opt_path = os.path.join(scene_dir, "audio_clips_optimized_gemini.json")
    result = os.path.exists(audio_clips_opt_path)
    logger.debug(f"Check description_optimize for {video_id}: {result}")
    return result

def check_description_optimize_combined(video_id: str) -> bool:
    scene_dir = os.path.join("videos", video_id, f"{video_id}_scenes")
    audio_clips_combined_path = os.path.join(scene_dir, "audio_clips_optimized_combined.json")
    result = os.path.exists(audio_clips_combined_path)
    logger.debug(f"Check description_optimize_combined for {video_id}: {result}")
    return result

def check_final_data(video_id: str) -> bool:
    final_data_path = os.path.join("videos", video_id, "final_data.json")
    result = os.path.exists(final_data_path)
    logger.debug(f"Check final_data for {video_id}: {result}")
    return result

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
    scene_info_path = os.path.join("videos", video_id, f"{video_id}_scenes", "scene_info_gemini.json")
    
    if not os.path.exists(scene_info_path):
        logger.warning(f"scene_info_gemini.json not found for video {video_id}")
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

def get_description_optimization_step(video_id: str, device_flag: str):
    if should_use_combined_optimization(video_id):
        logger.info(f"Using combined optimization for video {video_id}")
        return {
            "command": f"python description_optimize_combined.py videos/{video_id} {device_flag}".strip(),
            "check": lambda: check_description_optimize_combined(video_id)
        }
    else:
        logger.info(f"Using standard optimization for video {video_id}")
        return {
            "command": f"python description_optimize.py videos/{video_id} {device_flag}".strip(),
            "check": lambda: check_description_optimize(video_id)
        }

def run_pipeline(video_id: str) -> bool:
    # Check if CUDA is available.
    if torch.cuda.is_available():
        logger.info("CUDA is available. Using CUDA for processing.")
        device_flag = ""
    else:
        logger.info("CUDA is not available. Using CPU for processing.")
        device_flag = "--device cpu"

    # Define pipeline steps with conditional description optimization
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
            "command": f"python video_caption.py videos/{video_id}",
            "check": lambda: check_video_caption_api(video_id)
        },
        # Conditional description optimization step - determined at runtime
        get_description_optimization_step(video_id, device_flag),
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run video processing pipeline with resume capability.")
    parser.add_argument("--video_id", required=True, help="YouTube video ID to process (e.g., dQw4w9WgXcQ)")
    args = parser.parse_args()

    if run_pipeline(args.video_id):
        print("Pipeline executed successfully.")
    else:
        print("Pipeline execution failed. Check logs for details.")