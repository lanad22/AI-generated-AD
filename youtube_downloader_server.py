import os
import json
import argparse
import subprocess
import re
from pathlib import Path
from youtube_transcript_api import YouTubeTranscriptApi
from datetime import timedelta
import urllib.request

# Global authentication flags to avoid bot detection
# This requires the one-time manual authentication: yt-dlp --username oauth2 --password ''
AUTH_FLAGS = ["--username", "oauth2", "--password", ""]

def get_video_metadata(video_id: str) -> dict:
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    # Updated commands to include OAuth2 flags
    title_cmd = ["yt-dlp"] + AUTH_FLAGS + ["--get-title", url]
    desc_cmd = ["yt-dlp"] + AUTH_FLAGS + ["--get-description", url]
    duration_cmd = ["yt-dlp"] + AUTH_FLAGS + ["--get-duration", url]
    category_cmd = ["yt-dlp"] + AUTH_FLAGS + ["--print", "categories", url]
    
    title = "Unknown Title"
    description = ""
    video_length = 0
    category = "Unknown Category"
    
    # Get title
    try:
        result = subprocess.run(title_cmd, capture_output=True, text=True, check=True)
        if result.stdout.strip():
            title = result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error fetching title: {e}")
    
    # Get description
    try:
        result = subprocess.run(desc_cmd, capture_output=True, text=True, check=True)
        if result.stdout.strip():
            description = result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error fetching description: {e}")
    
    # Get category
    try:
        result = subprocess.run(category_cmd, capture_output=True, text=True, check=True)
        if result.stdout.strip():
            category_str = result.stdout.strip()
            if category_str.startswith("['") and category_str.endswith("']"):
                category = category_str[2:-2]
            else:
                category = category_str
    except subprocess.CalledProcessError as e:
        print(f"Error fetching category: {e}")
        try:
            info_json_cmd = ["yt-dlp"] + AUTH_FLAGS + ["--dump-json", url]
            result = subprocess.run(info_json_cmd, capture_output=True, text=True, check=True)
            if result.stdout.strip():
                video_info = json.loads(result.stdout)
                if "categories" in video_info and video_info["categories"]:
                    category = video_info["categories"][0]
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e2:
            print(f"Alternative category fetch failed: {e2}")
    
    # Get duration
    try:
        result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
        duration_str = result.stdout.strip()
        
        if re.match(r'^\d+:\d+:\d+$', duration_str) or re.match(r'^\d+:\d+$', duration_str):
            parts = [int(p) for p in duration_str.split(':')]
            if len(parts) == 3:
                video_length = int(timedelta(hours=parts[0], minutes=parts[1], seconds=parts[2]).total_seconds())
            elif len(parts) == 2:
                video_length = int(timedelta(minutes=parts[0], seconds=parts[1]).total_seconds())
            elif len(parts) == 1:
                video_length = int(parts[0])
        else:
            print(f"Invalid duration format: {duration_str}")
            info_cmd = ["yt-dlp"] + AUTH_FLAGS + ["--print", "duration", url]
            try:
                result = subprocess.run(info_cmd, capture_output=True, text=True, check=True)
                if result.stdout.strip().isdigit():
                    video_length = int(result.stdout.strip())
            except subprocess.CalledProcessError:
                print("Could not determine video duration")
    except subprocess.CalledProcessError as e:
        print(f"Error fetching duration: {e}")
        
    return {
        "title": title, 
        "description": description, 
        "video_length": video_length,
        "category": category
    }

def download_with_captions(video_id: str):
    url = f"https://www.youtube.com/watch?v={video_id}"
    output_dir = os.path.join(os.getcwd(), "videos", video_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Fetching video metadata...")
    metadata = get_video_metadata(video_id)
    print(f"Title: {metadata['title']}")
    print(f"Category: {metadata['category']}")
    print(f"Duration: {metadata['video_length']} seconds")

    captions_path = os.path.join(output_dir, f"{video_id}.json")
    captions_data = {
        "title": metadata["title"], 
        "description": metadata["description"], 
        "video_length": metadata["video_length"],
        "category": metadata["category"],
        "captions": []
    }

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        captions_data["captions"] = transcript
        print("Captions successfully downloaded.")
    except Exception as e:
        print(f"Captions not available: {str(e)}")

    with open(captions_path, 'w', encoding='utf-8') as f:
        json.dump(captions_data, f, indent=2, ensure_ascii=False)
    
    print(f"Metadata saved to: {captions_path}")

    # Download thumbnail
    thumbnail_path = os.path.join(output_dir, f"{video_id}_thumbnail.jpg")
    print("\nDownloading thumbnail...")
    thumbnail_cmd = ["yt-dlp"] + AUTH_FLAGS + [
        "--write-thumbnail",
        "--skip-download",
        "--convert-thumbnails", "jpg",
        "-o", os.path.join(output_dir, video_id),
        url
    ]
    
    try:
        subprocess.run(thumbnail_cmd, check=True)
        possible_thumbnails = [
            os.path.join(output_dir, f"{video_id}.jpg"),
            os.path.join(output_dir, f"{video_id}.webp")
        ]
        
        thumbnail_found = False
        for thumb in possible_thumbnails:
            if os.path.exists(thumb):
                if thumb.endswith('.webp'):
                    os.rename(thumb, thumbnail_path)
                thumbnail_found = True
                print(f"Thumbnail downloaded to: {thumbnail_path}")
                break
                
        if not thumbnail_found:
            print("Thumbnail download failed. Trying alternative method...")
            info_cmd = ["yt-dlp"] + AUTH_FLAGS + ["--dump-json", url]
            result = subprocess.run(info_cmd, capture_output=True, text=True, check=True)
            if result.stdout.strip():
                video_info = json.loads(result.stdout)
                if "thumbnail" in video_info and video_info["thumbnail"]:
                    urllib.request.urlretrieve(video_info["thumbnail"], thumbnail_path)
                    print(f"Thumbnail downloaded with alternative method to: {thumbnail_path}")
                    thumbnail_found = True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading thumbnail: {e}")

    # Video Download
    video_path = os.path.join(output_dir, f"{video_id}.mp4")
    print("\nDownloading video...")
    
    # Standard download command
    command = ["yt-dlp"] + AUTH_FLAGS + [
        "-f", "bv*+ba/b", 
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--hls-prefer-native",  
        "--ignore-errors",
        "-o", video_path,
        url
    ]
    
    # Alternative (Fallback) command
    fallback_cmd = ["yt-dlp"] + AUTH_FLAGS + [
        "--format-sort", "res,codec",
        "--merge-output-format", "mp4",
        "--allow-unplayable-formats",
        "--ignore-errors",
        "--no-playlist",
        "-o", video_path,
        url
    ]

    for cmd in [command, fallback_cmd]:
        try:
            subprocess.run(cmd, check=True)
            if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                print(f"Success! Video downloaded to: {video_path}")
                return True
        except subprocess.CalledProcessError as e:
            print(f"Method failed, trying next...")

    print("All download attempts failed.")
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YouTube video by ID with captions and metadata.")
    parser.add_argument("video_id", help="YouTube video ID (e.g. dQw4w9WgXcQ)")
    args = parser.parse_args()

    download_with_captions(args.video_id)