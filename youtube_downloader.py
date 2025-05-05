import os
import json
import argparse
import subprocess
import re
from pathlib import Path
from youtube_transcript_api import YouTubeTranscriptApi
from datetime import timedelta

def get_video_metadata(video_id: str) -> dict:
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    # Separate commands for each piece of metadata for more reliable extraction
    title_cmd = ["yt-dlp", "--get-title", url]
    desc_cmd = ["yt-dlp", "--get-description", url]
    duration_cmd = ["yt-dlp", "--get-duration", url]
    # Add command to get the category
    category_cmd = ["yt-dlp", "--print", "categories", url]
    
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
            # Clean the category string from "['category']" format to just "category"
            category_str = result.stdout.strip()
            # Check if it's in list format like "['entertainment']"
            if category_str.startswith("['") and category_str.endswith("']"):
                # Extract the content between quotes
                category = category_str[2:-2]
            else:
                category = category_str
    except subprocess.CalledProcessError as e:
        print(f"Error fetching category: {e}")
        # Alternative method to get category
        try:
            info_json_cmd = ["yt-dlp", "--dump-json", url]
            result = subprocess.run(info_json_cmd, capture_output=True, text=True, check=True)
            if result.stdout.strip():
                video_info = json.loads(result.stdout)
                if "categories" in video_info and video_info["categories"]:
                    # Take just the first category as a string
                    category = video_info["categories"][0]
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e2:
            print(f"Alternative category fetch failed: {e2}")
    
    # Get duration with error handling
    try:
        result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
        duration_str = result.stdout.strip()
        
        # Make sure duration is in the expected format (HH:MM:SS)
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
            # Try to get duration through a different method
            info_cmd = ["yt-dlp", "--print", "duration", url]
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
    """Download YouTube video, metadata, and captions into a folder named after the video ID"""
    
    # Get video ID
    url = f"https://www.youtube.com/watch?v={video_id}"

    # Create a folder with the video ID
    output_dir = os.path.join(os.getcwd(), "videos", video_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Fetch video metadata (title, description, and category)
    print("Fetching video metadata...")
    metadata = get_video_metadata(video_id)
    print(f"Title: {metadata['title']}")
    print(f"Category: {metadata['category']}")
    print(f"Duration: {metadata['video_length']} seconds")

    # Prepare JSON data
    captions_path = os.path.join(output_dir, f"{video_id}.json")
    captions_data = {
        "title": metadata["title"], 
        "description": metadata["description"], 
        "video_length": metadata["video_length"],
        "category": metadata["category"],
        "captions": []
    }

    # Try downloading captions
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        captions_data["captions"] = transcript
        print("Captions successfully downloaded.")
    except Exception as e:
        print(f"Captions not available: {str(e)}")

    # Save JSON file with title, description, category, and (if available) captions
    with open(captions_path, 'w', encoding='utf-8') as f:
        json.dump(captions_data, f, indent=2, ensure_ascii=False)
    
    print(f"Metadata saved to: {captions_path}")

    # Output path
    video_path = os.path.join(output_dir, f"{video_id}.mp4")
    
    # Universal download command that works for all format types
    print("\nDownloading video...")
    command = [
        "yt-dlp",
        # Format selection that works for both normal and HLS formats
        "-f", "bv*+ba/b",  # Best video + best audio / or best combined if needed
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--hls-prefer-native",  
        "--ignore-errors",
        "-o", video_path,
        url
    ]
    
    try:
        subprocess.run(command, check=True)
        if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
            print(f"Success! Video downloaded to: {video_path}")
            print(f"File size: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
            return True
        else:
            print("Download failed or produced an empty file.")
            return False
    except subprocess.CalledProcessError as e:
        print(f"Error downloading video with standard method: {e}")
        
        # If standard method fails, try even more universal approach
        print("\nTrying alternative download method...")
        fallback_cmd = [
            "yt-dlp",
            # More universal/flexible format selection
            "--format-sort", "res,codec",
            "--merge-output-format", "mp4",
            "--allow-unplayable-formats",
            "--ignore-errors",
            "--no-playlist",
            "-o", video_path,
            url
        ]
        
        try:
            subprocess.run(fallback_cmd, check=True)
            if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                print(f"Success with alternative method! Video downloaded to: {video_path}")
                print(f"File size: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
                return True
            else:
                print("Alternative download attempt failed.")
                return False
        except subprocess.CalledProcessError as e2:
            print(f"Error with alternative method: {e2}")
            
            # Final attempt with the simplest possible approach
            print("\nMaking final download attempt...")
            last_resort_cmd = [
                "yt-dlp",
                "--no-check-formats",  # Skip format checking
                "--ignore-errors",
                "--no-playlist",
                "-o", video_path,
                url
            ]
            
            try:
                subprocess.run(last_resort_cmd, check=True)
                if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                    print(f"Success with final method! Video downloaded to: {video_path}")
                    print(f"File size: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
                    return True
                else:
                    print("All download attempts failed.")
                    return False
            except subprocess.CalledProcessError as e3:
                print(f"All download methods failed: {e3}")
                return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YouTube video by ID with captions and metadata.")
    parser.add_argument("video_id", help="YouTube video ID (e.g. dQw4w9WgXcQ)")
    args = parser.parse_args()

    download_with_captions(args.video_id)