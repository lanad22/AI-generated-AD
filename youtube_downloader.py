import os
import json
import argparse
import subprocess
from pathlib import Path
from youtube_transcript_api import YouTubeTranscriptApi
from datetime import timedelta

def get_video_metadata(video_id: str) -> dict:
    url = f"https://www.youtube.com/watch?v={video_id}"
    command = [
        "yt-dlp",
        "--get-title",
        "--get-description",
        "--get-duration",
        url
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        metadata = result.stdout.strip().splitlines()
        title = metadata[0] if len(metadata) > 0 else "Unknown Title"
        description = metadata[1] if len(metadata) > 1 else ""
        duration_str = metadata[2] if len(metadata) > 2 else "0"

        # Inline duration-to-seconds conversion using timedelta
        parts = [int(p) for p in duration_str.strip().split(":")]
        if len(parts) == 3:
            video_length = int(timedelta(hours=parts[0], minutes=parts[1], seconds=parts[2]).total_seconds())
        elif len(parts) == 2:
            video_length = int(timedelta(minutes=parts[0], seconds=parts[1]).total_seconds())
        elif len(parts) == 1:
            video_length = int(timedelta(seconds=parts[0]).total_seconds())
        else:
            video_length = 0

        return {"title": title, "description": description, "video_length": video_length}
    except subprocess.CalledProcessError as e:
        print(f"Error fetching metadata: {e}")
        return {"title": "Unknown Title", "description": "", "video_length": 0}

def download_with_captions(video_id: str):
    """Download YouTube video, metadata, and captions into a folder named after the video ID"""
    
    # Get video ID
    url = f"https://www.youtube.com/watch?v={video_id}"

    # Create a folder with the video ID
    output_dir = os.path.join(os.getcwd(), "videos", video_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Fetch video metadata (title & description)
    metadata = get_video_metadata(video_id)

    # Prepare JSON data
    captions_path = os.path.join(output_dir, f"{video_id}.json")
    captions_data = {"title": metadata["title"], "description": metadata["description"], 
                     "video_length": metadata["video_length"],"captions": []}

    # Try downloading captions
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        captions_data["captions"] = transcript
        print("Captions successfully downloaded.")
    except Exception as e:
        print(f"Captions not available: {str(e)}")

    # Save JSON file with title, description, and (if available) captions
    with open(captions_path, 'w', encoding='utf-8') as f:
        json.dump(captions_data, f, indent=2, ensure_ascii=False)
    
    print(f"Metadata saved to: {captions_path}")

    # Download video
    video_path = os.path.join(output_dir, f"{video_id}.mp4")
    command = f'yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4" -o "{video_path}" {url}'
    print("\nDownloading video...")
    os.system(command)

    if os.path.exists(video_path):
        print(f"Video downloaded to: {video_path}")
        print(f"File size: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
    else:
        print("No MP4 file found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YouTube video by ID with captions and metadata.")
    parser.add_argument("video_id", help="YouTube video ID (e.g. dQw4w9WgXcQ)")
    args = parser.parse_args()

    download_with_captions(args.video_id)
