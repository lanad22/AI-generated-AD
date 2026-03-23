import os
import sys
import time
import json
import logging
import argparse
import boto3
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("fetch_video")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

S3_VIDEO_BUCKET = os.getenv("S3_VIDEO_BUCKET", "youdescribe-downloaded-youtube-videos")
AWS_REGION = os.getenv("AWS_REGION", "us-west-1")
YOUTUBE_DOWNLOADER_URL = os.getenv("YOUTUBE_DOWNLOADER_URL", "http://localhost:8001")

POLL_INTERVAL = 5
POLL_TIMEOUT = 600

s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)


def check_local_exists(video_id: str) -> bool:
    output_dir = os.path.join("videos", video_id)
    video_path = os.path.join(output_dir, f"{video_id}.mp4")
    metadata_path = os.path.join(output_dir, f"{video_id}.json")
    return os.path.exists(video_path) and os.path.exists(metadata_path)


def check_s3_exists(video_id: str) -> bool:
    try:
        s3_client.head_object(Bucket=S3_VIDEO_BUCKET, Key=f"videos/{video_id}/{video_id}.mp4")
        return True
    except Exception:
        return False


def download_from_s3(video_id: str) -> bool:
    output_dir = os.path.join("videos", video_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    files_to_download = [
        (f"videos/{video_id}/{video_id}.mp4", os.path.join(output_dir, f"{video_id}.mp4")),
        (f"videos/{video_id}/{video_id}.json", os.path.join(output_dir, f"{video_id}.json")),
        (f"videos/{video_id}/{video_id}_thumbnail.jpg", os.path.join(output_dir, f"{video_id}_thumbnail.jpg")),
    ]

    for s3_key, local_path in files_to_download:
        try:
            logger.info(f"Downloading s3://{S3_VIDEO_BUCKET}/{s3_key} -> {local_path}")
            s3_client.download_file(S3_VIDEO_BUCKET, s3_key, local_path)
        except Exception as e:
            if s3_key.endswith("_thumbnail.jpg"):
                logger.warning(f"Thumbnail not found in S3 (non-critical): {e}")
            else:
                logger.error(f"Failed to download {s3_key}: {e}")
                return False

    logger.info(f"Successfully downloaded video {video_id} from S3")
    return True


def trigger_youtube_download(video_id: str) -> bool:
    url = f"{YOUTUBE_DOWNLOADER_URL}/api/download"
    try:
        resp = requests.post(url, json={"youtube_id": video_id}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"Download triggered for {video_id}: status={data.get('status')}")

        if data.get("status") == "completed":
            return True

        return poll_download_status(video_id)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to trigger download for {video_id}: {e}")
        return False


def poll_download_status(video_id: str) -> bool:
    url = f"{YOUTUBE_DOWNLOADER_URL}/api/download/status/{video_id}"
    start_time = time.time()

    while time.time() - start_time < POLL_TIMEOUT:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status")
            logger.info(f"[{video_id}] Download status: {status}")

            if status == "completed":
                return True
            elif status == "failed":
                logger.error(f"[{video_id}] Download failed: {data.get('error')}")
                return False

            time.sleep(POLL_INTERVAL)
        except requests.exceptions.RequestException as e:
            logger.warning(f"[{video_id}] Poll request failed: {e}, retrying...")
            time.sleep(POLL_INTERVAL)

    logger.error(f"[{video_id}] Download timed out after {POLL_TIMEOUT}s")
    return False


def fetch_video(video_id: str) -> bool:
    if check_local_exists(video_id):
        logger.info(f"[{video_id}] Already exists locally, skipping download")
        return True

    if check_s3_exists(video_id):
        logger.info(f"[{video_id}] Found in S3, downloading to local...")
        return download_from_s3(video_id)

    logger.info(f"[{video_id}] Not found locally or in S3, triggering Youtube-Downloader...")
    success = trigger_youtube_download(video_id)
    if not success:
        return False

    return download_from_s3(video_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch video from local, S3, or Youtube-Downloader")
    parser.add_argument("video_id", help="YouTube video ID")
    args = parser.parse_args()

    if fetch_video(args.video_id):
        print(f"Video {args.video_id} is ready locally.")
    else:
        print(f"Failed to fetch video {args.video_id}.")
        sys.exit(1)
