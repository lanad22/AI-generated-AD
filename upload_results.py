import os
import sys
import logging
import argparse
import glob
import boto3
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("upload_results")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

S3_VIDEO_BUCKET = os.getenv("S3_VIDEO_BUCKET", "youdescribe-downloaded-youtube-videos")
AWS_REGION = os.getenv("AWS_REGION", "us-west-1")

s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

CONTENT_TYPES = {
    ".json": "application/json",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".mp4": "video/mp4",
}


def upload_file(local_path: str, s3_key: str) -> bool:
    ext = os.path.splitext(local_path)[1].lower()
    extra_args = {}
    if ext in CONTENT_TYPES:
        extra_args["ContentType"] = CONTENT_TYPES[ext]

    try:
        logger.info(f"Uploading {local_path} -> s3://{S3_VIDEO_BUCKET}/{s3_key}")
        s3_client.upload_file(local_path, S3_VIDEO_BUCKET, s3_key, ExtraArgs=extra_args)
        return True
    except Exception as e:
        logger.error(f"Failed to upload {local_path}: {e}")
        return False


def upload_directory(local_dir: str, s3_prefix: str) -> int:
    uploaded = 0
    for root, dirs, files in os.walk(local_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, local_dir)
            s3_key = f"{s3_prefix}/{relative_path}"
            if upload_file(local_path, s3_key):
                uploaded += 1
    return uploaded


def upload_results(video_id: str) -> bool:
    base_dir = os.path.join("videos", video_id)
    s3_prefix = f"results/{video_id}"

    if not os.path.exists(base_dir):
        logger.error(f"Video directory not found: {base_dir}")
        return False

    success = True

    keyframes_dir = os.path.join(base_dir, "keyframes")
    if os.path.isdir(keyframes_dir):
        count = upload_directory(keyframes_dir, f"{s3_prefix}/keyframes")
        logger.info(f"Uploaded {count} keyframe files")
    else:
        logger.warning(f"Keyframes directory not found: {keyframes_dir}")

    scenes_dir = os.path.join(base_dir, f"{video_id}_scenes")
    if os.path.isdir(scenes_dir):
        count = upload_directory(scenes_dir, f"{s3_prefix}/{video_id}_scenes")
        logger.info(f"Uploaded {count} scene files")
    else:
        logger.warning(f"Scenes directory not found: {scenes_dir}")

    metadata_path = os.path.join(base_dir, f"{video_id}.json")
    if os.path.exists(metadata_path):
        upload_file(metadata_path, f"{s3_prefix}/{video_id}.json")

    for final_data_path in glob.glob(os.path.join(base_dir, "final_data*.json")):
        filename = os.path.basename(final_data_path)
        upload_file(final_data_path, f"{s3_prefix}/{filename}")

    logger.info(f"Results upload complete for {video_id}")
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload processing results to S3")
    parser.add_argument("video_id", help="YouTube video ID")
    args = parser.parse_args()

    if upload_results(args.video_id):
        print(f"Results for {args.video_id} uploaded to S3.")
    else:
        print(f"Failed to upload results for {args.video_id}.")
        sys.exit(1)
