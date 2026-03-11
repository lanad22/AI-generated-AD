"""
S3 Fetcher / Manager Module
Handles downloading video files from S3 for AI processing,
uploading pipeline results back to S3, and cleaning up local files.

Used when the video was downloaded by the local Youtube-Downloader service
and uploaded to S3, instead of downloading directly from YouTube.
"""

import os
import json
import shutil
import logging
import glob
import boto3
from botocore.exceptions import ClientError
from pathlib import Path

logger = logging.getLogger("narration_bot")


class S3Fetcher:
    def __init__(self, bucket_name: str = None, region: str = None):
        self.bucket_name = bucket_name or os.getenv("S3_VIDEO_BUCKET", "youdescribe-downloaded-youtube-videos")
        self.region = region or os.getenv("AWS_REGION", "us-west-1")
        self.s3_client = boto3.client(
            "s3",
            region_name=self.region,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )

    # =========================================================================
    # FETCH: Download video from S3 to local filesystem for pipeline processing
    # =========================================================================

    def download_file(self, s3_key: str, local_path: str) -> bool:
        """Download a single file from S3."""
        try:
            Path(os.path.dirname(local_path)).mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading s3://{self.bucket_name}/{s3_key} -> {local_path}")
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            logger.info(f"Downloaded successfully: {local_path} ({os.path.getsize(local_path) / (1024*1024):.2f} MB)")
            return True
        except ClientError as e:
            logger.error(f"Failed to download from S3: {e}")
            return False

    def fetch_video_package(self, video_id: str, output_dir: str = "videos",
                            s3_video_path: str = None, s3_metadata_path: str = None) -> bool:
        """
        Download video and metadata from S3 to local filesystem.
        Files are placed in output_dir/{video_id}/ where the pipeline expects them.
        """
        local_dir = os.path.join(output_dir, video_id)
        Path(local_dir).mkdir(parents=True, exist_ok=True)

        s3_prefix = f"videos/{video_id}"
        video_s3_key = s3_video_path or f"{s3_prefix}/{video_id}.mp4"
        metadata_s3_key = s3_metadata_path or f"{s3_prefix}/{video_id}.json"

        local_video = os.path.join(local_dir, f"{video_id}.mp4")
        local_metadata = os.path.join(local_dir, f"{video_id}.json")

        if os.path.exists(local_video) and os.path.getsize(local_video) > 0 and os.path.exists(local_metadata):
            logger.info(f"Video {video_id} already exists locally. Skipping S3 download.")
            return True

        video_ok = self.download_file(video_s3_key, local_video)
        if not video_ok:
            logger.error(f"Failed to download video for {video_id} from S3")
            return False

        metadata_ok = self.download_file(metadata_s3_key, local_metadata)
        if not metadata_ok:
            logger.warning(f"Metadata not found in S3 for {video_id}, pipeline may generate it")

        # Try thumbnail (optional)
        thumbnail_s3_key = f"{s3_prefix}/{video_id}_thumbnail.jpg"
        local_thumbnail = os.path.join(local_dir, f"{video_id}_thumbnail.jpg")
        try:
            self.download_file(thumbnail_s3_key, local_thumbnail)
        except Exception:
            logger.info(f"Thumbnail not available in S3 for {video_id} (optional)")

        return True

    def check_video_exists(self, video_id: str) -> bool:
        """Check if a video exists in S3."""
        try:
            s3_key = f"videos/{video_id}/{video_id}.mp4"
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError:
            return False

    # =========================================================================
    # UPLOAD: Push pipeline results back to S3
    # =========================================================================

    def upload_file(self, local_path: str, s3_key: str, content_type: str = None) -> bool:
        """Upload a single file to S3."""
        try:
            extra_args = {}
            if content_type:
                extra_args["ContentType"] = content_type

            logger.info(f"Uploading {local_path} -> s3://{self.bucket_name}/{s3_key}")
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key, ExtraArgs=extra_args)
            logger.info(f"Uploaded successfully: s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to upload {local_path} to S3: {e}")
            return False
        except FileNotFoundError:
            logger.error(f"File not found for upload: {local_path}")
            return False

    def upload_pipeline_results(self, video_id: str, base_dir: str = "videos") -> dict:
        """
        Upload pipeline output files to S3.
        Returns dict of uploaded S3 paths.

        Uploads:
        - final_data_*.json (the main AI output)
        - scene_info.json (base scene segmentation — needed by video query)
        - scene_info_gpt.json (scene-level captions with transcripts)
        - audio_clips_optimized_gpt.json (optimized audio clips)
        - keyframes/ directory (keyframe images + keyframe_info.json — needed by video query)
        """
        video_dir = os.path.join(base_dir, video_id)
        scene_dir = os.path.join(video_dir, f"{video_id}_scenes")
        keyframes_dir = os.path.join(video_dir, "keyframes")
        s3_prefix = f"results/{video_id}"
        uploaded = {}

        # Upload final_data files (could be final_data_gpt.json, final_data_gemini.json, etc.)
        final_data_files = glob.glob(os.path.join(video_dir, "final_data*.json"))
        for fpath in final_data_files:
            fname = os.path.basename(fpath)
            s3_key = f"{s3_prefix}/{fname}"
            if self.upload_file(fpath, s3_key, content_type="application/json"):
                uploaded[fname] = s3_key

        # Upload base scene_info.json (needed by video query for scene timing/transcript)
        scene_info_base = os.path.join(scene_dir, "scene_info.json")
        if os.path.exists(scene_info_base):
            s3_key = f"{s3_prefix}/scene_info.json"
            if self.upload_file(scene_info_base, s3_key, content_type="application/json"):
                uploaded["scene_info.json"] = s3_key

        # Upload scene_info_gpt.json (enriched with GPT captions)
        scene_info_gpt = os.path.join(scene_dir, "scene_info_gpt.json")
        if os.path.exists(scene_info_gpt):
            s3_key = f"{s3_prefix}/scene_info_gpt.json"
            if self.upload_file(scene_info_gpt, s3_key, content_type="application/json"):
                uploaded["scene_info_gpt.json"] = s3_key

        # Upload optimized audio clips
        audio_clips_opt = os.path.join(scene_dir, "audio_clips_optimized_gpt.json")
        if os.path.exists(audio_clips_opt):
            s3_key = f"{s3_prefix}/audio_clips_optimized_gpt.json"
            if self.upload_file(audio_clips_opt, s3_key, content_type="application/json"):
                uploaded["audio_clips_optimized_gpt.json"] = s3_key

        # Upload keyframes directory (images + keyframe_info.json — needed by video query)
        if os.path.isdir(keyframes_dir):
            keyframe_files = glob.glob(os.path.join(keyframes_dir, "*"))
            for fpath in keyframe_files:
                fname = os.path.basename(fpath)
                content_type = "application/json" if fname.endswith(".json") else "image/jpeg"
                s3_key = f"{s3_prefix}/keyframes/{fname}"
                if self.upload_file(fpath, s3_key, content_type=content_type):
                    uploaded[f"keyframes/{fname}"] = s3_key
            logger.info(f"Uploaded {len(keyframe_files)} keyframe files to S3 for {video_id}")

        if uploaded:
            logger.info(f"Uploaded {len(uploaded)} result files to S3 for {video_id}")
        else:
            logger.warning(f"No result files found to upload for {video_id}")

        return uploaded

    # =========================================================================
    # FETCH FOR VIDEO QUERY: Download keyframes and scene info from S3
    # =========================================================================

    def download_keyframes(self, video_id: str, base_dir: str = "videos") -> bool:
        """
        Download keyframes from S3 for video query use.
        Fetches results/{video_id}/keyframes/* -> videos/{video_id}/keyframes/
        """
        local_keyframes_dir = os.path.join(base_dir, video_id, "keyframes")

        # If already local, skip
        keyframe_info_path = os.path.join(local_keyframes_dir, "keyframe_info.json")
        if os.path.exists(keyframe_info_path):
            logger.info(f"Keyframes already exist locally for {video_id}")
            return True

        s3_prefix = f"results/{video_id}/keyframes/"
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=s3_prefix
            )
            if "Contents" not in response:
                logger.warning(f"No keyframes found in S3 for {video_id}")
                return False

            Path(local_keyframes_dir).mkdir(parents=True, exist_ok=True)
            for obj in response["Contents"]:
                s3_key = obj["Key"]
                fname = os.path.basename(s3_key)
                if not fname:
                    continue
                local_path = os.path.join(local_keyframes_dir, fname)
                self.download_file(s3_key, local_path)

            logger.info(f"Downloaded {len(response['Contents'])} keyframe files from S3 for {video_id}")
            return True
        except ClientError as e:
            logger.error(f"Failed to download keyframes from S3: {e}")
            return False

    def download_scene_info(self, video_id: str, base_dir: str = "videos") -> bool:
        """
        Download scene_info.json from S3 for video query use.
        Fetches results/{video_id}/scene_info.json -> videos/{video_id}/{video_id}_scenes/scene_info.json
        Also fetches scene_info_gpt.json if available (has transcript data).
        """
        scene_dir = os.path.join(base_dir, video_id, f"{video_id}_scenes")
        scene_info_path = os.path.join(scene_dir, "scene_info.json")

        if os.path.exists(scene_info_path):
            logger.info(f"Scene info already exists locally for {video_id}")
            return True

        Path(scene_dir).mkdir(parents=True, exist_ok=True)
        s3_prefix = f"results/{video_id}"

        # Download scene_info.json (base segmentation)
        ok = self.download_file(f"{s3_prefix}/scene_info.json", scene_info_path)

        # Also try scene_info_gpt.json (has transcript and audio_clips data)
        gpt_path = os.path.join(scene_dir, "scene_info_gpt.json")
        self.download_file(f"{s3_prefix}/scene_info_gpt.json", gpt_path)

        return ok

    def ensure_video_for_query(self, video_id: str, base_dir: str = "videos") -> bool:
        """
        Ensure keyframes, scene_info, and full video are available locally for video query.
        Downloads from S3 if not present.
        """
        ok = True
        if not self.download_keyframes(video_id, base_dir):
            logger.error(f"Could not obtain keyframes for {video_id}")
            ok = False
        if not self.download_scene_info(video_id, base_dir):
            logger.error(f"Could not obtain scene info for {video_id}")
            ok = False

        video_path = os.path.join(base_dir, video_id, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            logger.info(f"Full video not local, fetching from S3 for {video_id}")
            self.fetch_video_package(video_id, base_dir)

        return ok

    # =========================================================================
    # CLEANUP: Remove local files after processing to free EC2 disk space
    # =========================================================================

    @staticmethod
    def cleanup_local_files(video_id: str, base_dir: str = "videos") -> bool:
        """
        Remove the local video directory after results have been uploaded to S3.
        Only call this AFTER upload_pipeline_results() succeeds.
        """
        video_dir = os.path.join(base_dir, video_id)
        if os.path.exists(video_dir):
            dir_size = sum(
                os.path.getsize(os.path.join(dirpath, f))
                for dirpath, _, filenames in os.walk(video_dir)
                for f in filenames
            )
            shutil.rmtree(video_dir)
            logger.info(f"Cleaned up {video_dir} ({dir_size / (1024*1024):.2f} MB freed)")
            return True
        else:
            logger.info(f"Nothing to clean up for {video_id}")
            return False
