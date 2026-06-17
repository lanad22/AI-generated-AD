from typing import Optional
from enum import Enum
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import shutil
import json
import os
import asyncio
import logging
import uvicorn
import sys
import requests
import glob
import boto3
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("info_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("info_bot")

app = FastAPI()

PYTHON = sys.executable
CLEANUP_AFTER_PROCESSING = os.getenv("CLEANUP_AFTER_PROCESSING", "false").lower() == "true"
YDX_API_URL = os.getenv("YDX_API_URL", "http://localhost:4001")

S3_VIDEO_BUCKET = os.getenv("S3_VIDEO_BUCKET", "youdescribe-downloaded-youtube-videos")
AWS_REGION = os.getenv("AWS_REGION", "us-west-1")

# Concurrency cap (defense-in-depth — the api is the primary scheduler).
# Default 2 matches AI_PIPELINE_CONCURRENCY on the api side for m5.large.
MAX_CONCURRENT_PIPELINES = int(os.getenv("MAX_CONCURRENT_PIPELINES", "2"))
pipeline_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PIPELINES)

s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)


# Models that pipeline subprocesses lazy-load from disk cache. If the cache is missing,
# each subprocess would download the same file in parallel and corrupt each other's cache
# (TOCTOU race). Pre-warm here so the cache exists before any concurrent pipeline starts.
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "large-v3-turbo")
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL", "ViT-B/32")


@app.on_event("startup")
async def preload_models():
    """Populate the Whisper and CLIP disk caches once at startup, in the background.

    Pipeline subprocesses (test_pipeline.py → transcribe_scenes.py / keyframe_scene_detector.py)
    call load_model()/clip.load() which read from ~/.cache/whisper and ~/.cache/clip.
    Concurrent pipelines on a cold cache race and corrupt the download (SHA256 mismatch).
    We warm those caches once in this server process so subsequent pipeline subprocesses
    find the files on disk and skip the download.

    Important: the warm runs as a fire-and-forget asyncio task so it does NOT block uvicorn
    from accepting connections. Blocking startup would fail the deploy's health-check window.
    On a warm-cache production EC2 the load completes in a few seconds; on a true cold start
    nothing is dispatching requests yet, so the warm has time to finish before the first
    pipeline. Failures are logged; pipelines will fall back to lazy load if the cache is
    somehow still cold when they run (original behavior).
    """

    def _warm():
        import gc
        try:
            logger.info(f"Pre-warming Whisper cache (model={WHISPER_MODEL_NAME})...")
            import whisper_timestamped
            _m = whisper_timestamped.load_model(WHISPER_MODEL_NAME, device="cpu")
            del _m
            gc.collect()
            logger.info("Whisper cache warmed.")
        except Exception as e:
            logger.error(f"Whisper pre-warm failed (pipelines will retry lazily): {e}")

        try:
            logger.info(f"Pre-warming CLIP cache (model={CLIP_MODEL_NAME})...")
            import clip
            _m, _p = clip.load(CLIP_MODEL_NAME, device="cpu")
            del _m, _p
            gc.collect()
            logger.info("CLIP cache warmed.")
        except Exception as e:
            logger.error(f"CLIP pre-warm failed (pipelines will retry lazily): {e}")

        try:
            # transcribe_scenes.py loads nltk.corpus.words and calls nltk.download('words')
            # on LookupError. Two concurrent pipelines doing this on a cold cache race on
            # the ~/nltk_data/corpora/words.zip file. Pre-warm once here so the corpus
            # is on disk before any pipeline subprocess runs.
            logger.info("Pre-warming NLTK 'words' corpus...")
            import nltk
            from nltk.corpus import words as _nltk_words
            try:
                _nltk_words.fileids()
            except LookupError:
                nltk.download('words', quiet=True)
                _nltk_words.fileids()
            logger.info("NLTK 'words' corpus warmed.")
        except Exception as e:
            logger.error(f"NLTK pre-warm failed (pipelines will retry lazily): {e}")

    # Schedule the warm in a worker thread without awaiting — server startup returns
    # immediately so /health responds and the deploy health-check passes.
    asyncio.create_task(asyncio.to_thread(_warm))


def download_results_from_s3(video_id: str) -> bool:
    s3_prefix = f"results/{video_id}/"
    local_base = os.path.join("videos", video_id)

    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=S3_VIDEO_BUCKET, Prefix=s3_prefix)

        found_any = False
        for page in pages:
            for obj in page.get("Contents", []):
                s3_key = obj["Key"]
                relative_path = s3_key[len(s3_prefix):]
                if not relative_path:
                    continue
                local_path = os.path.join(local_base, relative_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                logger.info(f"Downloading s3://{S3_VIDEO_BUCKET}/{s3_key} -> {local_path}")
                s3_client.download_file(S3_VIDEO_BUCKET, s3_key, local_path)
                found_any = True

        return found_any
    except Exception as e:
        logger.error(f"Failed to download results from S3 for {video_id}: {e}")
        return False


def check_and_download_final_data_from_s3(video_id: str) -> bool:
    s3_prefix = f"results/{video_id}/"
    local_base = os.path.join("videos", video_id)

    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=S3_VIDEO_BUCKET, Prefix=s3_prefix)

        found_final = False
        for page in pages:
            for obj in page.get("Contents", []):
                s3_key = obj["Key"]
                filename = os.path.basename(s3_key)
                if filename.startswith("final_data") and filename.endswith(".json"):
                    local_path = os.path.join(local_base, filename)
                    os.makedirs(local_base, exist_ok=True)
                    logger.info(f"Downloading s3://{S3_VIDEO_BUCKET}/{s3_key} -> {local_path}")
                    s3_client.download_file(S3_VIDEO_BUCKET, s3_key, local_path)
                    found_final = True

        return found_final
    except Exception as e:
        logger.error(f"Failed to check S3 for final_data of {video_id}: {e}")
        return False


def cleanup_video_dir(video_id: str):
    video_dir = os.path.join("videos", video_id)
    if os.path.exists(video_dir):
        shutil.rmtree(video_dir)
        logger.info(f"Cleaned up local directory: {video_dir}")

class QueryModel(BaseModel):
    question: Optional[str] = None
    current_time: str
    video_id: str

class DataType(str, Enum):
    HUMAN = "human"
    QWEN = "qwen"
    GEMINI = "gemini"
    GPT = "gpt"
    BAD = "bad"

# Unified request model for both endpoints
class UnifiedVideoRequest(BaseModel):
    youtube_id: str
    user_id: Optional[str] = None
    ai_user_id: Optional[str] = None
    data_type: DataType = DataType.GEMINI

async def run_query_script(command):
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    return {
        "returncode": process.returncode,
        "stdout": stdout.decode(),
        "stderr": stderr.decode()
    }

async def get_response_from_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return f.read()
    return None

@app.post("/api/info-bot")
async def receive_data(data: QueryModel):
    logger.info(f"Received request: {data}")

    if data.question is None:
        data.question = "describe the scene"

    scene_info_path = os.path.join("videos", data.video_id, f"{data.video_id}_scenes", "scene_info.json")
    if not os.path.exists(scene_info_path):
        logger.info(f"Scene info not found locally for {data.video_id}, downloading results from S3...")
        downloaded = await asyncio.to_thread(download_results_from_s3, data.video_id)
        if not downloaded:
            return {"status": "error", "message": f"No processed data found for video {data.video_id}"}

    video_query_script = "video_query_keyframe.py"

    command = [
        PYTHON,
        video_query_script,
        data.video_id,
        data.current_time,
        data.question
    ]

    try:
        logger.info(f"Running command: {' '.join(command)}")
        result = await run_query_script(command)
        
        if result["returncode"] != 0:
            logger.error(f"Script error: {result['stderr']}")
            return {
                "status": "error", 
                "message": f"Error processing video query: {result['stderr']}"
            }
        
        # Check for response file
        response_file = f"videos/{data.video_id}/{data.video_id}_{int(float(data.current_time))}s.txt"
        response_text = await get_response_from_file(response_file)
        
        if response_text:
            logger.info(f"Successfully processed request, response in {response_file}")
            return {
                "status": "success", 
                "message": "Query processed successfully",
                "response": response_text
            }
        else:
            logger.error(f"Response file not found: {response_file}")
            return {
                "status": "error", 
                "message": "Response file not found"
            }
    
    except Exception as e:
        logger.error(f"Error running script: {str(e)}")
        return {"status": "error", "message": f"Error: {str(e)}"}

async def safe_forward(data: UnifiedVideoRequest):
    try:
        await forward_final_data(data)
        logger.info(f"Background forward succeeded for {data.youtube_id}")
    except Exception as e:
        logger.error(f"Background forward failed for {data.youtube_id}: {str(e)}")
        
        
async def run_pipeline_and_forward(video_id: str, user_id: Optional[str], ai_user_id: Optional[str], data_type: DataType):
    try:
        logger.info(f"Starting background pipeline processing for {video_id}")
        
        command = [PYTHON, "test_pipeline.py", "--video_id", video_id, "--model", data_type.value,]
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=sys.stdout,
            stderr=sys.stderr
        )

        await process.wait()

        if process.returncode != 0:
            reason = f"Pipeline exited with code {process.returncode}"
            logger.error(f"Pipeline failed for {video_id}: {reason}")
            await notify_pipeline_failure(video_id, reason, user_id, ai_user_id)
            if CLEANUP_AFTER_PROCESSING:
                cleanup_video_dir(video_id)
            return
        
        logger.info(f"Pipeline completed successfully for {video_id}")
        
        final_data_path = os.path.join("videos", video_id, "final_data.json")
        if ai_user_id and os.path.exists(final_data_path):
            with open(final_data_path, "r") as f:
                final_data = json.load(f)
            final_data["aiUserId"] = ai_user_id
            with open(final_data_path, "w") as f:
                json.dump(final_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Forwarding {video_id} with data_type={data_type.value}")
        forward_request = UnifiedVideoRequest(
            youtube_id=video_id,
            user_id=user_id,
            ai_user_id=ai_user_id,
            data_type=data_type
        )
        
        try:
            await forward_final_data(forward_request)
            logger.info(f"Forward succeeded for {video_id}")
        except Exception as e:
            reason = f"Forward to YDX failed: {str(e)}"
            logger.error(f"{reason} for {video_id}")
            await notify_pipeline_failure(video_id, reason, user_id, ai_user_id)
            return

        if CLEANUP_AFTER_PROCESSING:
            logger.info(f"Cleaning up local files for {video_id}...")
            cleanup_video_dir(video_id)

    except Exception as e:
        logger.error(f"Error in background pipeline processing for {video_id}: {str(e)}")
    finally:
        pipeline_semaphore.release()
        logger.info(f"Released pipeline slot for {video_id}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {
        "status": "healthy",
        "service": "Lana GenAD API",
        "message": "Service is running"
    }
    
@app.post("/api/generate-ai-description")
async def narration_bot(data: UnifiedVideoRequest):
    logger.info(f"Received narration bot request: {data}")
    
    video_id = data.youtube_id
    pattern = os.path.join("videos", video_id, "final_data*.json")

    if glob.glob(pattern):
        logger.info(f"Final data exists locally for {video_id}. Skipping pipeline and forwarding.")
        asyncio.create_task(safe_forward(data))
        return {
            "status": "already_exists",
            "message": "Video found. Forwarding existing data now."
        }

    if check_and_download_final_data_from_s3(video_id) and glob.glob(pattern):
        logger.info(f"Final data found in S3 for {video_id}. Skipping pipeline and forwarding.")
        asyncio.create_task(safe_forward(data))
        return {
            "status": "already_exists",
            "message": "Video found in S3. Forwarding existing data now."
        }

    logger.info(f"No existing data found for {video_id}. Starting pipeline.")

    # Try to acquire a pipeline slot without blocking the request.
    # 50ms timeout: long enough to give asyncio.Semaphore.acquire() an event-loop turn
    # to complete when a slot is free, short enough to be effectively non-blocking when
    # at capacity. Works on Python 3.10+ (avoids the wait_for(timeout=0) bug on 3.13).
    try:
        await asyncio.wait_for(pipeline_semaphore.acquire(), timeout=0.05)
    except asyncio.TimeoutError:
        logger.warning(f"Pipeline at capacity ({MAX_CONCURRENT_PIPELINES}); rejecting {video_id} with 503")
        return JSONResponse(
            status_code=503,
            content={"status": "busy", "message": "AI pipeline at capacity. Please retry."},
        )

    # Slot acquired — schedule the pipeline. The worker releases the slot in its finally block.
    asyncio.create_task(
        run_pipeline_and_forward(video_id, data.user_id, data.ai_user_id, data.data_type)
    )

    return {
        "status": "processing",
        "message": f"Pipeline started in background for {video_id}"
    }

async def notify_pipeline_failure(video_id: str, reason: str, user_id: Optional[str] = None, ai_user_id: Optional[str] = None):
    """Notify YDX backend that pipeline failed so it can update status, email user, and clean up."""
    target_url = f"{YDX_API_URL}/api/audio-descriptions/aidescription-failure"
    payload = {
        "youtube_id": video_id,
        "reason": reason,
        "user_id": user_id,
        "ai_user_id": ai_user_id,
    }
    try:
        response = await asyncio.to_thread(
            requests.post, target_url, json=payload, headers={"Content-Type": "application/json"}, timeout=30
        )
        response.raise_for_status()
        logger.info(f"Failure notification sent for {video_id}")
    except Exception as e:
        logger.error(f"Failed to notify YDX of failure for {video_id}: {e}")
                
@app.post("/api/newaidescription")
async def forward_final_data(data: UnifiedVideoRequest):
    """
    API to forward specified final_data file to another server.
    Supports: final_data_human.json, final_data_qwen.json, final_data_gemini.json, final_data_gpt.json
    
    Usage examples:
    - {"youtube_id": "abc123", "data_type": "human"}
    - {"youtube_id": "abc123", "data_type": "qwen"}
    - {"youtube_id": "abc123"} # defaults to gpt
    - {"youtube_id": "abc123", "user_id": "user1", "ai_user_id": "ai1", "data_type": "gemini"}
    """
    logger.info(f"Received request to forward final_data_{data.data_type.value}.json for YouTube ID: {data.youtube_id}")
    
    try:
        filename = f"final_data_{data.data_type.value}.json"
        final_data_path = os.path.join("videos", data.youtube_id, filename)
        
        if not os.path.exists(final_data_path):
            raise HTTPException(
                status_code=404,
                detail=f"{filename} not found for YouTube ID: {data.youtube_id}"
            )
        
        try:
            with open(final_data_path, "r") as f:
                final_data = json.load(f)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load {filename}: {str(e)}"
            )

        if not final_data.get("audio_clips"):
            logger.error(f"No audio clips in {filename} for {data.youtube_id}. Skipping forward.")
            raise HTTPException(
                status_code=400,
                detail=f"audio_clips is empty in {filename} for {data.youtube_id}. Pipeline produced no descriptions."
            )

        target_url = f"{YDX_API_URL}/api/audio-descriptions/newaidescription"
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(target_url, data=json.dumps(final_data), headers=headers)
            response.raise_for_status()  
            json_response = response.json()
            logger.info(f"json_response: {json_response}")

            if json_response.get('_id'):
                generateAudioClips = f"{YDX_API_URL}/api/audio-clips/processAllClipsInDB/{json_response['_id']}"
                r = requests.get(generateAudioClips)

                if r.status_code == 200:
                    logger.info("Processed all clips in DB")
                    logger.info(r.text)
                else:
                    logger.warning(f"Failed to process clips. Status: {r.status_code}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to forward data. Error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to forward data: {str(e)}"
            )
        
        try:
            json_response = response.json()
            logger.info(f"Successfully forwarded {filename} to {target_url}")
            return {
                "status": "success", 
                "message": f"Data forwarded successfully from {filename}", 
                "data_type": data.data_type.value,
                "response": json_response
            }
        except ValueError as e:
            logger.error(f"Failed to parse JSON response. Error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse JSON response: {str(e)}"
            )
    
    except HTTPException as http_exc:
        logger.error(f"HTTPException: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

if __name__ == "__main__":
    logger.info("Starting Info Bot API server")
    uvicorn.run(app, host="0.0.0.0", port=8000)