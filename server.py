from typing import Optional
from enum import Enum
from fastapi import FastAPI, HTTPException
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

s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)


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
    data_type: DataType = DataType.GPT

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
        
        command = [PYTHON, "test_pipeline.py", "--video_id", video_id]
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=sys.stdout,
            stderr=sys.stderr
        )

        await process.wait()

        if process.returncode != 0:
            logger.error(f"Pipeline failed for {video_id} (exit code {process.returncode})")
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
        
        await safe_forward(forward_request)

        if CLEANUP_AFTER_PROCESSING:
            logger.info(f"Cleaning up local files for {video_id}...")
            cleanup_video_dir(video_id)

    except Exception as e:
        logger.error(f"Error in background pipeline processing for {video_id}: {str(e)}")

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
    asyncio.create_task(
        run_pipeline_and_forward(video_id, data.user_id, data.ai_user_id, data.data_type)
    )
    
    return {
        "status": "processing",
        "message": f"Pipeline started in background for {video_id}"
    }
        
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