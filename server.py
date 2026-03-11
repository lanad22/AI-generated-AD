# YouDescribe AI Service: video query API and AI description pipeline.
from typing import Optional, Dict
from enum import Enum
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import json
import os
import asyncio
import logging
import uvicorn
import sys
import requests
import glob
import time

from config import API_BASE_URL

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ai_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ai_service")

app = FastAPI()

pipeline_status: Dict[str, dict] = {}

MAX_CALLBACK_RETRIES = 3
CALLBACK_RETRY_DELAY_SECONDS = 5

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
    # S3 source (optional): when set, pipeline fetches video from S3 instead of YouTube
    s3_bucket: Optional[str] = None  # defaults to config.S3_VIDEO_BUCKET
    s3_video_path: Optional[str] = None
    s3_metadata_path: Optional[str] = None

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

@app.post("/api/query-video-frame")
async def query_video_frame(data: QueryModel):
    logger.info(f"Received request: {data}")

    if data.question is None:
        data.question = "describe the scene"
    
    # Run video_query_keyframe.py for visual Q&A at a given timestamp
    video_query_script = "video_query_keyframe.py"
    
    command = [
        "python", 
        video_query_script,
        data.video_id,
        data.current_time,
        data.question
    ]
    
    try:
        # Run the script
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


async def run_pipeline_and_forward(video_id: str, user_id: Optional[str], ai_user_id: Optional[str], data_type: DataType,
                                    s3_video_path: Optional[str] = None, s3_metadata_path: Optional[str] = None):
    """Run pipeline (bucket from config); only S3 paths are passed per request."""
    start_time = time.time()
    pipeline_status[video_id] = {
        "status": "processing",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": 0,
        "step": "initializing",
        "error": None,
    }

    try:
        logger.info(f"Starting background pipeline processing for {video_id}")
        pipeline_status[video_id]["step"] = "running_pipeline"
        
        # test_pipeline.py reads S3 bucket from config; we only pass per-request paths when present
        command = ["python", "test_pipeline.py", "--video_id", video_id]
        if s3_video_path:
            command.extend(["--s3_video_path", s3_video_path])
        if s3_metadata_path:
            command.extend(["--s3_metadata_path", s3_metadata_path])
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        stdout_bytes, _ = await process.communicate()
        elapsed = time.time() - start_time
        pipeline_status[video_id]["elapsed_seconds"] = round(elapsed)
        output_text = (stdout_bytes or b"").decode("utf-8", errors="replace").strip()

        if process.returncode != 0:
            logger.error(f"Pipeline failed for {video_id} after {elapsed:.0f}s (exit {process.returncode})")
            # Keep last ~2000 chars of output so status API can show why it failed
            pipeline_status[video_id].update({
                "status": "failed",
                "step": "pipeline_execution",
                "error": f"Exit code {process.returncode}",
                "log_tail": output_text[-2000:] if output_text else None,
            })
            return
        
        logger.info(f"Pipeline completed successfully for {video_id} in {elapsed:.0f}s")
        pipeline_status[video_id]["step"] = "forwarding_results"
        
        # Update aiUserId if provided
        final_data_path = os.path.join("videos", video_id, "final_data.json")
        if ai_user_id and os.path.exists(final_data_path):
            with open(final_data_path, "r") as f:
                final_data = json.load(f)
            final_data["aiUserId"] = ai_user_id
            with open(final_data_path, "w") as f:
                json.dump(final_data, f, indent=2, ensure_ascii=False)
        
        # Forward to newaidescription endpoint with retry logic
        logger.info(f"Forwarding {video_id} with data_type={data_type.value}")
        forward_request = UnifiedVideoRequest(
            youtube_id=video_id,
            user_id=user_id,
            ai_user_id=ai_user_id,
            data_type=data_type
        )
        
        forwarded = False
        for attempt in range(1, MAX_CALLBACK_RETRIES + 1):
            try:
                await forward_final_data(forward_request)
                logger.info(f"Successfully forwarded {video_id} to production (attempt {attempt})")
                forwarded = True
                break
            except Exception as e:
                logger.warning(f"Forward attempt {attempt}/{MAX_CALLBACK_RETRIES} failed for {video_id}: {e}")
                if attempt < MAX_CALLBACK_RETRIES:
                    await asyncio.sleep(CALLBACK_RETRY_DELAY_SECONDS * attempt)

        elapsed = time.time() - start_time
        if forwarded:
            pipeline_status[video_id].update({"status": "completed", "step": "done", "elapsed_seconds": round(elapsed)})
        else:
            logger.error(f"All {MAX_CALLBACK_RETRIES} forward attempts failed for {video_id}")
            pipeline_status[video_id].update({"status": "completed_no_forward", "step": "forward_failed", "elapsed_seconds": round(elapsed),
                                               "error": "Results produced but API callback failed after retries. Results are in S3."})
                        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Error in background pipeline processing for {video_id}: {str(e)}")
        pipeline_status[video_id].update({"status": "failed", "step": "unknown", "error": str(e), "elapsed_seconds": round(elapsed)})

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the API is running."""
    return {
        "status": "healthy",
        "service": "YouDescribe AI Service",
        "message": "Service is running",
        "active_pipelines": sum(1 for v in pipeline_status.values() if v["status"] == "processing"),
    }

@app.get("/api/pipeline-status/{video_id}")
async def get_pipeline_status(video_id: str):
    """
    Check the processing status of a video pipeline.
    Useful for long-running jobs (30 min - several hours).
    """
    if video_id in pipeline_status:
        entry = pipeline_status[video_id]
        # Update elapsed time if still processing
        if entry["status"] == "processing":
            started = entry.get("started_at", "")
            entry["elapsed_seconds"] = round(time.time() - time.mktime(time.strptime(started, "%Y-%m-%d %H:%M:%S"))) if started else 0
        return entry

    # Check if final data already exists on disk (previously completed)
    pattern = os.path.join("videos", video_id, "final_data*.json")
    if glob.glob(pattern):
        return {"status": "completed", "step": "done", "message": "Results available (processed in a prior session)"}

    return {"status": "not_found", "message": f"No pipeline record for {video_id}"}
    
@app.post("/api/generate-ai-description")
async def narration_bot(data: UnifiedVideoRequest, background_tasks: BackgroundTasks):
    logger.info(f"Received narration bot request: {data}")
    
    video_id = data.youtube_id

    # Reject duplicate requests for the same video that's already processing
    if video_id in pipeline_status and pipeline_status[video_id]["status"] == "processing":
        return {
            "status": "already_processing",
            "message": f"Pipeline already running for {video_id}. Check /api/pipeline-status/{video_id}",
        }

    pattern = os.path.join("videos", video_id, "final_data*.json")
    
    # glob.glob returns a list of matching files
    if glob.glob(pattern):
        logger.info(f"File {pattern} exists. Skipping pipeline and JUMPING to forwarding.")
        
        # skip run pipeline but still sends the data to the backend
        background_tasks.add_task(forward_final_data, data)
        
        return {
            "status": "already_exists",
            "message": "Video found. Forwarding existing data now."
        }

    # Start the heavy lifting in the background (bucket from config in test_pipeline)
    background_tasks.add_task(
        run_pipeline_and_forward, 
        video_id, 
        data.user_id, 
        data.ai_user_id, 
        data.data_type,
        data.s3_video_path,
        data.s3_metadata_path,
    )
    
    # Return immediately so Node.js doesn't time out
    return {
        "status": "processing",
        "message": f"Pipeline started in background for {video_id}. Check /api/pipeline-status/{video_id}",
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
        # Construct the file path based on data_type
        filename = f"final_data_{data.data_type.value}.json"
        final_data_path = os.path.join("videos", data.youtube_id, filename)
        
        # Check if the specified file exists
        if not os.path.exists(final_data_path):
            raise HTTPException(
                status_code=404,
                detail=f"{filename} not found for YouTube ID: {data.youtube_id}"
            )
        
        # Load the final_data file
        try:
            with open(final_data_path, "r") as f:
                final_data = json.load(f)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load {filename}: {str(e)}"
            )
        
        target_url = f"{API_BASE_URL}/api/audio-descriptions/newaidescription"
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(target_url, data=json.dumps(final_data), headers=headers)
            response.raise_for_status()  
            json_response = response.json()
            logger.info(f"json_response: {json_response}")

            if json_response.get('_id'):
                generateAudioClips = f"{API_BASE_URL}/api/audio-clips/processAllClipsInDB/{json_response['_id']}"
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
        
        # Parse the JSON response
        try:
            json_response = response.json()
            logger.info(f"Successfully forwarded {filename} to {target_url}")
            logger.info(f"Response: {json_response}")
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
    logger.info("Starting YouDescribe AI Service")
    uvicorn.run(app, host="0.0.0.0", port=8000)