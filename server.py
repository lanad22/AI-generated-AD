# info_bot_api.py
from typing import Optional
from enum import Enum
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import subprocess
import json
import os
import asyncio
import logging
import uvicorn
import sys
import requests
import glob

# Set up logging
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

class QueryModel(BaseModel):
    question: Optional[str] = None
    current_time: str
    video_id: str

class DataType(str, Enum):
    HUMAN = "human"
    QWEN = "qwen"
    GEMINI = "gemini"
    GPT = "gpt"

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
    
    # Create command to run the video_query.py script
    video_query_script = "video_query_keyframe.py"  # Path to your script
    
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


async def run_pipeline_and_forward(video_id: str, user_id: Optional[str], ai_user_id: Optional[str], data_type: DataType):
    try:
        logger.info(f"Starting background pipeline processing for {video_id}")
        
        # Run the pipeline asynchronously
        command = ["python", "test_pipeline.py", "--video_id", video_id]
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=sys.stdout,
            stderr=sys.stderr
        )

        await process.wait()

        if process.returncode != 0:
            logger.error(f"Pipeline failed for {video_id}")
            return
        
        logger.info(f"Pipeline completed successfully for {video_id}")
        
        # Update aiUserId if provided
        final_data_path = os.path.join("videos", video_id, "final_data.json")
        if ai_user_id and os.path.exists(final_data_path):
            with open(final_data_path, "r") as f:
                final_data = json.load(f)
            final_data["aiUserId"] = ai_user_id
            with open(final_data_path, "w") as f:
                json.dump(final_data, f, indent=2, ensure_ascii=False)
        
        # Forward to newaidescription endpoint
        logger.info(f"Forwarding {video_id} with data_type={data_type.value}")
        forward_request = UnifiedVideoRequest(
            youtube_id=video_id,
            user_id=user_id,
            ai_user_id=ai_user_id,
            data_type=data_type
        )
        
        # Call the forward function directly
        try:
            await forward_final_data(forward_request)
            logger.info(f"Successfully forwarded {video_id} to production")
        except Exception as e:
            logger.error(f"Failed to forward {video_id}: {str(e)}")
                        
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
async def narration_bot(data: UnifiedVideoRequest, background_tasks: BackgroundTasks):
    logger.info(f"Received narration bot request: {data}")
    
    video_id = data.youtube_id
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

    # Start the heavy lifting in the background
    background_tasks.add_task(
        run_pipeline_and_forward, 
        video_id, 
        data.user_id, 
        data.ai_user_id, 
        data.data_type
    )
    
    # Return immediately so Node.js doesn't time out
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
        
        target_url = "http://localhost:4001/api/audio-descriptions/newaidescription" 
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(target_url, data=json.dumps(final_data), headers=headers)
            response.raise_for_status()  
            json_response = response.json()
            logger.info(f"json_response: {json_response}")

            if json_response.get('_id'):
                generateAudioClips = f"http://localhost:4001/api/audio-clips/processAllClipsInDB/{json_response['_id']}"
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
    logger.info("Starting Info Bot API server")
    uvicorn.run(app, host="0.0.0.0", port=8000)