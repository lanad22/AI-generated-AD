# info_bot_api.py
from typing import Optional
from fastapi import FastAPI, Request
from pydantic import BaseModel
import subprocess
import json
import os
import asyncio
import logging
import uvicorn
import sys

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

class NarrationBotRequest(BaseModel):
    youtube_id: str
    user_id: str
    ai_user_id: str

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

@app.post("/api/narration-bot")
async def narration_bot(data: NarrationBotRequest):
    logger.info(f"Received narration bot request: {data}")
    try:
        video_id = data.youtube_id
        final_data_path = os.path.join("videos", video_id, "final_data.json")
        
        if not os.path.exists(final_data_path):
            logger.info("final_data.json not found. Running pipeline via test_pipeline.py...")
            # Call test_pipeline.py using subprocess.
            command = f"python test_pipeline.py --video_id {video_id}"
            result = subprocess.run(
                command,
                shell=True,
                stdout=sys.stdout, 
                stderr=sys.stderr,
                text=True
            )
            if result.returncode != 0:
                logger.error("Pipeline failed to generate final_data.json")
                return {"status": "error", "message": "Pipeline failed to generate final_data.json"}
        
        with open(final_data_path, "r") as f:
            final_data = json.load(f)
        
        final_data["aiUserId"] = data.ai_user_id
        with open(final_data_path, "w") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        
        response_message = f"Narration bot processing complete for YouTube ID: {video_id}"
        logger.info(response_message)
        
        return {"status": "success", "message": response_message, "final_data": final_data}
    except Exception as e:
        logger.error(f"Error in narration bot endpoint: {str(e)}")
        return {"status": "error", "message": f"Error: {str(e)}"}
    
if __name__ == "__main__":
    logger.info("Starting Info Bot API server")
    uvicorn.run(app, host="0.0.0.0", port=8000)