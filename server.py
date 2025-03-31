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

def run_pipeline(video_id: str) -> bool:
    pipeline_commands = [
        f"python youtube_downloaded.py {video_id}",
        f"python keyframe_scene_detector.py videos/{video_id}",
        f"python transcribe_scene.py videos/{video_id}",
        f"python video_caption_api.py videos/{video_id}",
        f"python description_optimize.py videos/{video_id}",
        f"python prepare_final_data.py --video_id {video_id}"
    ]
    
    for command in pipeline_commands:
        logger.info(f"Running command: {command}")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        logger.info(result.stdout)
        if result.returncode != 0:
            logger.error(f"Command failed: {command}\nError: {result.stderr}")
            return False
    return True

@app.post("/api/narration-bot")
async def narration_bot(data: NarrationBotRequest):
    logger.info(f"Received narration bot request: {data}")
    try:
        video_id = data.youtube_id
        final_data_path = os.path.join("videos", video_id, "final_data.json")
        
        if not os.path.exists(final_data_path):
            logger.info("final_data.json not found. Running pipeline...")
            if not run_pipeline(video_id):
                return {"status": "error", "message": "Pipeline failed to generate final_data.json"}
        
        # Load the final_data.json file
        with open(final_data_path, "r") as f:
            final_data = json.load(f)
        
        # Write the ai_user_id into final_data.json
        final_data["aiUserId"] = data.ai_user_id
        
        # Save the updated final_data.json back to disk
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