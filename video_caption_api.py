import os
import json
import re
import ast
import argparse
import subprocess
import base64
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def convert_for_qwen(input_path):
    """Convert video to a format compatible with Qwen model."""
    base_path, ext = os.path.splitext(input_path)
    output_path = f"{base_path}_qwen{ext}"
    
    command = [
        "ffmpeg", "-y",
        "-loglevel", "quiet",
        "-i", input_path,
        "-an",
        "-c:v", "libx264",
        "-vf", "scale='min(1280,iw)':'-2'",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

def encode_video_to_base64(video_path):
    """Encode video file to base64 for API transmission."""
    with open(video_path, "rb") as video_file:
        encoded_string = base64.b64encode(video_file.read())
    return encoded_string.decode('utf-8')

def extract_and_parse_json(response):
    """Extract and parse JSON from the model response."""
    try:
        # Remove markdown code block markers if present
        response = re.sub(r'```json|```', '', response)
        json_pattern = r'\[\s*{.*}\s*\]'
        json_match = re.search(json_pattern, response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0).strip()
            try:
                parsed_data = json.loads(json_str)
                return parsed_data
            except json.JSONDecodeError:
                try:
                    parsed_data = ast.literal_eval(json_str)
                    return parsed_data
                except Exception:
                    return []
        else:
            return []
    except Exception:
        return []

def prepare_context(scene_data, previous_description=None):
    """Prepare context information from transcript and previous description."""
    context_parts = []
    
    # Add transcript information
    if scene_data.get("transcript"):
        transcripts_info = ""
        for transcript in scene_data.get("transcript"):
            transcripts_info += f"{transcript['text']}\n"
        context_parts.append(
            f"TRANSCRIPT:\n{transcripts_info}\n"
            "Do not repeat this information in the description.\n"
        )
    
    # Add captions information
    if scene_data.get("captions"):
        captions_info = ""
        for caption in scene_data.get("captions"):
            captions_info += f"{caption['text']}\n"
        context_parts.append(f"CAPTIONS:\n{captions_info}\n")
    
    # Add previous description if available
    if previous_description:
        context_parts.append(
            f"PREVIOUS SCENE CONTEXT (for reference only, do not repeat):\n{previous_description}\n\n"
            "Focus on new observations, actions, and details from the current scene.\n"
        )
    
    return "\n\n".join(context_parts)

def process_scene(scene_data, scene_path, client, previous_description=None, system_message=None, max_retries=3):
    """Process a scene to create audio descriptions using the Qwen API."""
    # Prepare context
    context = prepare_context(scene_data, previous_description)
    scene_duration = scene_data.get("duration")
    
    # Create a more concise scene prompt that focuses on context review
    prompt = f"""
            SCENE DURATION: {scene_duration:.2f} seconds

            CONTEXT:
            {context}

            You are analyzing a video scene. First, identify specific characters, locations, and any important elements mentioned in the context.

            Then generate a JSON array of scene events with the following rules:

            EVENT TYPES:
            1. Text Events ("type": "Text on Screen"):
            - Capture ONLY visible on-screen text.
            - DO NOT include transcript or dialogue.
            - CRITICAL: For each text event, include the EXACT `start_time` in seconds when the text appears.

            2. Visual Description Events ("type": "Visual"):
            - Provide a brief, focused description of the visual content (e.g., who is present, what they are doing).
            - DO NOT include any text or camera movements.
            - Use exact `start_time` for when the visual starts.
            - Do not repeat descriptions from earlier.

            ### RULES:
            - ALWAYS use specific character names from the context if available. Do not use generic terms like "man" or "person".
            - Format the output as a JSON array. Each object should include:
            - `start_time` (in seconds)
            - `type` ("Text on Screen" or "Visual")
            - `text` (description or on-screen text)
            
            Now generate the JSON array of events for this scene.
        """

    
    print(f"SCENE PROMPT LENGTH: {len(prompt)} characters")
    
    if not system_message:
        system_message = "You are a professional audio describer following strict guidelines."
    
    # Encode video to base64
    print("Encoding video for API transmission...")
    encoded_video = encode_video_to_base64(scene_path)
    
    # Set up retry mechanism
    retry_count = 0
    response = None
    
    while retry_count < max_retries:
        try:
            print(f"Calling Qwen API (attempt {retry_count + 1})...")
            
            # Call the Qwen API
            completion = client.chat.completions.create(
                model="qwen2.5-vl-72b-instruct",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{encoded_video}"}}
                    ]}
                ],
                max_tokens=2048,
                temperature=0.7,
                #timeout=120  # Increased timeout for longer responses
            )
            
            # Extract response
            response = completion.choices[0].message.content
            print(f'RESPONSE: {response}')
            break
            
        except Exception as e:
            retry_count += 1
            print(f"API call failed (attempt {retry_count}): {str(e)}")
            
            if retry_count < max_retries:
                wait_time = 2 ** retry_count  
                print(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                print("Maximum retries reached. Moving on...")
                return []
    
    if not response:
        print("Failed to get a response from the API.")
        return []
    
    # Extract the JSON events
    events = extract_and_parse_json(response)
    
    # Convert any "start_time" strings to floats
    if not isinstance(events, list):
        print(f"Warning: Could not extract valid JSON from response. Using empty list.")
        events = []
    
    for event in events:
        if "start_time" in event and isinstance(event["start_time"], str):
            try:
                event["start_time"] = float(event["start_time"])
            except ValueError:
                event["start_time"] = 0.0
    
    # Deduplicate text events (if same text appears multiple times)
    unique_texts = {}
    deduplicated_events = []
    
    for event in events:
        if event.get("type") == "text":
            text_content = event.get("text", "").strip()
            if text_content and text_content not in unique_texts:
                unique_texts[text_content] = True
                deduplicated_events.append(event)
        else:
            deduplicated_events.append(event)
    
    if len(events) != len(deduplicated_events):
        print(f"Deduplicated text events: {len(events)} → {len(deduplicated_events)}")
    
    deduplicated_events.sort(key=lambda e: e.get("start_time", 0))
    return deduplicated_events

def process_all_scenes(video_folder, client, skip_guidelines=False):
    """Process all scenes in the provided video folder with initial model priming."""
    video_id = os.path.basename(os.path.normpath(video_folder))
    video_metadata_path = os.path.join(video_folder, f"{video_id}.json")
    scenes_folder = os.path.join(video_folder, f"{video_id}_scenes")
    scenes_json_path = os.path.join(scenes_folder, "scene_info.json")
    
    if not os.path.exists(video_metadata_path):
        print(f"Error: {video_metadata_path} not found. Unable to retrieve title and description.")
        return
    
    # Load video metadata for context
    with open(video_metadata_path, "r") as f:
        video_metadata = json.load(f)
    
    video_title = video_metadata.get("title", "Unknown Title")
    video_description = video_metadata.get("description", "")
    previous_description = f"Video Title: {video_title}\n{video_description}"
    
    if not os.path.exists(scenes_json_path):
        print(f"Error: scene_info.json not found in {scenes_folder}")
        return
    
    with open(scenes_json_path, "r") as f:
        scene_list = json.load(f)
    
    print(f"Processing {len(scene_list)} scenes in {scenes_folder}...")
    
    # Define guidelines
    guidelines = """
    AUDIO DESCRIPTION GUIDELINES:
    - Describe what you see in a concise, factual manner.
    - Always read on-screen text exactly as it appears.
    - Be factual, objective, and precise in your descriptions.
    - Use proper terminology and names from the context when possible.
    - Match the tone and mood of the video.
    - Do not describe what can be inferred from the audio.
    - Do not over-describe - less is more.
    - Do not interpret or editorialize about what you see.
    - Do not give away surprises before they happen.
    - DO NOT mention or describe any camera movements or transitions.
   
    IMPORTANT CHARACTER IDENTIFICATION:
    - When you recognize a character from the context, ALWAYS use their specific name.
    - Before each scene, carefully review context to identify all named characters.
    - Use the most specific identification possible based on the context information.
    """
    
    # First, have the model understand the guidelines by sending a one-time prompt
    print("\n----- ESTABLISHING GUIDELINES WITH MODEL -----")
    guidelines_prompt = f"""
        You are a professional audio describer following these guidelines:
        
        {guidelines}
        
        Do you understand these guidelines? Respond with "YES" and a brief confirmation.
        """
        
    # Call the Qwen API for guidelines understanding
    try:
            completion = client.chat.completions.create(
                model="qwen2.5-vl-72b-instruct",
                messages=[
                    {"role": "system", "content": "You are a professional audio describer."},
                    {"role": "user", "content": guidelines_prompt}
                ],
                max_tokens=200,
                temperature=0.3,
                timeout=30
            )
            
            guidelines_response = completion.choices[0].message.content
            print(f"MODEL RESPONSE: {guidelines_response}")
            print("----- GUIDELINES ESTABLISHED -----\n")
            
    except Exception as e:
            print(f"Warning: Failed to establish guidelines: {str(e)}")
            print("----- CONTINUING WITHOUT GUIDELINES ESTABLISHMENT -----\n")
    
    # Create a system message that includes the guidelines reference
    system_message = "You are a professional audio describer following strict guidelines."
    
    # Process each scene, building context from previous scenes
    for idx, scene_data in enumerate(scene_list, start=1):
        print(f"\nProcessing Scene {idx}/{len(scene_list)}: {scene_data.get('scene_number', 'Unknown')}")
        scene_path = scene_data.get('scene_path', '')
        if not os.path.exists(scene_path):
            print(f"Scene file not found: {scene_path}")
            continue
        
        # Get previous scene descriptions for context
        if idx > 1:
            prev_scene = scene_list[idx-2]  # idx-2 because idx starts at 1
            prev_descriptions = []
            
            if prev_scene.get('audio_clips'):
                # Extract visual descriptions from previous scene
                prev_descriptions = [clip.get('text', '') for clip in prev_scene.get('audio_clips', []) 
                                    if clip.get('type') == 'visual']
                
                if prev_descriptions:
                    # Update previous_description with actual previous scene data
                    previous_description = f"{video_title}\n{video_description}\n\nPrevious scene: {prev_descriptions[0]}"
        
        try:
            print("Converting video for Qwen format...")
            converted_scene_path = convert_for_qwen(scene_path)
            
            try:
                # Process scene with our API approach
                scene_events = process_scene(
                    scene_data, 
                    converted_scene_path, 
                    client, 
                    previous_description=previous_description,
                    system_message=system_message
                )
                
                scene_data['audio_clips'] = scene_events
                
            except Exception as e:
                print(f"Error processing scene {idx}: {str(e)}")
                scene_data['audio_clips'] = []
            
            if converted_scene_path != scene_path and os.path.exists(converted_scene_path):
                os.remove(converted_scene_path)
                print(f"Removed temporary Qwen format file: {converted_scene_path}")
        
        except Exception as e:
            print(f"Error converting video: {str(e)}")
            scene_data['audio_clips'] = []
        
    with open(scenes_json_path, "w") as f:
        json.dump(scene_list, f, indent=4)
    
    print(f"\nScene descriptions updated in: {scenes_json_path}")
    return scene_list

def main():
    parser = argparse.ArgumentParser(
        description="Generate audio descriptions for video scenes using Qwen2.5-VL-72B-Instruct API."
    )
    parser.add_argument("video_folder", type=str,
                        help="Folder containing video files and metadata.")
    args = parser.parse_args()
    
    # Setup API client
    api_key = os.getenv("API_KEY")
    base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    
    if not api_key:
        raise ValueError("API key must be provided via API_KEY environment variable")
    
    print("Setting up OpenAI client for DashScope API...")
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    
    # Process all scenes
    process_all_scenes(args.video_folder, client)
    
    print("Processing complete.")

if __name__ == "__main__":
    main()