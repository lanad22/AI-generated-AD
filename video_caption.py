import os
import json
import re
import ast
import argparse
import subprocess
import base64
import time
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


MODEL_GEMINI = "gemini"
MODEL_QWEN = "qwen"


AUDIO_DESCRIPTION_GUIDELINES = """
AUDIO DESCRIPTION GUIDELINES (for "Visual" events):
- Describe what you see in a concise, factual manner.
- Be factual, objective, and precise in your descriptions.
- Use proper terminology and names from the context (like character names) when possible.
- Match the tone and mood of the video.
- Do not over-describe; less is often more.
- Do not interpret or editorialize what you see.
- Do not give away surprises before they happen.
- Do not describe camera movements.

CHARACTER IDENTIFICATION GUIDELINES (for "Visual" events):
- When you recognize a character from the context, ALWAYS use their specific name.
- Before describing a scene, carefully review any provided context to identify all named characters.
- Use the most specific identification possible based on context.
"""

PROMPT_TEMPLATE = """
        Scene Duration: {scene_duration:.2f} seconds

        CONTEXT FOR CURRENT SCENE ANALYSIS:
        {context_block}

        You are analyzing a video scene. Identify specific characters, locations, and any important elements mentioned in the context.

        First, generate a JSON array of Text on Screen events.
            Text Events ("type": "Text on Screen"):
            - Capture visible on-screen text.
            - DO NOT include transcript or dialogue.
            - CRITICAL: For each text event, include the EXACT `start_time` in seconds when the text appears.
            - Combine events that have the same start_time or appear within 2s. 
            
            INCLUDE:
            - Titles, headings, names
            - Informational text
            - Important dates or events

            EXCLUDE:
            - Brand logos and watermarks
            - Network logos
            - Social media handles
            - Copyright notices
            
        Second, generate a JSON array of Visual event.
            - Provide contextually rich visual description of the scene using very concise wording
            - Describe each action in this scene in every specific details. 
            - ALWAYS use specific character names from context (not "person" or "woman")
            - Focus on key actions, settings, objects that aren't mentioned in previous description
            - Include clear start times for each visual event
            - IMPORTANT: DO NOT describe Text on Screen

            ### RULES:
            - Format the output as a JSON array. Each object should include:
            - `start_time` (in seconds)
            - `type` ("Text on Screen" or "Visual")
            - `text` (description or on-screen text)       
            Now generate the JSON array of events for this scene.
        """


MODEL_CONFIGS = {
    MODEL_GEMINI: {
        "model_name": "gemini-1.5-pro-latest",
        "system_instruction": f"""
        You are an expert video analysis AI. The user will provide context including video title, overall description, and potentially details from the immediately preceding scene.

        Follow these CORE GUIDELINES for your analysis:
        {AUDIO_DESCRIPTION_GUIDELINES}
        """,
        "generation_config": {
            "temperature": 0.6,
            "max_output_tokens": 4096,
            "response_mime_type": "application/json",
        },
        "safety_settings": {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        },
        "max_retries": 2
    },
    MODEL_QWEN: {
        "model_name": "qwen2.5-vl-72b-instruct",
        "system_message": f"""
            You are a professional audio describer and video analysis AI. The user will provide context including video title, overall description, and potentially details from the immediately preceding scene.
            You MUST strictly follow these CORE GUIDELINES for your analysis and descriptions:
            {AUDIO_DESCRIPTION_GUIDELINES}

            SPECIFIC RULE FOR 'Howto & Style' CATEGORY VIDEOS:
            If the video category (provided in context) is identified as "howto & style", for "Visual" events, DO NOT mention any person or body part. ONLY describe the action itself, often in the imperative mood (e.g., "Unscrew the lid," "Fold the paper."). For other video categories, describe persons and actions normally.
            """,
                    "api_base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                    "generation_config": {
                        "max_tokens": 2048,
                        "temperature": 0.7,
                    },
                    "max_retries": 3
                }
}

def convert_video_for_processing(input_path, model_type_for_filename_suffix):
    base_path, ext = os.path.splitext(input_path)
    output_path = f"{base_path}_{model_type_for_filename_suffix}_temp{ext}"
    command = [
        "ffmpeg", "-y", "-loglevel", "quiet", "-i", input_path,
        "-an", "-c:v", "libx264", "-vf", "scale='min(1280,iw)':'-2'",
        "-pix_fmt", "yuv420p", output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error during ffmpeg conversion for {input_path} (suffix: {model_type_for_filename_suffix}): {e}. Using original.")
        return input_path

def encode_video_to_base64(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode('utf-8')

def extract_and_parse_json(response_text):
    if not response_text:
        return []
    try:
        response_text = re.sub(r'```json|```', '', response_text).strip()
        json_match = re.search(r'^\s*\[[\s\S]*?\]\s*$', response_text, re.MULTILINE)
        if not json_match:
            json_match = re.search(r'\[\s*{[\s\S]*?}\s*\]', response_text, re.DOTALL)

        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError: {e}. Attempting ast.literal_eval on: {json_str[:200]}...")
                try:
                    evaluated_data = ast.literal_eval(json_str)
                    return evaluated_data if isinstance(evaluated_data, list) else []
                except Exception as e_ast:
                    print(f"ast.literal_eval failed: {e_ast} on content: {json_str[:200]}")
                    return []
        else:
            print(f"No valid JSON array found in response: {response_text[:200]}...")
            return []
    except Exception as e:
        print(f"Generic error in extract_and_parse_json: {e}")
        return []

def prepare_context_block_for_scene(base_context_from_previous, video_category, current_scene_data, scene_idx):
    context_parts = [base_context_from_previous]

    context_parts.append(f"\nVideo Category: {video_category}") 

    current_scene_specific_context = []
    if current_scene_data.get("transcript"):
        transcripts_info = "".join(f"{t['text']}\n" for t in current_scene_data["transcript"])
        current_scene_specific_context.append(f"TRANSCRIPT FOR CURRENT SCENE (Scene {scene_idx + 1}):\n{transcripts_info}")

    if current_scene_data.get("captions"):
        captions_info = "".join(f"{c['text']}\n" for c in current_scene_data["captions"])
        current_scene_specific_context.append(f"CAPTIONS FOR CURRENT SCENE (Scene {scene_idx + 1}):\n{captions_info}")

    if current_scene_specific_context:
        context_parts.append("\nADDITIONAL CONTEXT FOR CURRENT SCENE ANALYSIS:")
        context_parts.extend(current_scene_specific_context)
    elif scene_idx == 0 and "PREVIOUS SCENE INFORMATION" not in base_context_from_previous: # For first scene if no prev info in base
         context_parts.append("\nThis is the first scene of the video.")


    return "\n\n".join(context_parts)



def get_scene_events_from_model(chosen_model_type, model_client_instance, scene_data, video_path,
                                scene_idx, # pass scene_idx
                                base_context_for_current_scene, # This is the evolving context string
                                video_category):
    scene_duration = scene_data.get("duration", 0.0)
    scene_number_display = scene_data.get('scene_number', scene_idx + 1)


    context_block = prepare_context_block_for_scene(
        base_context_for_current_scene,
        video_category,
        scene_data,
        scene_idx
    )

    user_prompt = PROMPT_TEMPLATE.format(
        scene_duration=scene_duration,
        context_block=context_block
    )

    print(f"\n--- Sending to {chosen_model_type.upper()} for Scene {scene_number_display} ---")
    # print(f"--- USER PROMPT for {chosen_model_type.upper()} ---")
    # print(user_prompt)
    # print("--- END USER PROMPT ---")


    encoded_video = encode_video_to_base64(video_path)
    video_part = {"mime_type": "video/mp4", "data": encoded_video}

    model_specific_config = MODEL_CONFIGS[chosen_model_type]
    max_retries = model_specific_config.get("max_retries", 2)
    events = []

    for attempt in range(max_retries):
        try:
            if chosen_model_type == MODEL_GEMINI:
                response = model_client_instance.generate_content(
                    [user_prompt, video_part],
                    generation_config=model_specific_config["generation_config"],
                    safety_settings=model_specific_config["safety_settings"],
                    request_options={"timeout": 240}
                )
                response_text = ""
                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    response_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))

                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    print(f"Prompt blocked: {response.prompt_feedback.block_reason}. Safety: {response.prompt_feedback.safety_ratings}")
                if response.candidates and response.candidates[0].finish_reason.name not in ["STOP", "MAX_TOKENS"]:
                    print(f"Generation finished due to: {response.candidates[0].finish_reason.name}")
                    if response.candidates[0].finish_reason.name == "SAFETY":
                        print(f"Safety ratings: {response.candidates[0].safety_ratings}")
                        return []
                events = extract_and_parse_json(response_text)
                return events

            elif chosen_model_type == MODEL_QWEN:
                system_message_for_api_call = model_specific_config["system_message"]
                completion = model_client_instance.chat.completions.create(
                    model=model_specific_config["model_name"],
                    messages=[
                        {"role": "system", "content": system_message_for_api_call},
                        {"role": "user", "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{encoded_video}"}}
                        ]}
                    ],
                    **model_specific_config["generation_config"]
                )
                response_text = completion.choices[0].message.content
                print(f'QWEN RESPONSE (first 300 chars): {response_text[:300]}...')
                events = extract_and_parse_json(response_text)
                return events

        except Exception as e:
            print(f"Error calling {chosen_model_type.upper()} API (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return []
            time.sleep(5 * (attempt + 1))
    return []

def process_video_folder(video_folder_path, model_client_instance, chosen_model_type, output_suffix):
    video_id = os.path.basename(os.path.normpath(video_folder_path))
    metadata_path = os.path.join(video_folder_path, f"{video_id}.json")
    scenes_input_json_path = os.path.join(video_folder_path, f"{video_id}_scenes", "scene_info.json")

    if not os.path.exists(metadata_path):
        print(f"Metadata file not found: {metadata_path}")
        return
    with open(metadata_path, "r", encoding="utf-8") as f:
        video_metadata = json.load(f)
    video_title = video_metadata.get("title", "Untitled Video")
    video_description = video_metadata.get("description", "")
    video_category = video_metadata.get("category", "Other")

    if not os.path.exists(scenes_input_json_path):
        print(f"Scene info file not found: {scenes_input_json_path}")
        return
    with open(scenes_input_json_path, "r", encoding="utf-8") as f:
        scene_list = json.load(f)

    print(f"Processing {len(scene_list)} scenes for video: '{video_title}' using {chosen_model_type.upper()}")

    context_for_api_call = f"Video Title: {video_title}"
    if video_description:
        context_for_api_call += f"\nVideo Description: {video_description}"
    context_for_api_call += "\n\nPREVIOUS SCENE INFORMATION: This is the first scene, or previous visual not available."


    for i, scene_data in enumerate(scene_list):
        original_scene_path = scene_data.get('scene_path')
        scene_number = scene_data.get('scene_number', i + 1) # 1-based for display/logging

        if not original_scene_path or not os.path.exists(original_scene_path):
            print(f"Scene {scene_number}: Path missing or file not found. Skipping.")
            scene_data['audio_clips'] = []
            # If a scene is skipped, context_for_api_call remains from the last successfully processed scene.
            continue

        converted_path = None
        try:
            converted_path = convert_video_for_processing(original_scene_path, chosen_model_type)
            scene_events_raw = get_scene_events_from_model(
                chosen_model_type, model_client_instance,
                scene_data, converted_path,
                i, # Pass 0-based scene_idx
                context_for_api_call, # This is the base context string
                video_category
            )

            processed_events = []
            unique_texts_on_screen = {}
            current_scene_visual_texts = []

            if isinstance(scene_events_raw, list):
                for event in scene_events_raw:
                    if not isinstance(event, dict) or "start_time" not in event or "type" not in event or "text" not in event:
                        print(f"Skipping malformed event: {event}")
                        continue
                    try:
                        event["start_time"] = float(event["start_time"])
                    except (ValueError, TypeError):
                        event["start_time"] = 0.0

                    event_type = event["type"]
                    if chosen_model_type == MODEL_QWEN and event_type.lower() == "text":
                        event_type = "Text on Screen"
                        event["type"] = "Text on Screen"

                    if event_type == "Text on Screen":
                        text_content = event["text"].strip()
                        if text_content and text_content not in unique_texts_on_screen:
                            unique_texts_on_screen[text_content] = True
                            processed_events.append(event)
                    elif event_type == "Visual":
                        processed_events.append(event)
                        if event["text"]:
                            current_scene_visual_texts.append(event["text"].strip())
                    else:
                        print(f"Skipping event with unknown or non-standard type '{event_type}': {event}")
            else:
                print(f"Warning: Scene {scene_number} returned non-list events: {scene_events_raw}")

            processed_events.sort(key=lambda e: e.get("start_time", 0))
            scene_data['audio_clips'] = processed_events

            next_base_context = f"Video Title: {video_title}"
            if video_description:
                next_base_context += f"\nVideo Description: {video_description}"

            if current_scene_visual_texts:
                next_base_context += f"\n\nPREVIOUS SCENE INFORMATION (Scene {scene_number}):\nKey Visual: {current_scene_visual_texts[0]}"
            else:
                next_base_context += f"\n\nPREVIOUS SCENE INFORMATION (Scene {scene_number}): No distinct key visual identified in this scene."
            context_for_api_call = next_base_context


        except Exception as e:
            print(f"Failed to process scene {scene_number}: {e}")
            import traceback
            traceback.print_exc()
            scene_data['audio_clips'] = []
        finally:
            if converted_path and converted_path != original_scene_path and os.path.exists(converted_path):
                try:
                    os.remove(converted_path)
                except OSError as e_remove:
                    print(f"Warning: Could not remove temp file {converted_path}: {e_remove}")

    input_dir = os.path.dirname(scenes_input_json_path)
    final_output_path = os.path.join(input_dir, f"scene_info_{output_suffix}.json")
    with open(final_output_path, "w", encoding="utf-8") as f:
        json.dump(scene_list, f, indent=4)
    print(f"\nProcessing complete. Updated scene descriptions saved to: {final_output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate video scene descriptions using a chosen LLM API.")
    parser.add_argument("video_folder", help="Folder containing video files and metadata.")
    parser.add_argument("--model", type=str, choices=["gemini", "qwen"], required=True,
                        help="Choose the model: 'gemini' or 'qwen'.")
    args = parser.parse_args()

    model_client = None
    output_file_suffix = ""

    if args.model == MODEL_GEMINI:
        google_api_key = os.getenv("GEMINI_API_KEY")
        if not google_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set for Gemini model.")
        genai.configure(api_key=google_api_key)
        print(f"Initializing Gemini model: {MODEL_CONFIGS[MODEL_GEMINI]['model_name']}")
        model_client = genai.GenerativeModel(
            MODEL_CONFIGS[MODEL_GEMINI]['model_name'],
            system_instruction=MODEL_CONFIGS[MODEL_GEMINI]['system_instruction']
        )
        output_file_suffix = "gemini"

    elif args.model == MODEL_QWEN:
        qwen_api_key = os.getenv("QWEN_API_KEY") 
        if not qwen_api_key:
            raise ValueError("QWEN_API_KEY (or API_KEY) environment variable not set for Qwen model.")

        print(f"Initializing Qwen client for model: {MODEL_CONFIGS[MODEL_QWEN]['model_name']}")
        model_client = OpenAI(
            api_key=qwen_api_key,
            base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
        output_file_suffix = "qwen"
    else:
        raise ValueError(f"Invalid model type specified: {args.model}")

    process_video_folder(args.video_folder, model_client, args.model, output_file_suffix)

if __name__ == "__main__":
    main()