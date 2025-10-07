import os
import json
import re
import ast
import argparse
import subprocess
import time
import torch
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import openai
import base64
import cv2  

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from dotenv import load_dotenv

load_dotenv()

MODEL_GEMINI = "gemini"
MODEL_QWEN = "qwen"
MODEL_GPT4 = "gpt"

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
            - Provide a precise, context-rich visual description using minimal but impactful words.
            - Describe each action in this scene in every specific details. 
            - ALWAYS use specific CHARACTER names from context (not "person" or "woman"). 
            - Focus on key actions, settings, objects that aren't mentioned in previous description
            - Include clear start times for each visual event
            - IMPORTANT: DO NOT repeat Text on Screen as Visual events.
            - DO NOT REPEAT visual events from previous scenes. 
            
        ### RULES FOR DESCRIBING PEOPLE:
            - CRITICAL: It is STRICTLY PROHIBITED to use the real names of actors, celebrities, or any public figures, even if you recognize them. This is a top-priority rule.
            - If character names are provided in the context above, you must use them.
            - If NO character names are available in the context, you MUST describe people using neutral, descriptive terms based on their appearance (e.g., "a young woman with reddish-brown hair," "the older woman driving the car," "a man in a red shirt"). Do NOT default to using actor names as a substitute for character names.

        ### OUTPUT:
            - Format the output as a JSON array. Each event should include:
            - `start_time` (in seconds) - exact time when event happened.
            - `type` ("Text on Screen" or "Visual")
            - `text` (description or on-screen text)       
            Now generate the JSON array of events for this scene.
        """

MODEL_CONFIGS = {
    MODEL_GEMINI: {
        "model_name": "gemini-2.5-pro",
        "system_instruction": f"You are an expert video analysis AI...\n{AUDIO_DESCRIPTION_GUIDELINES}",
        "generation_config": {
            "temperature": 0.6,
            "max_output_tokens": 512,
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
        "model_path": "Qwen/Qwen2.5-VL-72B-Instruct",
        "max_retries": 3,
        "generation_config": {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
        }
    },
    MODEL_GPT4: {
        "model_name": "gpt-4o",  # Use "gpt-4.1-..." when available
        "system_instruction": f"You are an expert video analysis AI...\n{AUDIO_DESCRIPTION_GUIDELINES}",
        "max_retries": 2,
        "generation_config": {
            "max_tokens": 512,
            "temperature": 0.6,
            "response_format": {"type": "json_object"}
        }
    }
}


def standardize_video_for_processing(input_path: str) -> str:
    input_dir = os.path.dirname(input_path)
    base_name, ext = os.path.splitext(os.path.basename(input_path))
    output_path = os.path.join(input_dir, f"{base_name}_temp{ext}")
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", input_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", output_path
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg failed to convert {input_path}: {e.stderr.decode()}. Using original path.")
        return input_path


def extract_and_parse_json(response_text: str) -> list:
    if not response_text: return []
    try:
        response_text = re.sub(r'```json|```', '', response_text).strip()
        json_match = re.search(r'^\s*\[[\s\S]*?\]\s*$', response_text, re.MULTILINE)
        if not json_match: json_match = re.search(r'\[\s*{[\s\S]*?}\s*\]', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                return ast.literal_eval(json_str)
        else:
            print(f"Warning: No valid JSON array found in response: {response_text[:200]}...")
            return []
    except Exception as e:
        print(f"Error parsing JSON response: {e}")
        return []


def prepare_context_block_for_scene(base_context, video_category, current_scene_data, scene_idx):
    context_parts = [base_context, f"\nVideo Category: {video_category}"]
    current_scene_context = []
    if current_scene_data.get("transcript"):
        transcripts = "\n".join(f"- {t['text']}" for t in current_scene_data["transcript"])
        current_scene_context.append(f"TRANSCRIPT FOR CURRENT SCENE (Scene {scene_idx + 1}):\n{transcripts}")
    if current_scene_data.get("captions"):
        captions = "\n".join(f"- {c['text']}" for c in current_scene_data["captions"])
        current_scene_context.append(f"CAPTIONS FOR CURRENT SCENE (Scene {scene_idx + 1}):\n{captions}")
    if current_scene_context:
        context_parts.append("\n" + "\n".join(current_scene_context))
    elif scene_idx == 0:
        context_parts.append("\nThis is the first scene of the video.")
    return "\n\n".join(context_parts)


def extract_video_frames(video_path: str, seconds_per_frame: int = 1) -> list:
    """Extracts frames from a video file at a given interval."""
    base64_frames = []
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return []

    fps = video.get(cv2.CAP_PROP_FPS)
    # Handle cases where fps is 0 to avoid division by zero error
    if fps == 0:
        print(f"Warning: Could not determine FPS for video {video_path}. Using default frame interval.")
        frame_interval = 25 # Default to extracting approximately 1 frame per second for a 25fps video
    else:
        frame_interval = int(fps * seconds_per_frame)

    frame_count = 0
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        if frame_count % frame_interval == 0:
            _, buffer = cv2.imencode(".jpg", frame)
            base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
        frame_count += 1

    video.release()
    print(f"Extracted {len(base64_frames)} frames from scene.")
    return base64_frames


def get_scene_events_from_model(chosen_model_type, model_client, scene_data, video_path,
                                scene_idx, base_context_for_current_scene, video_category):
    scene_duration = scene_data.get("duration", 0.0)
    scene_number_display = scene_data.get('scene_number', scene_idx + 1)
    context_block = prepare_context_block_for_scene(
        base_context_for_current_scene, video_category, scene_data, scene_idx)
    user_prompt = PROMPT_TEMPLATE.format(
        scene_duration=scene_duration, context_block=context_block)

    print(f"\n--- Processing Scene {scene_number_display} with {chosen_model_type.upper()} ---")
    model_specific_config = MODEL_CONFIGS[chosen_model_type]
    max_retries = model_specific_config.get("max_retries", 2)

    for attempt in range(max_retries):
        try:
            if chosen_model_type == MODEL_GEMINI:
                with open(video_path, "rb") as video_file:
                    video_part = {"mime_type": "video/mp4", "data": video_file.read()}
                response = model_client.generate_content(
                    [user_prompt, video_part],
                    generation_config=model_specific_config["generation_config"],
                    safety_settings=model_specific_config["safety_settings"],
                    request_options={"timeout": 240}
                )
                response_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))

                print("\n--- Raw Gemini Response ---")
                print(response_text)
                print("--- End Gemini Response ---\n")

                return extract_and_parse_json(response_text)

            elif chosen_model_type == MODEL_QWEN:
                model = model_client['model']
                processor = model_client['processor']

                messages = [{"role": "user", "content": [{"type": "text", "text": user_prompt}, {"type": "video", "video": video_path}]}]

                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)

                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to(model.device)

                output_ids = model.generate(**inputs, **model_specific_config["generation_config"])

                input_token_len = inputs.input_ids.shape[1]
                generated_ids = output_ids[:, input_token_len:]
                response_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

                print("\n--- Raw Local Qwen Response ---")
                print(response_text)
                print("--- End Local Qwen Response ---\n")

                return extract_and_parse_json(response_text)

            elif chosen_model_type == MODEL_GPT4:
                approx_tokens_per_frame = 2000
                target_frames = 30000 // approx_tokens_per_frame  
                interval = max(1, int(scene_duration / target_frames))        
                base64_frames = extract_video_frames(video_path, seconds_per_frame=interval)

                if not base64_frames:
                    print("Skipping API call as no frames were extracted.")
                    return []

                prompt_content = [{"type": "text", "text": user_prompt}]

                for frame in base64_frames:
                    prompt_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame}",
                            "detail": "low"
                        }
                    })

                response = model_client.chat.completions.create(
                    model=model_specific_config["model_name"],
                    messages=[
                        {
                            "role": "system",
                            "content": model_specific_config["system_instruction"]
                        },
                        {
                            "role": "user",
                            "content": prompt_content
                        }
                    ],
                    **model_specific_config["generation_config"]
                )
                response_text = response.choices[0].message.content

                print("\n--- Raw GPT-4 Response ---")
                print(response_text)
                print("--- End GPT-4 Response ---\n")

                return extract_and_parse_json(response_text)

        except Exception as e:
            print(f"Error during model generation (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return []
            time.sleep(5 * (attempt + 1))
    return []


def process_video_folder(video_folder_path, model_client, chosen_model_type, output_suffix):
    video_id = os.path.basename(os.path.normpath(video_folder_path))
    metadata_path = os.path.join(video_folder_path, f"{video_id}.json")
    scenes_input_json_path = os.path.join(video_folder_path, f"{video_id}_scenes", "scene_info.json")

    with open(metadata_path, "r", encoding="utf-8") as f:
        video_metadata = json.load(f)
    video_title = video_metadata.get("title", "Untitled Video")
    video_description = video_metadata.get("description", "")
    video_category = video_metadata.get("category", "Other")

    with open(scenes_input_json_path, "r", encoding="utf-8") as f:
        scene_list = json.load(f)
    print(f"Processing {len(scene_list)} scenes for video: '{video_title}'...")

    context_for_api_call = f"Video Title: {video_title}"

    if video_description:
        context_for_api_call += f"\nVideo Description: {video_description}"

    context_for_api_call += "\n\nPREVIOUS SCENE INFORMATION: This is the first scene, or previous visual not available."

    for i, scene_data in enumerate(scene_list):
        original_scene_path = scene_data.get('scene_path')
        scene_number = scene_data.get('scene_number', i + 1)

        if not original_scene_path or not os.path.exists(original_scene_path):
            print(f"Scene {scene_number}: Path missing or file not found. Skipping.")
            scene_data['audio_clips'] = []
            continue

        compatible_path = None
        try:
            start_time = time.time()
            compatible_path = standardize_video_for_processing(original_scene_path)

            scene_events_raw = get_scene_events_from_model(
                chosen_model_type, model_client, scene_data, compatible_path,
                i, context_for_api_call, video_category
            )

            processed_events = []
            current_scene_visual_texts = []
            if isinstance(scene_events_raw, list):
                for event in scene_events_raw:
                    if not (isinstance(event, dict) and "start_time" in event and "type" in event and "text" in event): continue
                    try:
                        event["start_time"] = float(event["start_time"])
                    except (ValueError, TypeError):
                        event["start_time"] = 0.0

                    if event["type"] == "Visual":
                        current_scene_visual_texts.append(event["text"].strip())
                    processed_events.append(event)

            scene_data['audio_clips'] = sorted(processed_events, key=lambda e: e.get("start_time", 0))

            next_base_context = f"Video Title: {video_title}"
            if video_description: next_base_context += f"\nVideo Description: {video_description}"
            if current_scene_visual_texts:
                next_base_context += f"\n\nPREVIOUS SCENE INFORMATION (Scene {scene_number}):\nKey Visual: {current_scene_visual_texts[0]}"
            else:
                next_base_context += f"\n\nPREVIOUS SCENE INFORMATION (Scene {scene_number}): No distinct key visual identified."
            context_for_api_call = next_base_context

            end_time = time.time()
            print(f"Scene processing finished in {end_time - start_time:.2f} seconds.")

        except Exception as e:
            print(f"FATAL: Failed to process scene {scene_number}: {e}")
            import traceback
            traceback.print_exc()
            scene_data['audio_clips'] = []
        finally:
            if compatible_path and compatible_path != original_scene_path and os.path.exists(compatible_path):
                try:
                    os.remove(compatible_path)
                except OSError as e_remove:
                    print(f"Warning: Could not remove temp file {compatible_path}: {e_remove}")

    final_output_path = os.path.join(os.path.dirname(scenes_input_json_path), f"scene_info_{output_suffix}.json")
    with open(final_output_path, "w", encoding="utf-8") as f:
        json.dump(scene_list, f, indent=4)
    print(f"\nProcessing complete. Updated descriptions saved to: {final_output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate video scene descriptions using various models.")
    parser.add_argument("video_folder", help="Folder containing video files and metadata.")
    parser.add_argument("--model", type=str, choices=[MODEL_GEMINI, MODEL_QWEN, MODEL_GPT4], default='gpt',
                        help="Choose the model: 'gemini', 'qwen', or 'gpt'.")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model_client = None
    output_file_suffix = ""

    if args.model == MODEL_GEMINI:
        google_api_key = os.getenv("GEMINI_API_KEY")
        if not google_api_key: raise ValueError("GEMINI_API_KEY environment variable not set.")
        genai.configure(api_key=google_api_key)

        print(f"Initializing Gemini API for model: {MODEL_CONFIGS[MODEL_GEMINI]['model_name']}")
        model_client = genai.GenerativeModel(
            MODEL_CONFIGS[MODEL_GEMINI]['model_name'],
            system_instruction=MODEL_CONFIGS[MODEL_GEMINI]['system_instruction']
        )
        output_file_suffix = "gemini"

    elif args.model == MODEL_QWEN:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_path = MODEL_CONFIGS[MODEL_QWEN]['model_path']
        print(f"Initializing LOCAL Qwen model with 4-bit quantization from: {model_path}")

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            quantization_config=quantization_config,
            cache_dir="../.cache"
        )

        processor = AutoProcessor.from_pretrained(model_path)

        model_client = {'model': model, 'processor': processor}
        output_file_suffix = "qwen"

    elif args.model == MODEL_GPT4:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key: raise ValueError("OPENAI_API_KEY environment variable not set.")

        print(f"Initializing OpenAI API for model: {MODEL_CONFIGS[MODEL_GPT4]['model_name']}")
        model_client = openai.OpenAI(api_key=openai_api_key)
        output_file_suffix = "gpt"

    process_video_folder(args.video_folder, model_client, args.model, output_file_suffix)


if __name__ == "__main__":
    main()