import os
import json
import re
import ast
import argparse
import subprocess
import time
import torch
from google import genai
from google.genai import types
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

INSTRUCTIONAL_VOICE_RULE = """
        ### INSTRUCTIONAL VOICE (this is a how-to/recipe/tutorial video):
        Use IMPERATIVE voice. Describe each step as the action itself, NOT as "a hand" or "the person" performing it.
        - BAD: "A hand pours water into the bowl."   GOOD: "Pour water into the bowl."
        - BAD: "The person whisks the eggs."          GOOD: "Whisk the eggs."
        - BAD: "Someone places the dough on a tray." GOOD: "Place the dough on a tray."
        Do not name a generic actor (hand, person, someone, you) when the action alone conveys the step.
"""

PROMPT_TEMPLATE = """
        Scene Duration: {scene_duration:.2f} seconds

        CONTEXT FOR CURRENT SCENE ANALYSIS:
        {context_block}

        You are analyzing a video scene for a blind and low-vision audience. Identify specific characters, locations, and any important elements mentioned in the context.

        ============================================================
        STEP 1: Text on Screen Events ("type": "Text on Screen")
        ============================================================
        **ACCESSIBILITY PRIORITY:** You MUST transcribe ALL major text that appears on screen, including narrative text blocks, slide titles, bullet points, and checklists. This text is NOT in the audio, so blind viewers rely on your transcription to understand the story.
        
        Capture text that meets these criteria:
        1. It is clearly visible narrative text.
        2. It is a heading, name, date, or informational list.
        
        ONLY EXCLUDE:
        - Brand/Network logos or watermarks.
        - Social media handles, URLs, or copyright fine print.
        - Decorative background text that has no narrative value.

        ============================================================
        STEP 2: Visual Events ("type": "Visual")
        ============================================================
        - Provide a precise, context-rich visual description using minimal but impactful words.
        - Describe each visual event in this scene in specific details.
        - Focus on key actions, settings, and objects that aren't mentioned in previous descriptions.
        - IMPORTANT: DO NOT repeat the "Text on Screen" content within a "Visual" event description.
        - DO NOT REPEAT visual events from previous scenes.

        ### RULES FOR DESCRIBING PEOPLE:
            - STRICTLY PROHIBITED: Never use the real names of actors or celebrities. Use character names from context.
            - If character names are provided (e.g., Jane), you MUST use them.
            - If NO names are available, use neutral descriptive terms.
            
        {voice_rule}
        
        ### SELF-CHECK BEFORE RESPONDING
        1. "Did I transcribe the checklists and narrative text blocks as 'Text on Screen' events?" (REQUIRED)
        2. "Did I use character names from the history for every 'Visual' event?" (REQUIRED)

        ============================================================
        TIMING REQUIREMENTS (CRITICAL)
        ============================================================
        - "timestamp" is in MM:SS format, RELATIVE TO THE START OF THIS SCENE.
        - The scene starts at 00:00 and ends at approximately {scene_mmss}.
        - Report the precise moment each event BEGINS. 
        - Observe the video; do not guess or default to sequential seconds.

        ### OUTPUT FORMAT (STRICT):
            Return ONLY a JSON object with this EXACT structure:
            {{"events": [ {{"timestamp": "MM:SS", "type": "Visual" or "Text on Screen", "text": "<description>"}}, ... ]}}

            - The `events` field MUST always be an array.
            - Do NOT wrap in markdown fences (no ```json).
            - If there are no events, return: {{"events": []}}.

            Now generate the JSON for this scene.
        """

MODEL_CONFIGS = {
    MODEL_GEMINI: {
        "model_name": "gemini-3-flash-preview",
        "system_instruction": f"You are an expert audio describer AI describing video content to blind and low vision audiences. You describe ONLY what is clearly visible in the frames. You NEVER invent objects, characters, or actions that are not actually shown. You always use character names rather than bare pronouns.\n{AUDIO_DESCRIPTION_GUIDELINES}",
        "max_retries": 2,
        "video_fps": 4.5,  
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
        "model_name": "gpt-4o",
        "system_instruction": f"You are an expert video analysis AI. You describe ONLY what is clearly visible in the frames. You NEVER invent objects, characters, or actions that are not actually shown — hallucinating content is the worst possible failure. You ALWAYS use character names from the provided context (including visual history from previous scenes) instead of generic terms like 'man' or 'woman' whenever any name is available. You are also VERY selective about Text on Screen events — most visible text in videos is NOT worth describing. Only include text that a blind viewer absolutely needs to know and that is not already in the audio.\n{AUDIO_DESCRIPTION_GUIDELINES}",
        "max_retries": 2,
        "generation_config": {
            "max_tokens": 512,
            "temperature": 0.3,
            "response_format": {"type": "json_object"}
        },
        # Frame sampling: scales with scene duration.
        "image_detail": "low",
        "seconds_per_frame": 0.5,
        "min_frames_per_scene": 4,
        "max_frames_per_scene": 60
    }
}

# Categories that should use instructional voice (no "a hand", "the person", etc.).
INSTRUCTIONAL_CATEGORIES = {
    "howto & style", "howto", "education", "science & technology",
    "recipe", "tutorial", "diy", "cooking",
}


def is_instructional_category(video_category: str) -> bool:
    """Return True if the video category warrants instructional-voice descriptions."""
    if not video_category:
        return False
    cat_lower = video_category.lower().strip()
    return any(keyword in cat_lower for keyword in INSTRUCTIONAL_CATEGORIES)


# JSON schema for Gemini structured output. Forces consistent shape and types.
GEMINI_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp": {
                        "type": "string",
                        "description": "Time in MM:SS format relative to start of scene"
                    },
                    "type": {
                        "type": "string",
                        "enum": ["Visual", "Text on Screen"]
                    },
                    "text": {"type": "string"},
                },
                "required": ["timestamp", "type", "text"],
            },
        }
    },
    "required": ["events"],
}


def seconds_to_mmss(total_seconds: float) -> str:
    """Convert seconds (float) to MM:SS string."""
    total_seconds = max(0, int(round(total_seconds)))
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


def mmss_to_seconds(timestamp) -> float:
    """Convert MM:SS string (or already-numeric value) to float seconds. Returns 0.0 on parse failure."""
    if isinstance(timestamp, (int, float)):
        return float(timestamp)
    if not isinstance(timestamp, str):
        return 0.0
    s = timestamp.strip()
    parts = s.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        return float(s)
    except (ValueError, TypeError):
        return 0.0


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


def _coerce_to_event_list(data) -> list:
    if isinstance(data, list):
        return [e for e in data if isinstance(e, dict)]

    if isinstance(data, dict):
        for key in ("events", "audio_clips", "clips", "results", "data"):
            if key in data and isinstance(data[key], list):
                return [e for e in data[key] if isinstance(e, dict)]

        if "text" in data and ("type" in data or "start_time" in data or "timestamp" in data):
            return [data]

        for v in data.values():
            if isinstance(v, list) and all(isinstance(e, dict) for e in v):
                return v

    return []


def extract_and_parse_json(response_text: str) -> list:
    if not response_text:
        return []

    cleaned = re.sub(r'```(?:json)?', '', response_text).replace('```', '').strip()

    for loader in (json.loads, ast.literal_eval):
        try:
            data = loader(cleaned)
            events = _coerce_to_event_list(data)
            if events or isinstance(data, (list, dict)):
                return events
        except (json.JSONDecodeError, ValueError, SyntaxError):
            pass

    array_match = re.search(r'\[\s*{[\s\S]*?}\s*\]', cleaned, re.DOTALL)
    if array_match:
        snippet = array_match.group(0)
        for loader in (json.loads, ast.literal_eval):
            try:
                return _coerce_to_event_list(loader(snippet))
            except (json.JSONDecodeError, ValueError, SyntaxError):
                continue

    object_match = re.search(r'\{[\s\S]*\}', cleaned, re.DOTALL)
    if object_match:
        snippet = object_match.group(0)
        for loader in (json.loads, ast.literal_eval):
            try:
                return _coerce_to_event_list(loader(snippet))
            except (json.JSONDecodeError, ValueError, SyntaxError):
                continue

    print(f"Warning: Could not parse JSON from response: {response_text[:200]}...")
    return []


def normalize_event_timing(event: dict, scene_duration: float) -> dict:
    """Convert any timestamp/start_time field to a clamped float `start_time` in seconds."""
    raw = event.get("timestamp", event.get("start_time", 0))
    seconds = mmss_to_seconds(raw)

    if scene_duration > 0:
        seconds = max(0.0, min(seconds, scene_duration))
    else:
        seconds = max(0.0, seconds)

    event["start_time"] = round(seconds, 2)
    event.pop("timestamp", None)
    return event


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


def extract_video_frames(video_path: str, seconds_per_frame: float = 1.0, max_frames: int = None) -> list:
    """Extracts frames from a video file at a given interval, optionally capped at max_frames."""
    base64_frames = []
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return []

    fps = video.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"Warning: Could not determine FPS for video {video_path}. Using default frame interval.")
        frame_interval = 25
    else:
        frame_interval = max(1, int(fps * seconds_per_frame))

    frame_count = 0
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        if frame_count % frame_interval == 0:
            _, buffer = cv2.imencode(".jpg", frame)
            base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
            if max_frames is not None and len(base64_frames) >= max_frames:
                break
        frame_count += 1

    video.release()
    print(f"Extracted {len(base64_frames)} frames from scene (interval: {seconds_per_frame:.2f}s).")
    return base64_frames


def post_filter_text_events(events: list, transcript_data: list) -> list:
    """Post-processing filter to remove Text on Screen events that overlap with transcript content."""
    if not events:
        return events

    all_transcript_text = " ".join(
        seg.get('text', '') for seg in transcript_data
    ).lower()

    filtered_events = []
    for event in events:
        if event.get('type') != 'Text on Screen':
            filtered_events.append(event)
            continue

        text_content = event.get('text', '').strip().lower()
        if not text_content:
            continue

        words = text_content.split()
        if len(words) <= 2:
            if text_content in all_transcript_text:
                print(f"  [POST-FILTER] Removed Text on Screen (in transcript): \"{event['text']}\"")
                continue
        else:
            matching_words = sum(1 for w in words if w in all_transcript_text)
            overlap_ratio = matching_words / len(words)
            if overlap_ratio >= 0.6:
                print(f"  [POST-FILTER] Removed Text on Screen (60%+ word overlap with transcript): \"{event['text']}\"")
                continue

        skip_patterns = [
            r'^https?://',
            r'^@',
            r'^#',
            r'©|®|™',
            r'subscribe',
            r'follow\s+(me|us)',
            r'like\s+and\s+share',
            r'\.(com|org|net|io)',
        ]
        should_skip = False
        for pattern in skip_patterns:
            if re.search(pattern, text_content, re.IGNORECASE):
                print(f"  [POST-FILTER] Removed Text on Screen (matches skip pattern): \"{event['text']}\"")
                should_skip = True
                break

        if not should_skip:
            filtered_events.append(event)

    removed_count = len(events) - len(filtered_events)
    if removed_count > 0:
        print(f"  [POST-FILTER] Removed {removed_count} Text on Screen events total")

    return filtered_events


def get_scene_events_from_model(chosen_model_type, model_client, scene_data, video_path,
                                scene_idx, base_context_for_current_scene, video_category):
    scene_duration = scene_data.get("duration", 0.0)
    scene_number_display = scene_data.get('scene_number', scene_idx + 1)
    context_block = prepare_context_block_for_scene(
        base_context_for_current_scene, video_category, scene_data, scene_idx)
    scene_mmss = seconds_to_mmss(scene_duration)
    voice_rule = INSTRUCTIONAL_VOICE_RULE if is_instructional_category(video_category) else ""
    user_prompt = PROMPT_TEMPLATE.format(
        scene_duration=scene_duration,
        scene_mmss=scene_mmss,
        context_block=context_block,
        voice_rule=voice_rule,
    )
    if is_instructional_category(video_category):
        print(f"  [voice] Using INSTRUCTIONAL voice for category: {video_category}")
        
    print(f"\n--- Processing Scene {scene_number_display} with {chosen_model_type.upper()} ---")
    model_specific_config = MODEL_CONFIGS[chosen_model_type]
    max_retries = model_specific_config.get("max_retries", 2)

    for attempt in range(max_retries):
        try:
            if chosen_model_type == MODEL_GEMINI:
                with open(video_path, "rb") as video_file:
                    video_bytes = video_file.read()

                video_fps = model_specific_config.get("video_fps", 1.0)
                video_part = types.Part(
                    inline_data=types.Blob(mime_type="video/mp4", data=video_bytes),
                    video_metadata=types.VideoMetadata(fps=video_fps),
                )

                response = model_client["client"].models.generate_content(
                    model=model_client["model_name"],
                    contents=[video_part, user_prompt],
                    config=types.GenerateContentConfig(
                        system_instruction=model_client["system_instruction"],
                        temperature=0.0,
                        max_output_tokens=8912,
                        thinking_config=types.ThinkingConfig(thinking_budget=512),
                        response_mime_type="application/json",
                        response_schema=GEMINI_RESPONSE_SCHEMA,
                        safety_settings=[
                            types.SafetySetting(category=c, threshold=types.HarmBlockThreshold.BLOCK_NONE)
                            for c in (
                                types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                                types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            )
                        ],
                    ),
                )
                response_text = response.text

                # Diagnostics — useful when things go wrong, harmless when they don't.
                try:
                    candidate = response.candidates[0]
                    finish_reason = getattr(candidate, "finish_reason", "unknown")
                    print(f"[DEBUG] finish_reason: {finish_reason}")
                    if response.usage_metadata:
                        um = response.usage_metadata
                        print(f"[DEBUG] tokens — prompt: {um.prompt_token_count}, "
                              f"output: {um.candidates_token_count}, "
                              f"thoughts: {getattr(um, 'thoughts_token_count', 'n/a')}")
                except Exception:
                    pass

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
                seconds_per_frame = model_specific_config.get("seconds_per_frame", 1.0)
                min_frames = model_specific_config.get("min_frames_per_scene", 4)
                max_frames = model_specific_config.get("max_frames_per_scene", 60)

                if scene_duration > 0:
                    target_count = int(round(scene_duration / seconds_per_frame))
                else:
                    target_count = min_frames
                target_count = max(min_frames, min(max_frames, target_count))

                interval = scene_duration / target_count if scene_duration > 0 else seconds_per_frame

                base64_frames = extract_video_frames(
                    video_path, seconds_per_frame=interval, max_frames=target_count
                )

                if not base64_frames:
                    print("Skipping API call as no frames were extracted.")
                    return []

                image_detail = model_specific_config.get("image_detail", "high")
                prompt_content = [{"type": "text", "text": user_prompt}]
                for frame in base64_frames:
                    prompt_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame}",
                            "detail": image_detail
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
    context_for_api_call += "\n\nPREVIOUS SCENE INFORMATION: This is the first scene."

    all_transcript_data = []
    for scene_data in scene_list:
        for segment in scene_data.get('transcript', []):
            all_transcript_data.append(segment)

    all_prior_scene_visuals = []

    for i, scene_data in enumerate(scene_list):
        original_scene_path = scene_data.get('scene_path')
        scene_number = scene_data.get('scene_number', i + 1)
        scene_duration = scene_data.get("duration", 0.0)

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
                    if not isinstance(event, dict):
                        continue
                    if "type" not in event or "text" not in event:
                        continue
                    if "timestamp" not in event and "start_time" not in event:
                        continue

                    event = normalize_event_timing(event, scene_duration)

                    if event["type"] == "Visual":
                        current_scene_visual_texts.append(event["text"].strip())
                    processed_events.append(event)

            scene_transcript = scene_data.get('transcript', [])
            processed_events = post_filter_text_events(processed_events, scene_transcript + all_transcript_data)

            scene_data['audio_clips'] = sorted(processed_events, key=lambda e: e.get("start_time", 0))

            all_prior_scene_visuals.append({
                "scene_number": scene_number,
                "visuals": list(current_scene_visual_texts)
            })

            next_base_context = f"Video Title: {video_title}"
            if video_description:
                next_base_context += f"\nVideo Description: {video_description}"

            history_lines = []
            for entry in all_prior_scene_visuals:
                if not entry["visuals"]:
                    continue
                for v in entry["visuals"]:
                    history_lines.append(f"[Scene {entry['scene_number']}] {v}")

            if history_lines:
                history_block = "\n".join(history_lines)
                next_base_context += (
                    "\n\nPREVIOUS SCENES — VISUAL HISTORY "
                    "(use character names that appear here whenever those characters reappear):\n"
                    f"{history_block}"
                )
            else:
                next_base_context += "\n\nPREVIOUS SCENE INFORMATION: No prior visuals recorded yet."

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
        if not google_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")

        print(f"Initializing Gemini API for model: {MODEL_CONFIGS[MODEL_GEMINI]['model_name']}")
        client = genai.Client(api_key=google_api_key)
        model_client = {
            "client": client,
            "model_name": MODEL_CONFIGS[MODEL_GEMINI]["model_name"],
            "system_instruction": MODEL_CONFIGS[MODEL_GEMINI]["system_instruction"],
        }
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