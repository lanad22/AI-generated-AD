import json
import tempfile
import subprocess
import os
import re
import base64
import argparse
import cv2
from gtts import gTTS
from typing import Dict, List
from dotenv import load_dotenv
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import google as genai
import openai

load_dotenv()

MODEL_QWEN = "qwen"
MODEL_GEMINI = "gemini"
MODEL_GPT = "gpt"

# How many frames to sample per scene when verifying a description visually
# How many frames to sample per scene when verifying a description visually
VERIFICATION_SECONDS_PER_FRAME = 0.5
VERIFICATION_MIN_FRAMES = 4
VERIFICATION_MAX_FRAMES = 30
VERIFICATION_IMAGE_DETAIL = "low"


def get_tts_duration(text):
    if not text or text.isspace():
        return 0.0
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=True) as temp_file:
        try:
            tts = gTTS(text=text, lang='en')
            tts.save(temp_file.name)
            cmd = (f'ffprobe -v error -select_streams a:0 -show_entries format=duration '
                   f'-of csv="p=0" "{temp_file.name}"')
            duration = float(subprocess.check_output(cmd, shell=True).decode().strip())
            return duration
        except Exception as e:
            print(f"Could not get TTS duration for text '{text}': {e}")
            words = len(text.split())
            estimated_duration = words / 2.5
            return max(1.0, estimated_duration)
def extract_scene_frames(video_path: str) -> List[str]:
    """Extract frames evenly spaced across the scene, count scaled by duration."""
    if not video_path or not os.path.exists(video_path):
        return []

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Could not open scene video for verification: {video_path}")
        return []

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS) or 25
    duration = total_frames / fps if fps else 0

    if total_frames <= 0:
        video.release()
        return []

    # Scale with duration, clamp to [min, max]
    target = int(round(duration / VERIFICATION_SECONDS_PER_FRAME))
    num_frames = max(VERIFICATION_MIN_FRAMES, min(VERIFICATION_MAX_FRAMES, target))

    if num_frames >= total_frames:
        target_indices = list(range(total_frames))
    else:
        step = total_frames / num_frames
        target_indices = [int(i * step) for i in range(num_frames)]

    base64_frames = []
    for idx in target_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = video.read()
        if not success:
            continue
        _, buffer = cv2.imencode(".jpg", frame)
        base64_frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    print(f"  Extracted {len(base64_frames)} verification frames ({duration:.1f}s scene)")
    return base64_frames

def _generate_with_qwen(client, prompt, max_tokens, temperature):
    model = client['model']
    processor = client['processor']
    messages = [
        {"role": "system", "content": "You are an expert audio describer."},
        {"role": "user", "content": prompt}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True
    )

    input_token_len = inputs.input_ids.shape[1]
    generated_ids = output_ids[:, input_token_len:]
    response_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    return response_text.strip()


def _build_evaluation_prompt(clip, transcript_text, cumulative_transcript, previous_desc_text, total_candidates):
    return f"""You are an accessibility expert deciding whether a visual description should be included in an audio description track for a blind viewer.

You are looking at frames from the actual scene the description was generated for. Use them to judge whether the description is BOTH accurate AND necessary.

PHILOSOPHY: LESS IS MORE. Extended audio descriptions interrupt the video's natural pacing. A sparse, well-chosen set of descriptions is far better than a dense one. Most candidate descriptions should be rejected. Default to REJECT and only override when truly justified.

### CONTEXT
CURRENT SCENE TRANSCRIPT:
{transcript_text}

CUMULATIVE TRANSCRIPT SO FAR:
{cumulative_transcript}

DESCRIPTIONS ALREADY INCLUDED:
{previous_desc_text}

CANDIDATE DESCRIPTION TO EVALUATE:
{clip['text']}

TOTAL CANDIDATES IN THIS VIDEO: {total_candidates}

### HOW TO DECIDE

STEP 1 — ACCURACY CHECK (look at the frames):
- Are the objects, characters, and actions in the description ACTUALLY VISIBLE in the frames?
- If the description mentions specific things (e.g. "a donut, a wedge of cheese, a strawberry") — verify each one is really there.
- If ANY significant part of the description is not visible in the frames, set accurate = false and necessary = false. Note in the reason that the description appears hallucinated.

STEP 2 — NECESSITY CHECK (only if STEP 1 passes):

REJECT (necessary = false) — DEFAULT TO THIS — if ANY of these apply:
- The description is inaccurate or hallucinated (failed STEP 1).
- A reasonable blind viewer could follow the video without it.
- The transcript, audio cues (music, sound effects, tone of voice), or a prior description already conveys the gist — even partially.
- It describes appearance, setting, decoration, atmosphere, or background — rather than a discrete action or change.
- It restates something already established (same character still on screen, same setting continuing, ongoing state).
- It would feel like an interruption to a sighted viewer watching with the audio description on.
- The visual is decorative rather than informational.

KEEP (necessary = true) — ONLY if ALL of these are true:
- The description is accurate (passed STEP 1).
- It conveys an ACTION, REACTION, CHANGE, REVEAL, or VISUAL GAG — not a static state or appearance.
- That information is genuinely missing from the transcript, audio cues, AND every prior description.
- A blind viewer would be confused or would miss something meaningful without it.
- It is one of the few most important visual moments in the video.

When in doubt, REJECT. The cost of one missing description is small; the cost of cluttered, interrupting descriptions is large.

### OUTPUT FORMAT
Return a JSON object with:
- "necessary": boolean
- "accurate": boolean (whether the description matches what is in the frames)
- "reason": brief explanation that mentions both accuracy and necessity
"""

def evaluate_clip_necessity(client, model_name, clip, transcript_data, previous_descriptions,
                            total_candidates, scene_frames):
    scene_number = clip.get('scene_number', 0)
    scene_transcript = [t for t in transcript_data if t.get('scene_number') == scene_number]

    transcript_text = " ".join([segment.get('text', '') for segment in scene_transcript])
    cumulative_transcript = " ".join([seg.get('text', '') for seg in transcript_data if seg.get('scene_number') <= scene_number])
    previous_desc_text = " ".join([f"[Scene {desc.get('scene_number')}] {desc.get('type')}: {desc.get('text')}" for desc in previous_descriptions])

    prompt = _build_evaluation_prompt(
        clip, transcript_text, cumulative_transcript, previous_desc_text, total_candidates
    )

    try:
        result = ""
        if model_name.startswith('gemini'):
            prompt += '\n\nIMPORTANT: Respond with ONLY the raw JSON object, without using markdown ```json ... ``` wrappers.'
            # Pass frames as inline image parts
            content_parts = [prompt]
            for f_b64 in scene_frames:
                content_parts.append({"mime_type": "image/jpeg", "data": base64.b64decode(f_b64)})
            response = client.generate_content(
                content_parts,
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": 400,
                    "response_mime_type": "application/json",
                }
            )
            result = response.text

        elif model_name == 'local_qwen':
            # Qwen path stays text-only — local multimodal pipeline isn't wired here.
            result = _generate_with_qwen(client, prompt, 400, 0.2)

        else:  # GPT models
            user_content = [{"type": "text", "text": prompt}]
            for f_b64 in scene_frames:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{f_b64}",
                        "detail": VERIFICATION_IMAGE_DETAIL
                    }
                })
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an accessibility expert deciding which visual descriptions a blind viewer needs. You MUST verify accuracy by examining the provided frames before considering necessity. Hallucinated or inaccurate descriptions must be rejected."},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.1,
                max_tokens=400,
                response_format={"type": "json_object"}
            )
            result = response.choices[0].message.content.strip()

        print(f"MODEL RESPONSE: {result}")

        try:
            matches = re.search(r'\{.*\}', result, re.DOTALL)
            json_str = matches.group(0) if matches else result
            analysis = json.loads(json_str)
            necessary = analysis.get('necessary', False)
            accurate = analysis.get('accurate', None)
            reason = analysis.get('reason', "No reason provided")
            # Defensive: if model says inaccurate but somehow set necessary=true, override.
            if accurate is False:
                necessary = False
            return necessary, reason
        except json.JSONDecodeError:
            print(f"Failed to parse JSON for clip in scene {scene_number}.")
            return False, "Failed to parse model response"

    except Exception as e:
        print(f"Error evaluating necessity with model {model_name}: {e}")
        return False, f"Error: {str(e)}"


def optimize_description(client, model_name, clip):
    if not clip:
        return None

    prompt = f"""TASK: Create an extremely concise version of this visual description for an audio description track.

ORIGINAL DESCRIPTION:
{clip['text']}

GUIDELINES:
- Focus ONLY on the most essential visual elements.
- Make it significantly more concise while keeping the most critical information.
- Use natural, conversational language.
- Use clear, vivid language suitable for audio description.
- Maintain a flowing sentence structure.
- Start with the most important element.
- Be extremely concise - every word must earn its place.

OUTPUT:
Provide only the optimized description text, with no extra commentary or quotation marks.
"""

    try:
        optimized_text = ""
        if model_name.startswith('gemini'):
            response = client.generate_content(
                prompt,
                generation_config={"temperature": 1.0, "max_output_tokens": 100}
            )
            optimized_text = response.text.strip()
        elif model_name == 'local_qwen':
            optimized_text = _generate_with_qwen(client, prompt, 100, 1.0)
        else:  # GPT models
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert audio describer who writes concise text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )
            optimized_text = response.choices[0].message.content.strip()

        optimized_text = optimized_text.strip('"')

        tts_duration = get_tts_duration(optimized_text)

        optimized_clip = clip.copy()
        optimized_clip['text'] = optimized_text
        optimized_clip['duration'] = tts_duration
        optimized_clip['end_time'] = clip['start_time'] + tts_duration
        optimized_clip['original_text'] = clip['text']

        return optimized_clip

    except Exception as e:
        print(f"Error optimizing clip with model {model_name}: {e}")
        return clip


def main():
    parser = argparse.ArgumentParser(description="Analyze and optimize visual descriptions for accessibility")
    parser.add_argument("video_folder", help="Path to the video folder containing relevant JSON files")
    parser.add_argument("--model", type=str, choices=[MODEL_GEMINI, MODEL_QWEN, MODEL_GPT], default=MODEL_GPT,
                        help="Choose the model for optimizing descriptions: 'gemini', 'qwen', or 'gpt'.")
    parser.add_argument("--no-analyze-necessity", action="store_true",
                        help="Skip analyzing whether descriptions are necessary (default is to analyze)")

    args = parser.parse_args()

    client = None
    model_to_use = ""
    if args.model == MODEL_GEMINI:
        try:
            model_to_use = "gemini-1.5-pro-latest"
            print(f"\nSetting up Google Gemini client for model: {model_to_use}...")
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("Error: GEMINI_API_KEY environment variable not set. This is required for Gemini models.")
                return
            genai.configure(api_key=api_key)
            client = genai.GenerativeModel(model_to_use)
        except ImportError:
            print("Error: 'google.generativeai' package not found. Please install it using 'pip install google-generativeai'")
            return
    elif args.model == MODEL_QWEN:
        model_to_use = "local_qwen"
        print(f"\nSetting up LOCAL Qwen model...")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-72B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            quantization_config=quantization_config,
            cache_dir="../.cache")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")
        client = {'model': model, 'processor': processor}
    elif args.model == MODEL_GPT:
        try:
            model_to_use = "gpt-4o"
            print(f"\nSetting up OpenAI GPT client for model: {model_to_use}...")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("Error: OPENAI_API_KEY environment variable not set. This is required for GPT models.")
                return
            client = openai.OpenAI(api_key=api_key)
        except ImportError:
            print("Error: 'openai' package not found. Please install it using 'pip install openai'")
            return

    if not client:
        print("Client setup failed. Exiting.")
        return

    video_id = os.path.basename(os.path.normpath(args.video_folder))
    scenes_folder = os.path.join(args.video_folder, f"{video_id}_scenes")

    audio_clips_path = os.path.join(scenes_folder, f"audio_clips_optimized_{args.model}.json")
    preferred_scene_info_path = os.path.join(scenes_folder, f"scene_info_{args.model}.json")
    fallback_scene_info_path = os.path.join(scenes_folder, "scene_info.json")

    scene_info_path = ""
    if os.path.exists(preferred_scene_info_path):
        scene_info_path = preferred_scene_info_path
    elif os.path.exists(fallback_scene_info_path):
        scene_info_path = fallback_scene_info_path
    else:
        print(f"Error: No suitable scene_info file found in {scenes_folder}.")
        print(f"  - Looked for: {os.path.basename(preferred_scene_info_path)}")
        print(f"  - And: {os.path.basename(fallback_scene_info_path)}")
        return

    print(f"\nUsing scene info file: {os.path.basename(scene_info_path)}")
    print(f"Processing audio clips file: {os.path.basename(audio_clips_path)}")

    if not os.path.exists(scene_info_path) or not os.path.exists(audio_clips_path):
        print(f"Error: Required input files not found.")
        if not os.path.exists(scene_info_path):
            print(f"  - Missing: {scene_info_path}")
        if not os.path.exists(audio_clips_path):
            print(f"  - Missing: {audio_clips_path}")
        return

    with open(scene_info_path, "r") as f:
        scene_info = json.load(f)
    with open(audio_clips_path, "r") as f:
        audio_clips = json.load(f)

    transcript_data = []
    for scene in scene_info:
        scene_number = scene.get('scene_number', 0)
        for segment in scene.get('transcript', []):
            transcript_segment = segment.copy()
            transcript_segment['scene_number'] = scene_number
            transcript_data.append(transcript_segment)

    # Build a lookup from scene_number -> scene_path so we can extract verification frames
    scene_path_by_number = {}
    for scene in scene_info:
        sn = scene.get('scene_number')
        sp = scene.get('scene_path')
        if sn is not None and sp:
            scene_path_by_number[sn] = sp

    print(f"Loaded transcript with {len(transcript_data)} segments")
    print(f"Loaded {len(audio_clips)} descriptions from {os.path.basename(audio_clips_path)}")

    non_gap_visuals = [desc for desc in audio_clips if not desc.get('fits_in_gap', True) and desc.get('type') == 'Visual']
    total_candidates = len(non_gap_visuals)
    print(f"\nFound {total_candidates} Visual descriptions where fits_in_gap is false")

    if not non_gap_visuals and not args.no_analyze_necessity:
        print("No Visual descriptions with fits_in_gap=false to process.")
        return

    audio_clips.sort(key=lambda x: (x.get('scene_number', 0), x.get('start_time', 0)))
    final_clips, previous_descriptions = [], []
    clips_kept, clips_removed = 0, 0

    # Cache scene frames so we don't re-extract for every clip in the same scene
    frames_cache = {}

    def get_frames_for_scene(scene_num):
        if scene_num in frames_cache:
            return frames_cache[scene_num]
        path = scene_path_by_number.get(scene_num)
        if not path:
            print(f"  [warn] No scene_path for scene {scene_num}; verification will fall back to text-only context.")
            frames = []
        else:
            frames = extract_scene_frames(path)
            print(f"  Extracted {len(frames)} verification frames for scene {scene_num}")
        frames_cache[scene_num] = frames
        return frames

    for clip in audio_clips:
        is_non_gap_visual = clip.get('type') == 'Visual' and not clip.get('fits_in_gap', True)

        if not args.no_analyze_necessity and is_non_gap_visual:
            print(f"\n===== EVALUATING CLIP IN SCENE {clip['scene_number']} =====")
            print(f"Description: \"{clip['text']}\"")

            scene_frames = get_frames_for_scene(clip.get('scene_number'))

            is_necessary, reason = evaluate_clip_necessity(
                client, model_to_use, clip, transcript_data, previous_descriptions,
                total_candidates, scene_frames
            )
            print(f"REASON: {reason}")

            if is_necessary:
                clips_kept += 1
                print("STATUS: Kept. Optimizing description...")
                optimized_clip = optimize_description(client, model_to_use, clip)
                print(f"Original ({len(clip['text'])} chars): {clip['text']}")
                print(f"Optimized ({len(optimized_clip['text'])} chars): {optimized_clip['text']}")
                final_clips.append(optimized_clip)
                previous_descriptions.append(optimized_clip)
            else:
                clips_removed += 1
                print("STATUS: Removed as unnecessary or inaccurate.")
        else:
            final_clips.append(clip)
            previous_descriptions.append(clip)

    final_clips.sort(key=lambda x: (x.get('scene_number', 0), x.get('start_time', 0)))

    with open(audio_clips_path, 'w') as f:
        json.dump(final_clips, f, indent=2)

    print(f"\n{'='*50}")
    print(f"FILTERING SUMMARY")
    print(f"{'='*50}")
    print(f"Results saved to: {audio_clips_path}")
    print(f"Final output: {len(final_clips)} clips total")

    if not args.no_analyze_necessity:
        print(f"Non-gap visual descriptions evaluated: {total_candidates}")
        print(f"  - Kept and optimized: {clips_kept}")
        print(f"  - Removed (unnecessary or inaccurate): {clips_removed}")
        if total_candidates > 0:
            print(f"  - Rejection rate: {clips_removed / total_candidates:.0%}")


if __name__ == "__main__":
    main()