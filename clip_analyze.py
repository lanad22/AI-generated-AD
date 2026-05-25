import json
import os
import re
import base64
import argparse
import cv2
from typing import List
from dotenv import load_dotenv
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from google import genai
from google.genai import types
import openai

load_dotenv()

MODEL_QWEN = "qwen"
MODEL_GEMINI = "gemini"
MODEL_GPT = "gpt"

# Match the captioner's video sampling rate.
SCENE_SAMPLING_FPS = 3.0
MAX_FRAMES_FOR_IMAGE_BACKEND = 60
VERIFICATION_IMAGE_DETAIL = "low"


# ============================================================
# GENRE-AWARE FILTERING STRATEGIES
# ============================================================
# Adjust the necessity bar in keep/drop decisions based on video genre.
# Injected into the evaluation prompt to modify Question 3 (MATTERS).

HOWTO_CATEGORIES = {
    "howto", "how-to", "howto & style", "recipe", "tutorial", "diy", "cooking",
}
EDUCATION_CATEGORIES = {
    "education", "science & technology", "informational", "informative",
    "documentary", "news & politics", "news",
}
PETS_ANIMALS_CATEGORIES = {
    "pets & animals", "pets", "animals",
}
ENTERTAINMENT_CATEGORIES = {
    "film & animation", "entertainment", "comedy", "shows", "movies",
    "trailers", "drama", "gaming",
}

FILTER_STRATEGIES = {
    "howto": """
        ### GENRE ADJUSTMENTS — HOW-TO / TUTORIAL
        Each procedural step IS the content. Do NOT drop steps as "windup" or "inferable from previous."
        Keep: tool choice, material state, technique, cues for doneness.
        Drop: actor's body with no action attached.
    """,
    "education": """
        ### GENRE ADJUSTMENTS — EDUCATIONAL / DOCUMENTARY
        Narration carries the information. Drop bias is HIGH; redundancy with transcript wins.
        Drop: clips whose content the narrator is already explaining.
        Keep: what's pointed to / highlighted / animated when narrator doesn't name the change;
              prominent text for names, dates, places not spoken; lower-third name captions on first appearance.
    """,
    "pets_animals": """
        ### GENRE ADJUSTMENTS — PETS & ANIMALS
        Body language and reactions ARE the story; they are NOT incidental.
        Keep: the animal's body language, expression, reactions, and interactions.
        Drop: generic locomotion with no payoff; descriptions of audible sounds with no added context.
    """,
    "entertainment": """
        ### GENRE ADJUSTMENTS — ENTERTAINMENT / NARRATIVE
        Drop bias is strongest here.
        Drop: windup/setup, incidental movement, mood already conveyed by voice, background framing.
        Keep: silent story beats (a meaningful look, hidden gesture, visual reveal), sight gags,
              clips that identify WHO is reacting when audio doesn't make it clear.
    """,
}


def get_genre_label(video_category: str) -> str:
    """Return one of: 'howto', 'education', 'pets_animals', 'entertainment', or 'none'."""
    if not video_category:
        return "none"
    cat_lower = video_category.lower().strip()
    if any(k in cat_lower for k in HOWTO_CATEGORIES):
        return "howto"
    if any(k in cat_lower for k in EDUCATION_CATEGORIES):
        return "education"
    if any(k in cat_lower for k in PETS_ANIMALS_CATEGORIES):
        return "pets_animals"
    if any(k in cat_lower for k in ENTERTAINMENT_CATEGORIES):
        return "entertainment"
    return "none"


def get_filter_strategy(video_category: str) -> str:
    """Return the genre-specific necessity-adjustment block, or '' if no match."""
    return FILTER_STRATEGIES.get(get_genre_label(video_category), "")


def extract_scene_frames_at_fps(video_path: str, target_fps: float = SCENE_SAMPLING_FPS,
                                max_frames: int = MAX_FRAMES_FOR_IMAGE_BACKEND) -> List[str]:
    if not video_path or not os.path.exists(video_path):
        return []

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Could not open scene video: {video_path}")
        return []

    fps = video.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps else 0

    if total_frames <= 0:
        video.release()
        return []

    frame_interval = max(1, int(round(fps / target_fps)))
    base64_frames = []
    frame_idx = 0
    while True:
        success, frame = video.read()
        if not success:
            break
        if frame_idx % frame_interval == 0:
            _, buffer = cv2.imencode(".jpg", frame)
            base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
            if len(base64_frames) >= max_frames:
                break
        frame_idx += 1

    video.release()
    print(f"  Extracted {len(base64_frames)} frames at {target_fps} FPS ({duration:.1f}s scene)")
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
        do_sample=True,
    )

    input_token_len = inputs.input_ids.shape[1]
    generated_ids = output_ids[:, input_token_len:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()


def _build_evaluation_prompt(clip_text, clip_type, clip_start, scene_transcript,
                             cumulative_transcript, kept_descriptions_text,
                             genre_strategy=""):
    scene_has_transcript = bool(scene_transcript and scene_transcript.strip())
    if scene_has_transcript:
        strictness_banner = (
            "### STRICTNESS MODE: STRICT\n"
            "This scene HAS spoken dialogue. Audio descriptions will interrupt the narration "
            "every time they fire. The bar to keep is HIGH — keep only what is essential to "
            "understanding. When in doubt, DROP."
        )
    else:
        strictness_banner = (
            "### STRICTNESS MODE: PERMISSIVE\n"
            "This scene has NO spoken dialogue. Audio description is the ONLY source of "
            "information for blind viewers during this scene. The bar to keep is LOWER — "
            "keep descriptions that convey what is happening, even if they would be borderline "
            "in a scene with dialogue. Without descriptions, the viewer experiences silence."
        )

    return f"""You are an accessibility expert filtering descriptions for an audio description track for blind viewers.

You are watching the scene this description was generated for. Use it to verify accuracy AND decide whether to keep, correct, or drop the description.

{strictness_banner}

### CONTEXT
CURRENT SCENE TRANSCRIPT:
{scene_transcript or "(no spoken dialogue in this scene)"}

CUMULATIVE TRANSCRIPT FROM EARLIER SCENES:
{cumulative_transcript or "(none)"}

DESCRIPTIONS ALREADY KEPT:
{kept_descriptions_text or "(none yet)"}

CANDIDATE TO EVALUATE:
- Type: {clip_type}
- Text: "{clip_text}"
- Approximate timestamp: {clip_start:.1f} seconds into the scene

A "Visual" clip describes something happening in the scene (an action, a moment).
A "Text on Screen" clip is a literal transcription of overlay text shown on screen (a name caption, title card, date stamp, etc.).

### DECISION PROCESS

STEP 1 — EVIDENCE (required for every clip)
One sentence describing what YOU see in the video at this moment, in your own words. This is not the same as the candidate description. Only items in DESCRIPTIONS ALREADY KEPT count as "previously narrated."

STEP 2 — ACCURACY
Does the candidate match your evidence (objects, characters, actions for Visual; text content for Text on Screen)?
- Yes → accurate=true, go to STEP 4.
- No → accurate=false, go to STEP 3.

STEP 3 — CORRECTION (Visual clips only)
If you can confidently describe what's actually happening, write a corrected version and go to STEP 4 with that text. Otherwise verdict=drop.

STEP 4 — NECESSITY

GUIDING PRINCIPLE: LESS IS MORE. Every clip interrupts the video; the default is DROP. Apply the strictness mode from the banner above: STRICT scenes (with dialogue) need a high bar; PERMISSIVE scenes (no dialogue) need a lower bar because description is the only signal.

KEEP only if YES to all three:

1. Without this, would a blind viewer be unable to comprehend or engage with the video content? In STRICT mode, "missing some detail" is NOT enough — the viewer must be genuinely unable to follow the scene. In PERMISSIVE mode, any meaningful contribution to understanding what is happening qualifies.

2. Is this UNAVAILABLE from any other source — dialogue, sound, voice, music, scene context, or DESCRIPTIONS ALREADY KEPT? Check the kept list for: near-identical strings, same idea in different words, descriptions that subsume this one. If any match, drop and quote the match in your reason. (Your own evidence does NOT count as "already narrated.")

3. Does this MATTER — core action, or a fact like name/date/place that affects understanding?

For "Text on Screen": MATTERS = prominent title cards, headings, location/time stamps, key info not spoken. DOES NOT MATTER = logos, watermarks, lower-third names when identity is already clear, decorative or environmental text, subtitles. (Prominence does NOT override redundancy — a duplicate title card is still a duplicate.)

For "Visual" — common DROP patterns:
- Windup/setup when a separate clip covers the payoff.
- Audible behaviors (laughing, crying) described WITHOUT added context (who, how many, what they're reacting to). With added context, keep.
- Background/framing details, incidental movement, mood the voice already conveys.

{genre_strategy}

### CORRECTION RULES (only when verdict="keep_corrected")
One short sentence, same style as the original. Describe only what's visible. Use known character names. Visual clips only — never Text on Screen.

### OUTPUT FORMAT
Return ONLY this JSON object:
{{
  "evidence": "<one sentence: what you see at this moment, in your own words. Note if text is front-and-center.>",
  "accurate": <true/false; always true for Text on Screen if the text matches>,
  "verdict": "keep_original" | "keep_corrected" | "drop",
  "corrected_text": "<corrected description if keep_corrected, else empty string>",
  "reason": "<1-2 sentences. If dropping for redundancy, QUOTE the matching item from DESCRIPTIONS ALREADY KEPT. If keeping, briefly say why it isn't redundant.>"
}}

"evidence" is REQUIRED.
"""

def evaluate_clip(client, model_name, clip_text, clip_type, clip_start,
                  scene_transcript_text, cumulative_transcript_text,
                  kept_descriptions_text,
                  scene_video_bytes=None, scene_frames=None,
                  genre_strategy=""):
    """
    Unified evaluator for Visual and Text on Screen clips.
    Returns dict: {verdict, corrected_text, reason, accurate, evidence}.
    """
    prompt = _build_evaluation_prompt(
        clip_text, clip_type, clip_start, scene_transcript_text,
        cumulative_transcript_text, kept_descriptions_text,
        genre_strategy=genre_strategy,
    )

    try:
        result = ""
        if model_name.startswith('gemini'):
            prompt += '\n\nIMPORTANT: Respond with ONLY the raw JSON object. No markdown, no preamble.'

            if not scene_video_bytes:
                print("  [warn] No scene video bytes for Gemini evaluation. Falling back to text-only.")
                contents = [prompt]
            else:
                video_part = types.Part(
                    inline_data=types.Blob(mime_type="video/mp4", data=scene_video_bytes),
                    video_metadata=types.VideoMetadata(fps=SCENE_SAMPLING_FPS),
                )
                contents = [prompt, video_part]

            response = client["client"].models.generate_content(
                model=client["model_name"],
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=8192,
                    thinking_config=types.ThinkingConfig(thinking_budget=2048),
                    response_mime_type="application/json",
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
            result = response.text

        elif model_name == 'local_qwen':
            result = _generate_with_qwen(client, prompt, 2048, 0.2)

        else:  # GPT
            user_content = [{"type": "text", "text": prompt}]
            for f_b64 in (scene_frames or []):
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{f_b64}",
                        "detail": VERIFICATION_IMAGE_DETAIL,
                    },
                })
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an accessibility expert filtering descriptions. You verify accuracy by examining the provided scene before judging necessity."},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.1,
                max_tokens=600,
                response_format={"type": "json_object"},
            )
            result = response.choices[0].message.content.strip()

        try:
            if not result or not isinstance(result, str):
                try:
                    candidate = response.candidates[0]
                    finish_reason = getattr(candidate, "finish_reason", "unknown")
                    print(f"  [warn] Empty response. finish_reason={finish_reason}")
                except Exception:
                    pass
                return {'verdict': 'drop', 'corrected_text': '', 'reason': 'Empty response from model',
                        'accurate': None, 'evidence': ''}

            matches = re.search(r'\{.*\}', result, re.DOTALL)
            json_str = matches.group(0) if matches else result
            analysis = json.loads(json_str)

            verdict = analysis.get('verdict', 'drop')
            if verdict not in ('keep_original', 'keep_corrected', 'drop'):
                verdict = 'drop'

            accurate = analysis.get('accurate', None)
            evidence = (analysis.get('evidence') or "").strip()
            corrected_text = (analysis.get('corrected_text') or "").strip()
            reason = analysis.get('reason', "No reason provided")

            # Sanity checks: enforce the decision tree.

            # keep_original requires accurate=true.
            if verdict == 'keep_original' and accurate is False:
                verdict = 'drop'
                reason = f"[downgraded] keep_original but accurate=false. Original reason: {reason}"

            # keep_corrected requires non-empty corrected_text.
            if verdict == 'keep_corrected' and not corrected_text:
                verdict = 'drop'
                reason = f"[downgraded] keep_corrected but no corrected_text. Original reason: {reason}"

            # keep_corrected with accurate=true is contradictory.
            if verdict == 'keep_corrected' and accurate is True:
                verdict = 'keep_original'
                corrected_text = ''
                reason = f"[downgraded] keep_corrected but accurate=true. Treating as keep_original. Original reason: {reason}"

            # Text on Screen clips can never be 'keep_corrected'.
            if clip_type == 'Text on Screen' and verdict == 'keep_corrected':
                verdict = 'keep_original'
                corrected_text = ''
                reason = f"[downgraded] Text on Screen cannot be corrected. Original reason: {reason}"

            return {'verdict': verdict, 'corrected_text': corrected_text, 'reason': reason,
                    'accurate': accurate, 'evidence': evidence}

        except json.JSONDecodeError:
            print(f"  [warn] Failed to parse JSON response: {result[:200]}")
            return {'verdict': 'drop', 'corrected_text': '', 'reason': 'Failed to parse model response',
                    'accurate': None, 'evidence': ''}

    except Exception as e:
        print(f"  [error] Evaluation call failed: {e}")
        return {'verdict': 'drop', 'corrected_text': '', 'reason': f"Error: {e}",
                'accurate': None, 'evidence': ''}


def main():
    parser = argparse.ArgumentParser(description="Filter AI-generated audio description clips for accuracy and necessity.")
    parser.add_argument("video_folder", help="Path to the video folder")
    parser.add_argument("--model", type=str, choices=[MODEL_GEMINI, MODEL_QWEN, MODEL_GPT], default=MODEL_GPT,
                        help="Model to use for clip evaluation: 'gemini', 'qwen', or 'gpt'.")
    args = parser.parse_args()

    client = None
    model_to_use = ""

    if args.model == MODEL_GEMINI:
        model_to_use = "gemini-3-flash-preview"
        print(f"\nSetting up Google Gemini client for model: {model_to_use}...")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("Error: GEMINI_API_KEY environment variable not set.")
            return
        gemini_client = genai.Client(api_key=api_key)
        client = {"client": gemini_client, "model_name": model_to_use}

    elif args.model == MODEL_QWEN:
        model_to_use = "local_qwen"
        print("\nSetting up LOCAL Qwen model...")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-72B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            quantization_config=quantization_config,
            cache_dir="../.cache",
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")
        client = {"model": model, "processor": processor}

    elif args.model == MODEL_GPT:
        model_to_use = "gpt-4o"
        print(f"\nSetting up OpenAI GPT client for model: {model_to_use}...")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable not set.")
            return
        client = openai.OpenAI(api_key=api_key)

    if not client:
        print("Client setup failed. Exiting.")
        return

    video_id = os.path.basename(os.path.normpath(args.video_folder))
    scenes_folder = os.path.join(args.video_folder, f"{video_id}_scenes")
    input_path = os.path.join(scenes_folder, f"scene_info_{args.model}_deduped.json")

    if not os.path.exists(input_path):
        fallback = os.path.join(scenes_folder, f"scene_info_{args.model}.json")
        if os.path.exists(fallback):
            input_path = fallback
        else:
            print(f"Error: No scene_info file found in {scenes_folder}.")
            return

    output_path = os.path.join(scenes_folder, f"scene_info_{args.model}_filtered.json")

    # Load video metadata to determine genre.
    metadata_path = os.path.join(args.video_folder, f"{video_id}.json")
    video_category = "Other"
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                video_metadata = json.load(f)
            video_category = video_metadata.get("category", "Other")
        except Exception as e:
            print(f"  [warn] Could not read metadata {metadata_path}: {e}. Defaulting category to 'Other'.")
    else:
        print(f"  [warn] Metadata file {metadata_path} not found. Defaulting category to 'Other'.")

    genre_strategy = get_filter_strategy(video_category)
    genre_label = get_genre_label(video_category)
    if genre_strategy:
        print(f"\n[genre] Applying '{genre_label}' filter strategy for category: {video_category}")
    else:
        print(f"\n[genre] No genre-specific filter strategy for category: {video_category}")

    print(f"\nReading: {input_path}")
    print(f"Writing: {output_path}\n")

    with open(input_path, "r", encoding="utf-8") as f:
        scenes = json.load(f)

    cumulative_transcript_segments = []
    kept_descriptions = []
    corrections_log = []

    total_evaluated = 0
    total_kept_original = 0
    total_kept_corrected = 0
    total_dropped = 0

    for scene in scenes:
        scene_number = scene.get("scene_number", "?")
        scene_path = scene.get("scene_path")
        audio_clips = scene.get("audio_clips", [])

        scene_transcript_segments = scene.get("transcript", []) or []
        scene_transcript_text = " ".join(seg.get("text", "") for seg in scene_transcript_segments).strip()
        cumulative_transcript_text = " ".join(
            seg.get("text", "") for seg in cumulative_transcript_segments
        ).strip()

        # Evaluate everything in original order so cumulative kept_descriptions is built correctly.
        ALLOWED_CLIP_KEYS = {"type", "text", "start_time"}
        clips_to_evaluate = [
            {k: v for k, v in c.items() if k in ALLOWED_CLIP_KEYS}
            for c in audio_clips if c.get("type") in ("Visual", "Text on Screen")
        ]
        clips_to_evaluate.sort(key=lambda c: c.get("start_time", 0))

        if not clips_to_evaluate:
            cumulative_transcript_segments.extend(scene_transcript_segments)
            continue

        visual_count = sum(1 for c in clips_to_evaluate if c.get("type") == "Visual")
        text_count = sum(1 for c in clips_to_evaluate if c.get("type") == "Text on Screen")
        mode = "STRICT" if scene_transcript_text else "PERMISSIVE"
        print(f"\n===== SCENE {scene_number} [{mode}]: {visual_count} Visual + {text_count} Text on Screen =====")

        # Load video evidence once per scene (used for all clips, regardless of type).
        scene_video_bytes = None
        scene_frames = None
        video_loaded = False

        if scene_path and os.path.exists(scene_path):
            if args.model == MODEL_GEMINI:
                with open(scene_path, "rb") as vf:
                    scene_video_bytes = vf.read()
                print(f"  Loaded scene video ({len(scene_video_bytes) / 1024:.0f} KB) for Gemini")
                video_loaded = True
            else:
                scene_frames = extract_scene_frames_at_fps(scene_path)
                video_loaded = bool(scene_frames)

        kept_in_scene = []

        for clip in clips_to_evaluate:
            total_evaluated += 1
            original_text = clip.get("text", "").strip()
            if not original_text:
                continue

            clip_type = clip.get("type", "Visual")
            clip_start = float(clip.get("start_time", 0))
            print(f"\n  [{clip_type} @ {clip_start:.1f}s] \"{original_text[:80]}\"")

            if not video_loaded:
                print(f"     [warn] No video for scene {scene_number}; keeping unverified.")
                kept_in_scene.append(clip)
                kept_descriptions.append(original_text)
                total_kept_original += 1
                continue

            kept_descriptions_text = "\n".join(f"- {d}" for d in kept_descriptions[-20:])

            decision = evaluate_clip(
                client, model_to_use, original_text, clip_type, clip_start,
                scene_transcript_text, cumulative_transcript_text,
                kept_descriptions_text,
                scene_video_bytes=scene_video_bytes,
                scene_frames=scene_frames,
                genre_strategy=genre_strategy,
            )

            verdict = decision['verdict']
            corrected_text = decision['corrected_text']
            reason = decision['reason']
            accurate = decision.get('accurate')
            evidence = decision.get('evidence', '')

            print(f"     evidence: {evidence}")
            print(f"     accurate: {accurate}")
            print(f"     verdict:  {verdict}")
            print(f"     reason:   {reason}")

            if verdict == 'keep_original':
                kept_in_scene.append(clip)
                kept_descriptions.append(original_text)
                total_kept_original += 1
                print(f"     KEPT (original)")

            elif verdict == 'keep_corrected':
                print(f"     >>> CORRECTION APPLIED <<<")
                print(f"     ORIGINAL : {original_text}")
                print(f"     CORRECTED: {corrected_text}")

                corrected_clip = dict(clip)
                corrected_clip['text'] = corrected_text
                corrected_clip['original_text'] = original_text
                corrected_clip['corrected'] = True
                corrected_clip['correction_reason'] = reason

                kept_in_scene.append(corrected_clip)
                kept_descriptions.append(corrected_text)
                total_kept_corrected += 1

                corrections_log.append({
                    'scene_number': scene_number,
                    'start_time': clip.get('start_time'),
                    'original': original_text,
                    'corrected': corrected_text,
                    'reason': reason,
                })

            else:
                total_dropped += 1
                print(f"     DROPPED")

        scene["audio_clips"] = sorted(
            kept_in_scene,
            key=lambda c: c.get("start_time", 0),
        )

        cumulative_transcript_segments.extend(scene_transcript_segments)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scenes, f, indent=2)

    print("\n" + "=" * 60)
    print("FILTERING SUMMARY")
    print("=" * 60)
    print(f"Output: {output_path}")
    print(f"Total clips evaluated: {total_evaluated}")
    print(f"  - Kept (original):  {total_kept_original}")
    print(f"  - Kept (corrected): {total_kept_corrected}")
    print(f"  - Dropped:          {total_dropped}")
    keep_total = total_kept_original + total_kept_corrected
    if total_evaluated > 0:
        line = f"  - Overall keep rate: {keep_total / total_evaluated:.0%}"
        if keep_total > 0:
            line += f" (of which {total_kept_corrected / keep_total:.0%} corrected)"
        print(line)

    if corrections_log:
        print("\n" + "=" * 60)
        print(f"CORRECTIONS APPLIED ({len(corrections_log)})")
        print("=" * 60)
        for c in corrections_log:
            print(f"\nScene {c['scene_number']} @ {c['start_time']}s")
            print(f"  ORIGINAL : {c['original']}")
            print(f"  CORRECTED: {c['corrected']}")
            print(f"  REASON   : {c['reason']}")

    print(f"\nNext step: run description_optimize_inline.py with --optimizer_model {args.model}")


if __name__ == "__main__":
    main()