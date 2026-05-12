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
                             cumulative_transcript, kept_descriptions_text):
    return f"""You are an accessibility expert filtering descriptions for an audio description track for blind viewers.

You are watching the scene this description was generated for. Use it to verify accuracy AND decide whether to keep, correct, or drop the description.

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

### DECISION PROCESS — FOLLOW THESE STEPS IN ORDER

STEP 1 — EVIDENCE
Write one sentence describing what you actually see in the video at this moment, in your own words. Required for every clip. Your evidence is what YOU observe — it is NOT a previously-narrated description. Only items in "DESCRIPTIONS ALREADY KEPT" count as previously narrated.

STEP 2 — ACCURACY CHECK
Compare your evidence to the candidate description. Are the objects, characters, actions, AND text in the description ACTUALLY present and correctly transcribed?
- Visual clips: check that actions and objects match the video.
- Text on Screen clips: check that the text reads as the candidate claims.

If accurate: set accurate=true, go to STEP 4.
If not: set accurate=false, go to STEP 3.

STEP 3 — CORRECTION ATTEMPT (only for inaccurate Visual clips)
Can you confidently describe what is actually happening based on your evidence?
- If YES: write the corrected description and continue to STEP 4 with the corrected text.
- If NO: verdict is "drop". Stop here.

STEP 4 — NECESSITY CHECK (only on accurate text — original or corrected)

GUIDING PRINCIPLE: LESS IS MORE.

Audio descriptions interrupt the natural pacing of the video. Every clip kept is an interruption. A sparse, well-chosen set of descriptions is far better than a dense one. The default verdict is DROP. The bar for keeping a clip is high.

KEEP only if you can answer YES to ALL THREE of these questions, IN ORDER:

1. Without this description, would a blind viewer be CONFUSED about what is happening — not just missing detail, but unable to follow the scene?

2. Is this information UNAVAILABLE and IMPOSSIBLE TO INFER from any other source — dialogue, sound effects, audible reactions, character voices, music, surrounding scene context, or descriptions already kept (only items in DESCRIPTIONS ALREADY KEPT count — your own evidence does NOT count)?

   This is the redundancy check. Check it FIRST and check it carefully. Scan DESCRIPTIONS ALREADY KEPT for:
   - An identical or near-identical string (especially for Text on Screen clips — the same text card often recurs across scene boundaries).
   - A description that conveys the same instruction, action, or fact in different words.
   - A description whose meaning subsumes this one (e.g., a kept clip describes a sequence that includes this moment).

   If ANY of the above apply, this clip fails question 2. The verdict is drop. Quote the matching kept item in your reason.

3. Does this clip describe something that MATTERS — the core action of the scene, or a fact (name, date, place) that affects understanding?

If you cannot answer YES to all three, the verdict is "drop".

APPLYING QUESTION 3 TO "Text on Screen" CLIPS:

Use prominence to judge whether on-screen text matters — but ONLY after question 2 has been satisfied. Question 2 always comes first; a duplicate prominent title card is still a duplicate.

- MATTERS (passes Q3): Text placed prominently front-and-center, taking up significant space, or presented as a title card, chapter heading, or large location/time/instruction stamp. The creator explicitly intended the audience to read it.
- DOES NOT MATTER (fails Q3): Logos, watermarks, channel branding, copyright notices; names on screen (lower-thirds, name tags, names printed on folders/desks/doors) when the character's identity is already obvious from context; decorative or environmental text (signs, posters, object labels) that doesn't drive the scene; subtitles or captions of the dialogue itself.

A prominent text clip that has already been narrated in DESCRIPTIONS ALREADY KEPT must still be dropped. Prominence is not a redundancy override.

APPLYING QUESTION 3 TO "Visual" CLIPS — common DROP patterns:
- The windup or setup of an action when a separate clip describes the action itself or its payoff. EXCEPTION: in instructional or procedural content (recipes, tutorials, how-tos), each sequential step is itself the content — do not drop a step as "inferable from a previous step."
- A character preparing, picking up, or moving toward something — when the consequential action is what matters.
- Audible behaviors (laughing, crying, screaming, sighing) when the clip describes ONLY the behavior itself. If the clip also identifies WHO is doing it, HOW MANY are doing it, where they are, or what they are reacting to — and that information is not otherwise established — that visual context is the contribution and the clip should be kept.
- Background characters or framing details that aren't the focus of the scene's action.
- Incidental movement (walking, standing, sitting) that doesn't change the situation.
- Mood, expression, or intent that the dialogue or voice conveys.

### CORRECTION GUIDELINES (only when verdict is "keep_corrected")
- Match the length and style of the original (one short sentence).
- Describe what is actually visible — be specific about objects and actions.
- Use any character names from the prior descriptions/transcript.
- Do NOT add information beyond what the scene shows.
- Corrections only apply to "Visual" clips, never "Text on Screen" clips.

### OUTPUT FORMAT
Return ONLY a JSON object with this exact structure:
{{
  "evidence": "<one short sentence describing what you actually see in the video at this moment, in your own words. Mention if text is front-and-center.>",
  "accurate": <true if the description matches what you see (always true for Text on Screen), false if not>,
  "verdict": "keep_original" | "keep_corrected" | "drop",
  "corrected_text": "<the corrected description, only when verdict is 'keep_corrected'; otherwise empty string>",
  "reason": "<one or two sentences. If verdict is 'drop' for redundancy, QUOTE the matching item from DESCRIPTIONS ALREADY KEPT. If verdict is 'keep_original' or 'keep_corrected', briefly confirm why this is NOT redundant with the kept list (e.g., 'no similar item in kept list' or 'kept list covers X but not this Y').>"
}}

The "evidence" field is REQUIRED for every clip. Do not skip it.
"""

def evaluate_clip(client, model_name, clip_text, clip_type, clip_start,
                  scene_transcript_text, cumulative_transcript_text,
                  kept_descriptions_text,
                  scene_video_bytes=None, scene_frames=None):
    """
    Unified evaluator for Visual and Text on Screen clips.
    Returns dict: {verdict, corrected_text, reason, accurate, evidence}.
    """
    prompt = _build_evaluation_prompt(
        clip_text, clip_type, clip_start, scene_transcript_text,
        cumulative_transcript_text, kept_descriptions_text,
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
        print(f"\n===== SCENE {scene_number}: {visual_count} Visual + {text_count} Text on Screen =====")

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