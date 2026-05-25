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

# Cluster window (seconds). A clip joins the current cluster if its start_time
# is within this window of the previous clip in the cluster.
CLUSTER_WINDOW_VISUAL = 0.5    # Visual + Visual: only exact same start time
CLUSTER_WINDOW_TOS = 1.0


# -------------------------------------------------------------------------
# Frame extraction (mirrors filter_clips.py so behavior is consistent)
# -------------------------------------------------------------------------

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


# -------------------------------------------------------------------------
# Clustering
# -------------------------------------------------------------------------

def cluster_clips_by_time(clips,
                          window_visual: float = CLUSTER_WINDOW_VISUAL,
                          window_tos: float = CLUSTER_WINDOW_TOS,
                          time_key: str = "global_start_time"):
    if not clips:
        return []

    clusters = [[clips[0]]]
    for clip in clips[1:]:
        anchor_clip = clusters[-1][0]
        anchor_start = float(anchor_clip.get(time_key, 0))
        curr_start = float(clip.get(time_key, 0))

        # Use the ToS window if EITHER side is a ToS; otherwise the visual window.
        anchor_is_tos = anchor_clip.get("type") == "Text on Screen"
        clip_is_tos = clip.get("type") == "Text on Screen"
        if anchor_is_tos or clip_is_tos:
            window = window_tos
        else:
            window = window_visual

        in_window = (curr_start - anchor_start) <= window

        # Never dedup ToS vs ToS: if this clip is a ToS and the current
        # cluster already contains a ToS, force a new cluster.
        cluster_has_tos = any(c.get("type") == "Text on Screen" for c in clusters[-1])
        tos_conflict = clip_is_tos and cluster_has_tos

        if in_window and not tos_conflict:
            clipped_in = True
        else:
            clipped_in = False

        if clipped_in:
            clusters[-1].append(clip)
        else:
            clusters.append([clip])
    return clusters

def _format_cluster_for_prompt(cluster):
    lines = []
    for i, clip in enumerate(cluster, start=1):
        ctype = clip.get("type", "Visual")
        # Show global time so the picker sees the absolute moment in the video.
        cstart = float(clip.get("global_start_time", clip.get("start_time", 0)))
        ctext = (clip.get("text", "") or "").strip()
        scene_no = clip.get("_scene_number", "?")
        lines.append(f'{i}. [{ctype} @ {cstart:.2f}s | scene {scene_no}] "{ctext}"')
    return "\n".join(lines)


def _build_picker_prompt_mixed(cluster, scene_transcript, cumulative_transcript):
    """Cluster contains a Text on Screen candidate plus one or more Visuals."""
    cluster_text = _format_cluster_for_prompt(cluster)
    n = len(cluster)
    return f"""You are an accessibility expert resolving redundancy among candidate audio descriptions for blind viewers.

The candidates below all fall within a short time window of each other in the video. This cluster contains a Text on Screen candidate (a literal transcription of overlay text) and one or more Visual candidates (descriptions of actions/moments).

### CONTEXT
RELEVANT SCENE TRANSCRIPT(S):
{scene_transcript or "(no spoken dialogue around this moment)"}

CUMULATIVE TRANSCRIPT FROM EARLIER IN THE VIDEO:
{cumulative_transcript or "(none)"}

CANDIDATES (numbered 1..{n}, in time order):
{cluster_text}

### STEP 1: CLASSIFY THE ON-SCREEN TEXT
- PROMINENT: title cards, captions, name tags, lower-thirds, on-screen questions or statements, dates/locations stamped on screen, intertitles, slide text in an explainer video. The creator put it on screen to communicate something.
- DECORATIVE: background signage, store names in a street scene, t-shirt logos, book titles on a shelf, brand labels on props, graffiti, license plates, background posters. It exists in the world the camera is filming, but it's not what the shot is about.

If unsure, default to PROMINENT.

### STEP 2: IS THE ON-SCREEN TEXT ALREADY NARRATED?
Only relevant if a transcript is present. Check whether the on-screen text content is already spoken in RELEVANT SCENE TRANSCRIPT(S) or CUMULATIVE TRANSCRIPT. A match counts if the narrator reads the text aloud, or the name/fact/quote already appears in dialogue. Minor wording differences still count.

### STEP 3: PICK THE WINNER

Apply the rules in order:

- If Text on Screen is DECORATIVE → pick the most informative Visual.
- Else if a transcript exists AND it already narrates the on-screen text → pick the most informative Visual (the text is redundant with audio).
- Else (PROMINENT and not already in audio, OR no transcript at all) → pick the Text on Screen candidate. ONLY exception: pick a Visual if it BOTH (a) quotes the on-screen text verbatim, AND (b) adds info the text alone doesn't (who's on screen, action, setting).

You are ONLY resolving redundancy, not judging necessity - a downstream step handles that.

### OUTPUT
Return ONLY this JSON:
{{
  "evidence": "<one sentence on what you see at this moment>",
  "text_role": "<'prominent' or 'decorative'>",
  "text_in_transcript": "<'yes' (quote the matching phrase), 'no', or 'no_transcript' if there is no transcript context>",
  "verbatim_check": "<if you picked a Visual because it quotes prominent text verbatim, state which one and quote the match. Otherwise: 'n/a'>",
  "winner_index": <1 to {n}>,
  "reason": "<one sentence citing which rule applied>"
}}
"""


def _build_picker_prompt_visual_only(cluster, scene_transcript, cumulative_transcript):
    """Cluster contains only Visual candidates."""
    cluster_text = _format_cluster_for_prompt(cluster)
    n = len(cluster)
    return f"""You are an accessibility expert resolving redundancy among candidate audio descriptions for blind viewers.

The candidates below all fall within a short time window of each other in the video. They are all Visual descriptions (actions or moments in the scene) competing for the same beat.

### CONTEXT
RELEVANT SCENE TRANSCRIPT(S):
{scene_transcript or "(no spoken dialogue around this moment)"}

CUMULATIVE TRANSCRIPT FROM EARLIER IN THE VIDEO:
{cumulative_transcript or "(none)"}

CANDIDATES (numbered 1..{n}, in time order):
{cluster_text}

### RULE
Pick the most informative candidate. If two are equally informative, pick the more concise one. Ignore any candidate that is clearly inaccurate to what is on screen.

You are ONLY resolving redundancy, not judging necessity - a downstream step handles that. Always pick the best candidate even if you think they'll all be dropped later.

### OUTPUT
Return ONLY this JSON:
{{
  "evidence": "<one sentence on what you see at this moment>",
  "winner_index": <1 to {n}>,
  "reason": "<one sentence on why this candidate is most informative/concise>"
}}
"""


def _build_picker_prompt(cluster, scene_transcript, cumulative_transcript):
    """Dispatcher: routes to the right prompt based on cluster composition."""
    has_tos = any(c.get("type") == "Text on Screen" for c in cluster)
    if has_tos:
        return _build_picker_prompt_mixed(cluster, scene_transcript, cumulative_transcript)
    return _build_picker_prompt_visual_only(cluster, scene_transcript, cumulative_transcript)


# -------------------------------------------------------------------------
# Picker dispatch
# -------------------------------------------------------------------------

def pick_best_in_cluster(client, model_name, cluster, scene_transcript_text,
                         cumulative_transcript_text,
                         scene_video_bytes=None, scene_frames=None):
    n = len(cluster)
    fallback = {'winner_index': 1, 'reason': 'fallback: picker failed',
                'evidence': '', 'verbatim_check': '',
                'text_in_transcript': '', 'text_role': ''}

    prompt = _build_picker_prompt(cluster, scene_transcript_text, cumulative_transcript_text)

    try:
        result = ""
        if model_name.startswith('gemini'):
            prompt += '\n\nIMPORTANT: Respond with ONLY the raw JSON object. No markdown, no preamble.'

            if not scene_video_bytes:
                print("  [warn] No scene video bytes for Gemini picker. Falling back to text-only.")
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
                    max_output_tokens=4096,
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
            result = _generate_with_qwen(client, prompt, 1024, 0.2)

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
                    {"role": "system", "content": "You are an accessibility expert resolving redundancy among candidate audio descriptions. You verify the scene before picking the best candidate."},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.1,
                max_tokens=400,
                response_format={"type": "json_object"},
            )
            result = response.choices[0].message.content.strip()

        if not result or not isinstance(result, str):
            print(f"  [warn] Empty picker response. Falling back to first clip.")
            return fallback

        matches = re.search(r'\{.*\}', result, re.DOTALL)
        json_str = matches.group(0) if matches else result
        analysis = json.loads(json_str)

        winner = analysis.get('winner_index')
        try:
            winner = int(winner)
        except (TypeError, ValueError):
            print(f"  [warn] Non-integer winner_index ({winner!r}). Falling back to first clip.")
            return fallback

        if winner < 1 or winner > n:
            print(f"  [warn] winner_index {winner} out of range [1, {n}]. Falling back to first clip.")
            return fallback

        return {
            'winner_index': winner,
            'reason': analysis.get('reason', "No reason provided"),
            'evidence': (analysis.get('evidence') or "").strip(),
            'verbatim_check': (analysis.get('verbatim_check') or "").strip(),
            'text_in_transcript': (analysis.get('text_in_transcript') or "").strip(),
            'text_role': (analysis.get('text_role') or "").strip(),
        }

    except json.JSONDecodeError:
        print(f"  [warn] Failed to parse picker JSON: {result[:200] if result else '<empty>'}")
        return fallback
    except Exception as e:
        print(f"  [error] Picker call failed: {e}")
        return fallback


# -------------------------------------------------------------------------
# Scene video evidence cache
# -------------------------------------------------------------------------

class SceneEvidenceCache:
    def __init__(self, model_name, scenes_by_number):
        self.model_name = model_name
        self.scenes_by_number = scenes_by_number  # scene_number -> scene dict
        self._video_bytes = {}    # scene_number -> bytes
        self._frames = {}         # scene_number -> List[str]

    def _load_for_scene(self, scene_number):
        if scene_number in self._video_bytes or scene_number in self._frames:
            return
        scene = self.scenes_by_number.get(scene_number)
        if not scene:
            return
        scene_path = scene.get("scene_path")
        if not scene_path or not os.path.exists(scene_path):
            return
        if self.model_name.startswith("gemini"):
            with open(scene_path, "rb") as vf:
                self._video_bytes[scene_number] = vf.read()
            print(f"  Loaded scene {scene_number} video "
                  f"({len(self._video_bytes[scene_number]) / 1024:.0f} KB) for Gemini")
        else:
            frames = extract_scene_frames_at_fps(scene_path)
            if frames:
                self._frames[scene_number] = frames

    def evidence_for_cluster(self, cluster):
        scene_numbers = []
        seen = set()
        for clip in cluster:
            sn = clip.get("_scene_number")
            if sn is not None and sn not in seen:
                seen.add(sn)
                scene_numbers.append(sn)

        for sn in scene_numbers:
            self._load_for_scene(sn)

        if self.model_name.startswith("gemini"):
            for sn in scene_numbers:
                if sn in self._video_bytes:
                    return self._video_bytes[sn], None
            return None, None

        combined_frames = []
        for sn in scene_numbers:
            combined_frames.extend(self._frames.get(sn, []))
            if len(combined_frames) >= MAX_FRAMES_FOR_IMAGE_BACKEND:
                break
        combined_frames = combined_frames[:MAX_FRAMES_FOR_IMAGE_BACKEND]
        return None, (combined_frames or None)


# -------------------------------------------------------------------------
# Helpers for global timing
# -------------------------------------------------------------------------

def _scene_offset(scene):
    for key in ("start_time", "scene_start", "scene_start_time", "global_start_time", "offset"):
        if key in scene:
            try:
                return float(scene[key])
            except (TypeError, ValueError):
                continue
    return 0.0


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=("Resolve redundancy among AI-generated AD clips by picking the best clip "
                     f"from each cluster, GLOBALLY across all scenes. "
                     f"Visual+Visual window: {CLUSTER_WINDOW_VISUAL}s (only same start_time). "
                     f"ToS+Visual window: {CLUSTER_WINDOW_TOS}s. "
                     "Two Text on Screen clips are never deduped against each other.")
    )
    parser.add_argument("video_folder", help="Path to the video folder")
    parser.add_argument("--model", type=str, choices=[MODEL_GEMINI, MODEL_QWEN, MODEL_GPT], default=MODEL_GPT,
                        help="Model to use for cluster picking: 'gemini', 'qwen', or 'gpt'.")
    parser.add_argument("--window-visual", type=float, default=CLUSTER_WINDOW_VISUAL,
                        help=f"Cluster window for Visual+Visual, in seconds "
                             f"(default: {CLUSTER_WINDOW_VISUAL}; 0 means only exact same start_time).")
    parser.add_argument("--window-tos", type=float, default=CLUSTER_WINDOW_TOS,
                        help=f"Cluster window when the anchor is a Text on Screen, in seconds "
                             f"(default: {CLUSTER_WINDOW_TOS}).")
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
    input_path = os.path.join(scenes_folder, f"scene_info_{args.model}.json")

    if not os.path.exists(input_path):
        fallback = os.path.join(scenes_folder, "scene_info.json")
        if os.path.exists(fallback):
            input_path = fallback
        else:
            print(f"Error: No scene_info file found in {scenes_folder}.")
            return

    output_path = os.path.join(scenes_folder, f"scene_info_{args.model}_deduped.json")

    print(f"\nReading: {input_path}")
    print(f"Writing: {output_path}")
    print(f"Cluster windows: Visual+Visual={args.window_visual}s, "
          f"ToS+Visual={args.window_tos}s GLOBAL (anchor-based, ToS-vs-ToS not deduped)\n")

    with open(input_path, "r", encoding="utf-8") as f:
        scenes = json.load(f)

    # ---------------------------------------------------------------------
    # 1. Build a global, time-ordered list of clusterable clips. Each clip
    #    gets `_scene_number`, `_scene_index`, and `global_start_time` tags
    #    so we can route it back to its scene later.
    # ---------------------------------------------------------------------
    scenes_by_number = {}
    cumulative_transcript_segments_by_scene_idx = []  # parallel to scenes
    all_clusterable = []
    passthrough_by_scene_idx = [[] for _ in scenes]

    running_transcript = []
    for s_idx, scene in enumerate(scenes):
        scene_number = scene.get("scene_number", s_idx)
        scenes_by_number[scene_number] = scene
        offset = _scene_offset(scene)

        # Save the cumulative transcript *before* this scene for later prompt context.
        cumulative_transcript_segments_by_scene_idx.append(list(running_transcript))

        scene_transcript_segments = scene.get("transcript", []) or []
        running_transcript.extend(scene_transcript_segments)

        for clip in scene.get("audio_clips", []):
            ctype = clip.get("type")
            try:
                local_start = float(clip.get("start_time", 0))
            except (TypeError, ValueError):
                local_start = 0.0
            global_start = offset + local_start

            if ctype in ("Visual", "Text on Screen"):
                tagged = dict(clip)
                tagged["_scene_number"] = scene_number
                tagged["_scene_index"] = s_idx
                tagged["global_start_time"] = global_start
                all_clusterable.append(tagged)
            else:
                passthrough_by_scene_idx[s_idx].append(clip)

    all_clusterable.sort(key=lambda c: c["global_start_time"])
    total_input_clips = len(all_clusterable)

    # ---------------------------------------------------------------------
    # 2. Cluster globally.
    # ---------------------------------------------------------------------
    clusters = cluster_clips_by_time(all_clusterable,
                                     window_visual=args.window_visual,
                                     window_tos=args.window_tos,
                                     time_key="global_start_time")
    multi_clusters = [c for c in clusters if len(c) > 1]
    singleton_clusters = [c for c in clusters if len(c) == 1]
    total_singletons = len(singleton_clusters)

    print(f"\n===== GLOBAL: {total_input_clips} clips -> {len(clusters)} clusters "
          f"({len(multi_clusters)} multi, {len(singleton_clusters)} singleton) =====")

    # ---------------------------------------------------------------------
    # 3. Resolve each multi-clip cluster.
    # ---------------------------------------------------------------------
    evidence_cache = SceneEvidenceCache(model_to_use, scenes_by_number)

    survivors = []  # list of (scene_idx, clip)
    picker_log = []
    total_clusters_resolved = 0
    total_clips_dropped_by_picker = 0

    for cluster in clusters:
        if len(cluster) == 1:
            clip = cluster[0]
            survivors.append((clip["_scene_index"], clip))
            continue

        total_clusters_resolved += 1
        scene_nums_in_cluster = sorted({c["_scene_number"] for c in cluster})
        spans_scenes = len(scene_nums_in_cluster) > 1

        print(f"\n  --- Cluster of {len(cluster)} clips "
              f"@ {cluster[0]['global_start_time']:.2f}s "
              f"(scenes: {scene_nums_in_cluster}{' SPANS' if spans_scenes else ''}) ---")
        for i, clip in enumerate(cluster, start=1):
            ctype = clip.get("type", "Visual")
            cstart = float(clip.get("global_start_time", 0))
            ctext = (clip.get("text", "") or "").strip()
            print(f'    {i}. [{ctype} @ {cstart:.2f}s | scene {clip["_scene_number"]}] '
                  f'"{ctext[:80]}"')

        # Build the transcript context for this cluster: union of transcripts
        # from every scene the cluster touches.
        scene_transcript_text_parts = []
        for sn in scene_nums_in_cluster:
            scene = scenes_by_number.get(sn, {})
            segs = scene.get("transcript", []) or []
            text = " ".join(seg.get("text", "") for seg in segs).strip()
            if text:
                scene_transcript_text_parts.append(f"[scene {sn}] {text}")
        scene_transcript_text = "\n".join(scene_transcript_text_parts)

        # Cumulative transcript = everything before the earliest scene in the cluster.
        earliest_scene_idx = min(c["_scene_index"] for c in cluster)
        cumulative_transcript_text = " ".join(
            seg.get("text", "")
            for seg in cumulative_transcript_segments_by_scene_idx[earliest_scene_idx]
        ).strip()

        scene_video_bytes, scene_frames = evidence_cache.evidence_for_cluster(cluster)
        if not scene_video_bytes and not scene_frames:
            print(f"  [warn] No video evidence available; using text-only picker.")

        decision = pick_best_in_cluster(
            client, model_to_use, cluster,
            scene_transcript_text, cumulative_transcript_text,
            scene_video_bytes=scene_video_bytes,
            scene_frames=scene_frames,
        )

        winner_idx = decision['winner_index']
        winner = cluster[winner_idx - 1]
        losers = [c for j, c in enumerate(cluster, start=1) if j != winner_idx]

        print(f"     evidence:      {decision.get('evidence', '')}")
        if decision.get('text_role'):
            print(f"     text_role:     {decision.get('text_role', '')}")
        if decision.get('text_in_transcript'):
            print(f"     in_transcript: {decision.get('text_in_transcript', '')}")
        if decision.get('verbatim_check'):
            print(f"     verbatim:      {decision.get('verbatim_check', '')}")
        print(f"     WINNER:        #{winner_idx} [{winner.get('type')} @ "
              f"{float(winner.get('global_start_time', 0)):.2f}s | scene {winner['_scene_number']}]")
        print(f"     reason:        {decision.get('reason', '')}")

        winner_clip = dict(winner)
        winner_clip['cluster_size'] = len(cluster)
        winner_clip['cluster_spans_scenes'] = spans_scenes
        if decision.get('text_role'):
            winner_clip['cluster_text_role'] = decision['text_role']
        if decision.get('text_in_transcript'):
            winner_clip['cluster_text_in_transcript'] = decision['text_in_transcript']
        if decision.get('verbatim_check'):
            winner_clip['cluster_verbatim_check'] = decision['verbatim_check']
        winner_clip['cluster_competitors'] = [
            {
                'type': c.get('type'),
                'start_time': c.get('start_time'),
                'global_start_time': c.get('global_start_time'),
                'scene_number': c.get('_scene_number'),
                'text': c.get('text', ''),
            }
            for c in losers
        ]
        winner_clip['cluster_pick_reason'] = decision.get('reason', '')

        survivors.append((winner['_scene_index'], winner_clip))
        total_clips_dropped_by_picker += len(losers)

        picker_log.append({
            'cluster_global_start': float(cluster[0]['global_start_time']),
            'cluster_global_end': float(cluster[-1]['global_start_time']),
            'cluster_size': len(cluster),
            'spans_scenes': spans_scenes,
            'scenes': scene_nums_in_cluster,
            'winner_index': winner_idx,
            'winner_scene': winner['_scene_number'],
            'winner_text': winner.get('text', ''),
            'text_role': decision.get('text_role', ''),
            'text_in_transcript': decision.get('text_in_transcript', ''),
            'dropped': [
                {'scene': c['_scene_number'], 'text': c.get('text', '')}
                for c in losers
            ],
            'reason': decision.get('reason', ''),
        })

    total_output_clips = len(survivors)

    # ---------------------------------------------------------------------
    # 4. Re-attach surviving clips to their original scenes, strip internal
    #    bookkeeping fields, and write output.
    # ---------------------------------------------------------------------
    survivors_by_scene_idx = [[] for _ in scenes]
    for s_idx, clip in survivors:
        # Strip internal-only fields. Keep cluster_* fields for traceability.
        clean = {k: v for k, v in clip.items()
                 if k not in ("_scene_number", "_scene_index", "global_start_time")}
        survivors_by_scene_idx[s_idx].append(clean)

    for s_idx, scene in enumerate(scenes):
        merged = survivors_by_scene_idx[s_idx] + passthrough_by_scene_idx[s_idx]
        scene["audio_clips"] = sorted(merged, key=lambda c: c.get("start_time", 0))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scenes, f, indent=2)

    print("\n" + "=" * 60)
    print("CLIP DEDUP SUMMARY (GLOBAL)")
    print("=" * 60)
    print(f"Output: {output_path}")
    print(f"Cluster windows: Visual+Visual={args.window_visual}s, "
          f"ToS+Visual={args.window_tos}s (anchor-based, ToS-vs-ToS not deduped)")
    print(f"Total clips in:           {total_input_clips}")
    print(f"  - Singleton clusters:   {total_singletons} (passed through)")
    print(f"  - Multi-clip clusters:  {total_clusters_resolved} (resolved by picker)")
    cross_scene = sum(1 for e in picker_log if e['spans_scenes'])
    print(f"    of which cross-scene: {cross_scene}")
    print(f"Total clips dropped:      {total_clips_dropped_by_picker}")
    print(f"Total clips out:          {total_output_clips}")
    if total_input_clips > 0:
        print(f"Reduction:                {total_clips_dropped_by_picker / total_input_clips:.0%}")

    if picker_log:
        print("\n" + "=" * 60)
        print(f"PICKER DECISIONS ({len(picker_log)} clusters resolved)")
        print("=" * 60)
        for entry in picker_log:
            span_marker = " [CROSS-SCENE]" if entry['spans_scenes'] else ""
            print(f"\n@ {entry['cluster_global_start']:.2f}s "
                  f"(cluster of {entry['cluster_size']}, scenes {entry['scenes']}){span_marker}")
            print(f"  WINNER (scene {entry['winner_scene']}) : {entry['winner_text']}")
            for dropped in entry['dropped']:
                print(f"  dropped (scene {dropped['scene']}): {dropped['text']}")
            if entry.get('text_role') or entry.get('text_in_transcript'):
                print(f"  text_role={entry.get('text_role', 'n/a')}, "
                      f"in_transcript={entry.get('text_in_transcript', 'n/a')}")
            print(f"  reason : {entry['reason']}")


if __name__ == "__main__":
    main()