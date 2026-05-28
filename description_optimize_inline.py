import argparse
import json
import os
import re
import time
from typing import Callable, Dict, List, Optional

import openai
import torch
from dotenv import load_dotenv
from google import genai
from google.genai import types
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

load_dotenv()

MODEL_QWEN, MODEL_GEMINI, MODEL_GPT4 = "qwen", "gemini", "gpt"

# Beats whose original start_times fall within this window narrate at the
# same wall time and are merged into one description.
MERGE_WINDOW = 1.0

# Strip leading "Text:", "Caption:", "On screen:", etc. that LLMs sometimes
# prepend despite being told not to.
_LABEL_PREFIX_RE = re.compile(
    r'^\s*(?:on[\s-]?screen\s*text|on[\s-]?screen\s*reads|on[\s-]?screen'
    r'|title\s*card|screen\s*reads|screen|caption|subtitle|text)'
    r'\s*[:\-–—]\s*',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def strip_label_prefix(text: str) -> str:
    if not text:
        return text
    cleaned = _LABEL_PREFIX_RE.sub('', text, count=1)
    if cleaned != text and cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned.strip()


def get_tts_duration(text: str, speaking_rate: float = 1.25) -> float:
    """Approximate TTS speaking duration. ~150 wpm * speaking_rate."""
    if not text or text.isspace():
        return 0.0
    words = len(text.split())
    return max(0.5, (words / (150 * speaking_rate)) * 60)


def make_clip(scene_number, text: str, clip_type: str, fits_in_gap: bool,
              duration: Optional[float] = None, **extra) -> Dict:
    """Build a placed-clip dict. start_time/end_time are filled in by the placer."""
    if duration is None:
        duration = get_tts_duration(text)
    return {
        'scene_number': scene_number,
        'text': text,
        'type': clip_type,
        'duration': duration,
        'fits_in_gap': fits_in_gap,
        **extra,
    }


# ---------------------------------------------------------------------------
# Scene parsing
# ---------------------------------------------------------------------------

def get_scene_clips(scene: Dict) -> List[Dict]:
    """Sorted list of (Visual + Text on Screen) clips with TTS durations."""
    out = []
    for c in scene.get('audio_clips', []):
        text = c.get('text')
        if not text or c.get('type') not in ('Visual', 'Text on Screen'):
            continue
        st = c.get('start_time', 0)
        dur = get_tts_duration(text)
        out.append({
            'start_time': st, 'text': text, 'type': c['type'],
            'scene_number': scene.get('scene_number', 'N/A'),
            'duration': dur, 'end_time': st + dur,
        })
    out.sort(key=lambda x: x['start_time'])
    return out


def find_dialogue_gaps(scene: Dict, min_gap_duration: float) -> List[Dict]:
    """Return dialogue-free windows >= min_gap_duration. If no transcript, the
    whole scene is one gap."""
    scene_duration = scene.get('duration') or (
        scene['end_time'] - scene['start_time']
        if 'start_time' in scene and 'end_time' in scene else 0
    )
    if not scene_duration:
        print(f"Warning: Scene {scene.get('scene_number')} missing duration.")
        return []

    segs = sorted(
        [{'start': s.get('start', 0), 'end': s.get('end', 0)}
         for s in (scene.get('transcript') or [])],
        key=lambda x: x['start'],
    )
    # Merge overlapping dialogue.
    merged = []
    for s in segs:
        if merged and s['start'] <= merged[-1]['end']:
            merged[-1]['end'] = max(merged[-1]['end'], s['end'])
        else:
            merged.append(dict(s))

    if not merged:
        return ([{'start_time': 0, 'end_time': scene_duration, 'duration': scene_duration}]
                if scene_duration >= min_gap_duration else [])

    gaps, cursor = [], 0.0
    for s in merged:
        if s['start'] - cursor >= min_gap_duration:
            gaps.append({'start_time': cursor, 'end_time': s['start'],
                         'duration': s['start'] - cursor})
        cursor = max(cursor, s['end'])
    if scene_duration - cursor >= min_gap_duration:
        gaps.append({'start_time': cursor, 'end_time': scene_duration,
                     'duration': scene_duration - cursor})
    return gaps


def cluster_into_beats(clips: List[Dict], window: float = MERGE_WINDOW) -> List[List[Dict]]:
    """Group clips whose start_times are within `window` of the beat's first clip."""
    if not clips:
        return []
    beats = [[clips[0]]]
    anchor = clips[0]['start_time']
    for c in clips[1:]:
        if c['start_time'] - anchor <= window:
            beats[-1].append(c)
        else:
            beats.append([c])
            anchor = c['start_time']
    return beats


# ---------------------------------------------------------------------------
# LLM call layer
# ---------------------------------------------------------------------------

_SYSTEM_MSG = ("You are an expert at creating concise audio descriptions that "
               "preserve visual details and on-screen text while fitting time constraints.")


def _llm_call(client, model_type: str, prompt: str, attempt: int,
              scene_label: str, max_retries: int) -> Optional[str]:
    """Single LLM call. Returns text on success, '' to retry, None on hard failure."""
    try:
        if model_type == MODEL_QWEN:
            model, proc = client['model'], client['processor']
            msgs = [{"role": "system", "content": _SYSTEM_MSG},
                    {"role": "user", "content": prompt}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = proc(text=[text], return_tensors="pt").to(model.device)
            out = model.generate(**inputs, max_new_tokens=150, temperature=0.7, do_sample=True)
            gen = out[:, inputs.input_ids.shape[1]:]
            return proc.batch_decode(gen, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True)[0].strip()
        if model_type == MODEL_GEMINI:
            cats = (types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT)
            resp = client["client"].models.generate_content(
                model=client["model_name"], contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0 if attempt == 0 else 0.7,
                    max_output_tokens=4196,
                    thinking_config=types.ThinkingConfig(thinking_budget=512),
                    safety_settings=[types.SafetySetting(
                        category=c, threshold=types.HarmBlockThreshold.BLOCK_NONE) for c in cats],
                ),
            )
            return (resp.text or "").strip()
        if model_type == MODEL_GPT4:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": _SYSTEM_MSG},
                          {"role": "user", "content": prompt}],
                temperature=0.5 if attempt == 0 else 1.0,
                max_tokens=200,
            )
            return resp.choices[0].message.content.strip()
        print(f"  - Unknown optimizer model type: {model_type}")
        return None
    except Exception as e:
        print(f"  - Error calling {model_type.upper()} (Scene {scene_label}, "
              f"Attempt {attempt+1}): {e}")
        if attempt == max_retries:
            return None
        time.sleep(2)
        return ""


def _retry_llm(client, model_type: str, build_prompt: Callable[[int, str], str],
               accept: Callable[[str], bool], scene_label: str,
               max_retries: int = 3) -> Optional[str]:
    """
    Run an LLM call with retries. `build_prompt(attempt, last_text)` produces
    the prompt for each attempt. `accept(text)` returns True if the response
    is acceptable. Returns the accepted text, or the last text seen if no
    attempt was accepted, or None on hard failure.
    """
    last = ""
    for attempt in range(max_retries + 1):
        prompt = build_prompt(attempt, last)
        result = _llm_call(client, model_type, prompt, attempt, scene_label, max_retries)
        if result is None:
            return None
        if result == "":
            continue
        last = result
        if accept(result):
            return result
    return last or None


# ---------------------------------------------------------------------------
# Single-clip operations: compress (time budget) and polish (verbosity only)
# ---------------------------------------------------------------------------

_COMPRESS_GUIDANCE = (
    "This is a VISUAL description. Preserve all concrete visual details "
    "(colors, materials, shapes, named objects, character actions and their "
    "order). Drop only filler: vague qualifiers ('very', 'really'), redundant "
    "phrasing, descriptions of state the action already implies. Do NOT change "
    "the meaning. Do NOT remove any factual visual detail."
)


def compress_single_clip(client, model_type: str, clip: Dict,
                         available: float, scene_label: str = "N/A") -> Dict:
    """Time-budget compression for a single Visual clip. Returns the shortest
    coherent attempt — if it fits, fits_in_gap=True; otherwise extended."""
    if clip['type'] == 'Text on Screen':
        raise ValueError("compress_single_clip is for Visual clips only.")
    original = clip['text']
    original_dur = get_tts_duration(original)

    def build(attempt: int, last: str) -> str:
        if attempt == 0:
            return (
                f'You are tightening an audio description for blind viewers so it fits a '
                f'narration window.\n\nORIGINAL TEXT: "{original}"\n\n{_COMPRESS_GUIDANCE}\n\n'
                f'AVAILABLE TIME: {available:.2f} seconds\n'
                f'TASK: Rewrite to fit in {available:.2f}s of speech '
                f'(~{available * 3:.0f} words at ~3 wps), without changing meaning.\n'
                f'GUIDELINES:\n- Result must be complete, grammatical sentence(s).\n'
                f'- Keep the same subject(s) and action(s).\n'
                f'- If already concise enough, return unchanged.\n'
                f'OUTPUT: Only the narration text. No explanations or markdown.'
            )
        return (
            f'Your previous attempt was still too long.\n\nORIGINAL TEXT: "{original}"\n'
            f'PREVIOUS ATTEMPT ({get_tts_duration(last):.2f}s): "{last}"\n\n'
            f'{_COMPRESS_GUIDANCE}\n\nAVAILABLE TIME: {available:.2f} seconds.\n'
            f'TASK: Tighter version that fits, still preserving meaning and detail. '
            f'Result must be a complete sentence.\nOUTPUT: One complete sentence.'
        )

    orig_words = len(original.split())
    min_words = max(3, int(orig_words * 0.3))
    best = {'text': None, 'dur': float('inf')}

    def accept(text: str) -> bool:
        dur = get_tts_duration(text)
        wc = len(text.split())
        coherent = wc >= min_words and text.rstrip().endswith(('.', '!', '?', '"'))
        print(f"  - Scene {scene_label}, compress: '{text[:60]}...' "
              f"Dur: {dur:.2f}s (Target: {available:.2f}s, Words: {wc}, "
              f"Coherent: {coherent})")
        if coherent and dur < best['dur']:
            best['text'], best['dur'] = text, dur
        return dur <= available and coherent

    _retry_llm(client, model_type, build, accept, scene_label)

    if best['text'] is None:
        print(f"  - Scene {scene_label}: no coherent attempt; keeping original "
              f"({original_dur:.2f}s) as extended.")
        return make_clip(clip['scene_number'], original, 'Visual', False,
                         duration=original_dur)

    fits = best['dur'] <= available
    if fits:
        print(f"  - Scene {scene_label}: compressed to {best['dur']:.2f}s, fits.")
    else:
        print(f"  - Scene {scene_label}: shortened {original_dur:.2f}s → "
              f"{best['dur']:.2f}s but still over {available:.2f}s; extended.")
    return make_clip(clip['scene_number'], best['text'], 'Visual', fits,
                     duration=best['dur'])


def polish_single_clip(client, model_type: str, clip: Dict,
                       scene_label: str = "N/A") -> Dict:
    """
    Two-step verbosity polish for orphan Visual clips. No time target.
    Step 1: ask if verbose (YES/NO). Step 2: rewrite only if YES.
    """
    if clip['type'] == 'Text on Screen':
        return make_clip(clip['scene_number'], clip['text'], clip['type'],
                         False, duration=clip['duration'])

    original = clip['text']

    # Step 1: verbosity check.
    verdict_prompt = (
        f'Is this audio description verbose? Reply with one word: YES or NO.\n\n'
        f'"{original}"'
    )
    verdict = _llm_call(client, model_type, verdict_prompt, 0, scene_label, max_retries=0)
    is_verbose = bool(verdict and verdict.strip().upper().startswith('YES'))

    if not is_verbose:
        print(f"  - Scene {scene_label}, polish: not verbose, kept as-is.")
        return make_clip(clip['scene_number'], original, 'Visual', False)

    # Step 2: rewrite.
    rewrite_prompt = (
        f'Rewrite this audio description more concisely. Keep the same meaning '
        f'and all visual details. Reply with only the rewritten sentence.\n\n'
        f'"{original}"'
    )
    result = _llm_call(client, model_type, rewrite_prompt, 0, scene_label, max_retries=0)

    polished = original
    if result:
        cand = result.strip().strip('"').strip("'").strip()
        if cand:
            polished = cand

    if polished != original:
        print(f"  - Scene {scene_label}, polish: '{original[:50]}...' -> '{polished[:50]}...'")
    else:
        print(f"  - Scene {scene_label}, polish: verbose but rewrite empty; kept original.")
    return make_clip(clip['scene_number'], polished, 'Visual', False)


# ---------------------------------------------------------------------------
# Multi-clip merging (mixed Visual+TOS, or pure Visual)
# ---------------------------------------------------------------------------

_MIXED_GUIDANCE = (
    "Some items are ON-SCREEN TEXT (verbatim quotes of what literally appears on "
    "screen) and some are VISUAL descriptions. ON-SCREEN TEXT MUST be reproduced "
    "verbatim, character-for-character — do NOT paraphrase, abbreviate, or drop "
    "any of it, and do NOT add narrative wrappers like 'a title card reads'. "
    "VISUAL portions MAY be tightened (drop filler, condense phrasing) while "
    "keeping key visual details. Combine items naturally into one or two sentences. "
    "Do NOT prefix output with labels like 'Text:', 'Caption:', or 'On screen:'."
)
_VISUAL_GUIDANCE = "All items are VISUAL descriptions; merge them into a flowing narration."


def merge_clips(client, model_type: str, clips: List[Dict],
                available: float, scene_label: str = "N/A") -> Dict:
    """Merge multi-clip beats. Pure-TOS beats are handled in _build_beat (no LLM)."""
    has_tos = any(c['type'] == 'Text on Screen' for c in clips)
    result_type = 'Visual'
    sn = clips[0]['scene_number']
    originals = [c['text'] for c in clips]
    flat = " ".join(originals)
    guidance = _MIXED_GUIDANCE if has_tos else _VISUAL_GUIDANCE

    parts = "\n".join(
        f'[ON-SCREEN TEXT]: "{c["text"]}"' if c['type'] == 'Text on Screen'
        else f'[VISUAL]: {c["text"]}'
        for c in clips
    )

    def build(attempt: int, last: str) -> str:
        if attempt == 0:
            return (
                f'You are merging audio descriptions for blind viewers.\n\n'
                f'INPUT ITEMS (in time order):\n{parts}\n\n{guidance}\n\n'
                f'AVAILABLE TIME: {available:.2f} seconds\n'
                f'TASK: Combine into ONE flowing narration (1-2 sentences) fitting '
                f'in {available:.2f}s of speech.\nGUIDELINES:\n'
                f'- Preserve on-screen text verbatim.\n'
                f'- Mention visual actions in original order.\n'
                f'- Keep visual details (colors, materials, shapes).\n'
                f'- Drop only filler.\n- Use conjunctions to chain ("as", "while", "then").\n'
                f'- ~3 words/second.\n- Complete grammatical sentence(s).\n'
                f'OUTPUT: Only the narration text. No explanations or markdown.'
            )
        return (
            f'Your previous attempt was slightly too long.\n\nPREVIOUS ATTEMPT '
            f'({get_tts_duration(last):.2f}s): "{last}"\n\nINPUT ITEMS:\n{parts}\n\n'
            f'{guidance}\n\nAVAILABLE TIME: {available:.2f}s.\n'
            f'TASK: Tighter version that fits, preserving on-screen text verbatim '
            f'and the key visual actions. Complete sentence(s) only.'
        )

    orig_words = len(flat.split())
    min_words = max(4, int(orig_words * 0.3))

    def accept(text: str) -> bool:
        text = strip_label_prefix(text) if has_tos else text
        # Hard verbatim check for TOS.
        if has_tos:
            missing = [c['text'] for c in clips
                       if c['type'] == 'Text on Screen' and c['text'] not in text]
            if missing:
                print(f"  - Scene {scene_label} merge: TOS altered/dropped "
                      f"{[m[:40]+'...' for m in missing]}. Retrying.")
                return False
        dur = get_tts_duration(text)
        wc = len(text.split())
        coherent = wc >= min_words and text.rstrip().endswith(('.', '!', '?', '"'))
        print(f"  - Scene {scene_label} merge ({result_type}): '{text[:60]}...' "
              f"Dur: {dur:.2f}s (Target: {available:.2f}s, Words: {wc}, "
              f"Coherent: {coherent})")
        return dur <= available and coherent

    result = _retry_llm(client, model_type, build, accept, scene_label)
    if result:
        result = strip_label_prefix(result) if has_tos else result
    if result and accept(result):
        return make_clip(sn, result, result_type, True,
                         duration=get_tts_duration(result),
                         original_texts=originals)

    # Fallback. For TOS-involved beats we MUST emit verbatim originals — the
    # LLM may have altered the on-screen text, so we can't trust its output.
    print(f"  - Scene {scene_label}: merge fallback to verbatim concatenation.")
    fallback = flat if has_tos else (result or flat)
    return make_clip(sn, fallback, result_type, False,
                     duration=get_tts_duration(fallback),
                     original_texts=originals)


# ---------------------------------------------------------------------------
# Beat dispatcher
# ---------------------------------------------------------------------------

def _build_beat(beat: List[Dict], client, model_type: str,
                available: float, scene_number) -> Dict:
    """
    Render a beat as one placed-clip dict.
      - Single TOS: verbatim (never compressed).
      - Single Visual fitting: verbatim.
      - Single Visual overflowing: compress via LLM (best attempt wins).
      - Multi pure-TOS: concatenate verbatim.
      - Multi mixed / pure-Visual: merge via LLM.
    """
    # Single-clip beats.
    if len(beat) == 1:
        c = beat[0]
        if c['type'] == 'Text on Screen':
            return make_clip(scene_number, c['text'], c['type'],
                             c['duration'] <= available, duration=c['duration'])
        if c['duration'] <= available:
            return make_clip(scene_number, c['text'], 'Visual', True,
                             duration=c['duration'])
        return compress_single_clip(client, model_type, c, available,
                                    scene_label=str(scene_number))

    # Multi-clip beats.
    if {c['type'] for c in beat} == {'Text on Screen'}:
        text = " ".join(c['text'] for c in beat)
        dur = get_tts_duration(text)
        print(f"  - Scene {scene_number}: pure-TOS multi-clip beat "
              f"({len(beat)} clips); concatenating verbatim ({dur:.2f}s, "
              f"available {available:.2f}s).")
        return make_clip(scene_number, text, 'Text on Screen', dur <= available,
                         duration=dur, original_texts=[c['text'] for c in beat])

    return merge_clips(client, model_type, beat, available, str(scene_number))


# ---------------------------------------------------------------------------
# Beat placement
# ---------------------------------------------------------------------------

def place_beats_at_targets(beats: List[List[Dict]], boundary_rel: float,
                           scene_start_abs: float, client, model_type: str,
                           scene_number, in_dialogue: bool = False) -> List[Dict]:
    """
    Place each beat at its target start time. Each beat's budget is the time
    until the next beat's target (or `boundary_rel` for the last beat).
    Overflowing beats are emitted extended at their target time and may overlap
    the next beat. Start times are never adjusted.

    Orphan beats (in_dialogue=True) skip the time-budget compressor entirely
    — single Visual clips go through polish_single_clip; TOS clips emit verbatim.
    """
    placed = []
    for i, beat in enumerate(beats):
        target = beat[0]['start_time']
        next_target = (beats[i + 1][0]['start_time']
                       if i + 1 < len(beats) else boundary_rel)
        available = max(0.0, next_target - target)

        if in_dialogue and len(beat) == 1:
            c = beat[0]
            bc = (make_clip(scene_number, c['text'], c['type'], False,
                            duration=c['duration'])
                  if c['type'] == 'Text on Screen'
                  else polish_single_clip(client, model_type, c, str(scene_number)))
        else:
            bc = _build_beat(beat, client, model_type, available, scene_number)

        bc['start_time'] = target + scene_start_abs
        bc['end_time'] = bc['start_time'] + bc['duration']
        bc['fits_in_gap'] = False if in_dialogue else (bc['duration'] <= available)
        placed.append(bc)
    return placed


# ---------------------------------------------------------------------------
# Scene processing
# ---------------------------------------------------------------------------

def process_scene(scene: Dict, client, model_type: str, min_gap: float) -> List[Dict]:
    sn = scene.get('scene_number', 'N/A')
    scene_start = scene.get('start_time', 0)
    scene_end_rel = ((scene.get('end_time', 0) - scene_start)
                     if scene.get('end_time') else scene.get('duration', 0))

    print(f"\n\n===== PROCESSING SCENE {sn} ({model_type.upper()}) =====")

    clips = get_scene_clips(scene)
    if not clips:
        print(f"-- Scene {sn}: no Visual or Text on Screen clips. Skipping.")
        return []

    vis = sum(1 for c in clips if c['type'] == 'Visual')
    tos = sum(1 for c in clips if c['type'] == 'Text on Screen')
    print(f"-- Scene {sn}: {vis} Visual + {tos} Text on Screen clips.")

    gaps = find_dialogue_gaps(scene, min_gap)
    print(f"-- Scene {sn}: {len(gaps)} dialogue-free gap(s) >= {min_gap}s")
    for i, g in enumerate(gaps, 1):
        print(f"     gap {i}: {g['start_time']:.2f}s..{g['end_time']:.2f}s "
              f"({g['duration']:.2f}s)")

    # Bucket clips into gaps; everything else is an orphan (during dialogue).
    buckets = [[] for _ in gaps]
    orphans = []
    for c in clips:
        for gi, g in enumerate(gaps):
            if g['start_time'] <= c['start_time'] < g['end_time']:
                buckets[gi].append(c)
                break
        else:
            orphans.append(c)

    placed = []
    for gi, (gap, bucket) in enumerate(zip(gaps, buckets), 1):
        if not bucket:
            continue
        print(f"\n  Gap {gi}: {len(bucket)} clip(s) to place.")
        beats = cluster_into_beats(bucket)
        print(f"    -> {len(beats)} beat(s) (within {MERGE_WINDOW}s)")
        placed.extend(place_beats_at_targets(
            beats, gap['end_time'], scene_start, client, model_type, sn,
            in_dialogue=False,
        ))

    if orphans:
        print(f"\n  Orphans: {len(orphans)} clip(s) during dialogue.")
        placed.extend(place_beats_at_targets(
            cluster_into_beats(orphans), scene_end_rel, scene_start,
            client, model_type, sn, in_dialogue=True,
        ))

    placed.sort(key=lambda x: x['start_time'])
    extended = sum(1 for c in placed if not c.get('fits_in_gap', True))
    if extended:
        print(f"  [info] Scene {sn}: {extended} extended clip(s).")
    return placed


# ---------------------------------------------------------------------------
# Client init + main
# ---------------------------------------------------------------------------

def _init_client(model_type: str):
    if model_type == MODEL_QWEN:
        print("Initializing LOCAL Qwen model with 4-bit quantization...")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-72B-Instruct", torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", device_map="auto",
            quantization_config=qcfg, cache_dir="../.cache",
        )
        proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")
        return {'model': model, 'processor': proc}
    if model_type == MODEL_GEMINI:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            print("Error: GEMINI_API_KEY not set."); return None
        print("Initializing Gemini API client...")
        return {"client": genai.Client(api_key=key), "model_name": "gemini-3-flash-preview"}
    if model_type == MODEL_GPT4:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            print("Error: OPENAI_API_KEY not set."); return None
        print("Initializing OpenAI API client...")
        return openai.OpenAI(api_key=key)
    return None


def main():
    p = argparse.ArgumentParser(description="Place and merge audio descriptions.")
    p.add_argument("video_folder")
    p.add_argument("--output", required=True)
    p.add_argument("--optimizer_model", choices=[MODEL_GEMINI, MODEL_QWEN, MODEL_GPT4],
                   default=MODEL_GPT4)
    p.add_argument("--min_gap", type=float, default=2.0)
    args = p.parse_args()

    video_id = os.path.basename(os.path.normpath(args.video_folder))
    scenes_folder = os.path.join(args.video_folder, f"{video_id}_scenes")
    candidates = [
        os.path.join(scenes_folder, f"scene_info_{args.optimizer_model}_filtered.json"),
        os.path.join(scenes_folder, f"scene_info_{args.optimizer_model}.json"),
        os.path.join(scenes_folder, "scene_info.json"),
    ]
    scenes_path = next((c for c in candidates if os.path.exists(c)), None)
    if not scenes_path:
        print(f"Error: No scene_info file in {scenes_folder}.")
        for c in candidates:
            print(f"    - {os.path.basename(c)}")
        return

    print(f"Using input scene file: {scenes_path}")
    if not scenes_path.endswith("_filtered.json"):
        print("WARNING: UNFILTERED scene_info. Run clip_analyze.py first.")

    with open(scenes_path, encoding="utf-8") as f:
        scenes = json.load(f)

    client = _init_client(args.optimizer_model)
    if not client:
        return

    all_clips = []
    for scene in scenes:
        all_clips.extend(process_scene(scene, client, args.optimizer_model, args.min_gap))

    out_path = os.path.join(scenes_folder, args.output)
    with open(out_path, 'w', encoding="utf-8") as f:
        json.dump(all_clips, f, indent=2)

    print(f"\nResults saved to: {out_path}")
    print(f"Total audio clips generated: {len(all_clips)}")


if __name__ == "__main__":
    main()