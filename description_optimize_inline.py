import json
import tempfile
import subprocess
import os
import argparse
import time
from typing import List, Dict
import torch
import openai
from google import genai
from google.genai import types
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

from dotenv import load_dotenv
load_dotenv()

MODEL_QWEN = "qwen"
MODEL_GEMINI = "gemini"
MODEL_GPT4 = "gpt"


def get_tts_duration(text: str, speaking_rate: float = 1.25) -> float:
    if not text or text.isspace():
        return 0.0
    words = len(text.split())
    # Google TTS speaks ~150 words per minute at normal speed; at 1.25x = 187.5 wpm.
    words_per_minute = 150 * speaking_rate
    duration = (words / words_per_minute) * 60
    return max(0.5, duration)


def get_scene_clips(scene: Dict) -> List[Dict]:
    clips = []
    for clip_data in scene.get('audio_clips', []):
        text = clip_data.get('text')
        if not text:
            continue
        duration = get_tts_duration(text)
        clips.append({
            'start_time': clip_data.get('start_time', 0),
            'text': text,
            'type': clip_data['type'],
            'scene_number': scene.get('scene_number', 'N/A'),
            'duration': duration,
            'end_time': clip_data.get('start_time', 0) + duration,
        })
    clips.sort(key=lambda x: x['start_time'])
    return clips


def separate_clip_types(clips: List[Dict]) -> tuple[List[Dict], List[Dict]]:
    text_clips = [c for c in clips if c['type'] == 'Text on Screen']
    visual_clips = [c for c in clips if c['type'] == 'Visual']
    return text_clips, visual_clips


def find_gaps_around_text_clips(scene: Dict, text_clips: List[Dict], min_gap_duration: float) -> List[Dict]:
    scene_duration = scene.get('duration', 0)
    if not scene_duration and 'start_time' in scene and 'end_time' in scene:
        scene_duration = scene['end_time'] - scene['start_time']
    if not scene_duration:
        print(f"Warning: Scene {scene.get('scene_number')} missing duration for gap calculation.")
        return []

    occupied_segments = [{'start': c['start_time'], 'end': c['end_time']} for c in text_clips]
    if 'transcript' in scene and scene['transcript']:
        for segment in scene['transcript']:
            occupied_segments.append({'start': segment.get('start', 0), 'end': segment.get('end', 0)})

    occupied_segments.sort(key=lambda x: x['start'])

    merged_segments = []
    if occupied_segments:
        current = occupied_segments[0].copy()
        for seg in occupied_segments[1:]:
            if seg['start'] <= current['end']:
                current['end'] = max(current['end'], seg['end'])
            else:
                merged_segments.append(current)
                current = seg.copy()
        merged_segments.append(current)

    eligible_gaps = []
    cursor = 0.0
    for seg in merged_segments:
        if seg['start'] > cursor:
            gap = seg['start'] - cursor
            if gap >= min_gap_duration:
                eligible_gaps.append({'start_time': cursor, 'end_time': seg['start'], 'duration': gap})
        cursor = max(cursor, seg['end'])

    if cursor < scene_duration:
        gap = scene_duration - cursor
        if gap >= min_gap_duration:
            eligible_gaps.append({'start_time': cursor, 'end_time': scene_duration, 'duration': gap})

    if not merged_segments and scene_duration >= min_gap_duration:
        eligible_gaps.append({'start_time': 0, 'end_time': scene_duration, 'duration': scene_duration})

    return eligible_gaps


def optimize_with_qwen(optimizer_client: Dict, prompt: str) -> str:
    model = optimizer_client['model']
    processor = optimizer_client['processor']
    messages = [
        {"role": "system", "content": "You are an expert at creating concise audio descriptions that preserve visual details (colors, materials, comparisons) while fitting time constraints."},
        {"role": "user", "content": prompt},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=150, temperature=0.7, do_sample=True)
    input_token_len = inputs.input_ids.shape[1]
    generated_ids = output_ids[:, input_token_len:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()


def optimize_combined_clips(optimizer_client, optimizer_model_type, clips_to_optimize, available_duration,
                            gap_start=None, max_retries=3, scene_number_for_logging="N/A"):
    """
    Merge all clips into a single description fitting available_duration if possible.
    If it doesn't fit after retries, return the best attempt as over-budget — DO NOT
    drop clips. Filtering is the responsibility of clip_analyze.py; this function's
    job is placement and merging only.
    """
    if not clips_to_optimize:
        return None, False

    result, fits = _attempt_merge(
        optimizer_client, optimizer_model_type,
        clips_to_optimize, available_duration, max_retries,
        scene_number_for_logging,
    )

    if fits:
        return result, True

    # Didn't fit cleanly, but we don't drop curated clips. Return the best attempt
    # marked as over-budget so downstream tools know.
    print(f"  - Scene {scene_number_for_logging}: {len(clips_to_optimize)}-clip merge "
          f"didn't fit in {available_duration:.2f}s. Keeping over-budget — clip_analyze "
          f"already decided these are necessary.")
    return result, False


def _attempt_merge(optimizer_client, optimizer_model_type, clips_to_optimize, available_duration,
                   max_retries, scene_number_for_logging):
    combined_text = " ".join([c['text'] for c in clips_to_optimize])
    optimized_text = ""
    tts_duration = float('inf')

    original_tts_duration = get_tts_duration(combined_text)
    if available_duration < original_tts_duration * 0.4:
        print(f"  - Scene {scene_number_for_logging}: {len(clips_to_optimize)} clips need "
              f"~{original_tts_duration:.2f}s but only {available_duration:.2f}s available. "
              f"Skipping merge attempt.")
        return {
            'scene_number': clips_to_optimize[0]['scene_number'],
            'text': combined_text, 'type': 'Visual',
            'duration': original_tts_duration,
            'fits_in_gap': False,
            'original_texts': [c['text'] for c in clips_to_optimize],
        }, False

    for attempt in range(max_retries + 1):
        if attempt == 0:
            prompt = (f'You are merging visual descriptions for an audio description track for blind viewers.\n'
                      f'ORIGINAL DESCRIPTIONS: "{combined_text}"\n'
                      f'AVAILABLE TIME: {available_duration:.2f} seconds\n'
                      f'TASK: Combine these descriptions into ONE flowing sentence (or two if needed) '
                      f'that fits in {available_duration:.2f} seconds of speech.\n'
                      f'GUIDELINES:\n'
                      f'- Mention ALL key actions in their original order.\n'
                      f'- KEEP visual details that a blind viewer cannot otherwise perceive — colors, '
                      f'materials, shapes, comparisons (e.g., "like dominoes", "chain-link", "red"). '
                      f'These details ARE the description for a blind viewer.\n'
                      f'- Drop only filler — articles where they can be omitted, vague qualifiers '
                      f'("very", "really"), and descriptions of state that the action already implies '
                      f'(e.g., "parked bicycles" can become "bicycles" if the action is kicking them).\n'
                      f'- Use conjunctions like "then", "before", "as", "while" to chain actions.\n'
                      f'- Use strong verbs that pack meaning ("hurled" rather than "threw hard").\n'
                      f'- The result MUST be a complete, grammatical sentence.\n'
                      f'- Speaking rate: ~3 words per second.\n'
                      f'OUTPUT FORMAT: Provide only the optimized description text. '
                      f'No explanations, no preamble, no markdown.')
        else:
            prompt = (f'Your previous attempt was slightly too long.\n'
                      f'PREVIOUS ATTEMPT (spoken duration {tts_duration:.2f}s): "{optimized_text}"\n'
                      f'ORIGINAL DESCRIPTIONS: "{combined_text}"\n'
                      f'AVAILABLE TIME: {available_duration:.2f} seconds.\n'
                      f'TASK: Produce a tighter version that fits in {available_duration:.2f} seconds, '
                      f'while still mentioning ALL the key actions and remaining a complete sentence.\n'
                      f'GUIDELINES: Combine actions with conjunctions. Drop filler, articles, and vague '
                      f'qualifiers — but KEEP visual details that carry the look of the scene (colors, '
                      f'materials, comparisons like "like dominoes", "chain-link"). Do NOT drop entire '
                      f'actions or visually distinctive details. Do NOT produce sentence fragments.\n'
                      f'OUTPUT FORMAT: One or two complete sentences. No explanations.')

        try:
            if optimizer_model_type == MODEL_QWEN:
                optimized_text = optimize_with_qwen(optimizer_client, prompt)

            elif optimizer_model_type == MODEL_GEMINI:
                response = optimizer_client["client"].models.generate_content(
                    model=optimizer_client["model_name"],
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=2048,
                        thinking_config=types.ThinkingConfig(thinking_budget=512),
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
                optimized_text = response.text.strip()

            elif optimizer_model_type == MODEL_GPT4:
                response = optimizer_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert at creating concise audio descriptions that preserve visual details (colors, materials, comparisons) while fitting time constraints."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.5 if attempt == 0 else 1.0,
                    max_tokens=150,
                )
                optimized_text = response.choices[0].message.content.strip()

            else:
                print(f"  - Unknown optimizer model type: {optimizer_model_type}")
                return None, False

        except Exception as e:
            print(f"  - Error calling {optimizer_model_type.upper()} (Scene {scene_number_for_logging}, Attempt {attempt+1}): {e}")
            if attempt == max_retries:
                return None, False
            time.sleep(2)
            continue

        tts_duration = get_tts_duration(optimized_text)

        word_count = len(optimized_text.split())
        ends_cleanly = optimized_text.rstrip().endswith(('.', '!', '?'))
        original_word_count = len(combined_text.split())
        min_acceptable_words = max(4, int(original_word_count * 0.3))
        looks_coherent = word_count >= min_acceptable_words and ends_cleanly

        print(f"  - Scene {scene_number_for_logging}, {len(clips_to_optimize)}-clip merge, "
              f"Attempt {attempt+1}: Text='{optimized_text[:60]}...', "
              f"Dur: {tts_duration:.2f}s (Target: {available_duration:.2f}s, "
              f"Words: {word_count}, Coherent: {looks_coherent})")

        if tts_duration <= available_duration and optimized_text and looks_coherent:
            print(f"  - Success! {len(clips_to_optimize)}-clip merge fits and is coherent.")
            return {
                'scene_number': clips_to_optimize[0]['scene_number'],
                'text': optimized_text, 'type': 'Visual', 'duration': tts_duration,
                'fits_in_gap': True, 'original_texts': [c['text'] for c in clips_to_optimize],
            }, True

    print(f"  - Scene {scene_number_for_logging}: Retries exhausted for {len(clips_to_optimize)}-clip merge.")
    return {
        'scene_number': clips_to_optimize[0]['scene_number'],
        'text': optimized_text if optimized_text else combined_text, 'type': 'Visual',
        'duration': tts_duration if optimized_text else original_tts_duration,
        'fits_in_gap': False,
        'original_texts': [c['text'] for c in clips_to_optimize],
    }, False


def process_scene(scene: Dict, optimizer_client, optimizer_model_type: str, min_gap_duration: float):
    scene_number = scene.get('scene_number', 'N/A')
    scene_start_abs = scene.get('start_time', 0)

    print(f"\n\n===== PROCESSING SCENE {scene_number} (OPTIMIZED PLACEMENT with {optimizer_model_type.upper()}) =====")

    clips_from_scene = get_scene_clips(scene)
    text_clips, visual_clips = separate_clip_types(clips_from_scene)
    print(f"\n-- Scene {scene_number}: Found {len(text_clips)} text clips and {len(visual_clips)} visual clips")

    placed_clips = [{'scene_number': c['scene_number'], 'start_time': c['start_time'] + scene_start_abs,
                     'end_time': c['end_time'] + scene_start_abs, 'duration': c['duration'],
                     'type': 'Text on Screen', 'text': c['text']} for c in text_clips]

    eligible_gaps = find_gaps_around_text_clips(scene, text_clips, min_gap_duration)
    print(f"\n-- Scene {scene_number}: Found {len(eligible_gaps)} eligible gaps >= {min_gap_duration}s")

    processed_clip_ids = set()

    for gap_idx, gap in enumerate(eligible_gaps):
        clips_in_gap_timeframe = [
            c for c in visual_clips
            if id(c) not in processed_clip_ids
            and gap['start_time'] - 1.5 <= c['start_time'] < gap['end_time']
        ]
        clips_in_gap_timeframe.sort(key=lambda x: x['start_time'])
        if not clips_in_gap_timeframe:
            continue

        print(f"\nProcessing Gap {gap_idx+1} (Duration: {gap['duration']:.2f}s) "
              f"with {len(clips_in_gap_timeframe)} associated clips.")

        gap_start_abs = gap['start_time'] + scene_start_abs
        gap_end_abs = gap['end_time'] + scene_start_abs

        # Sub-cluster within the gap by proximity. Only clips at essentially the
        # same moment (within MERGE_WINDOW of the cluster's first clip) get merged.
        # Clips that happen at distinctly different moments stay separate beats.
        MERGE_WINDOW = 1.0
        sub_clusters = []
        current_cluster = [clips_in_gap_timeframe[0]]
        for clip in clips_in_gap_timeframe[1:]:
            if clip['start_time'] - current_cluster[0]['start_time'] <= MERGE_WINDOW:
                current_cluster.append(clip)
            else:
                sub_clusters.append(current_cluster)
                current_cluster = [clip]
        sub_clusters.append(current_cluster)

        # For each sub-cluster, merge if it has multiple clips, otherwise place as-is.
        # Each sub-cluster gets placed at its own original timestamp within the gap.
        for sub_cluster in sub_clusters:
            cluster_start_rel = sub_cluster[0]['start_time']
            cluster_start_abs = cluster_start_rel + scene_start_abs

            if len(sub_cluster) == 1:
                clip = sub_cluster[0]
                placed_clips.append({
                    'scene_number': scene_number,
                    'start_time': cluster_start_abs,
                    'end_time': cluster_start_abs + clip['duration'],
                    'duration': clip['duration'],
                    'type': 'Visual',
                    'text': clip['text'],
                    'fits_in_gap': cluster_start_abs + clip['duration'] <= gap_end_abs,
                })
                processed_clip_ids.add(id(clip))
                continue

            # Multi-clip sub-cluster: merge into one description.
            # Available duration = how much room until either the next sub-cluster or the gap end.
            sub_idx = sub_clusters.index(sub_cluster)
            if sub_idx + 1 < len(sub_clusters):
                next_start_rel = sub_clusters[sub_idx + 1][0]['start_time']
                available = next_start_rel - cluster_start_rel
            else:
                available = gap['end_time'] - cluster_start_rel

            optimized_clip_data, fits = optimize_combined_clips(
                optimizer_client, optimizer_model_type,
                sub_cluster, available,
                gap_start=cluster_start_rel,
                scene_number_for_logging=scene_number,
            )

            if optimized_clip_data:
                optimized_clip_data['start_time'] = cluster_start_abs
                optimized_clip_data['end_time'] = cluster_start_abs + optimized_clip_data['duration']
                if optimized_clip_data['end_time'] > gap_end_abs:
                    optimized_clip_data['fits_in_gap'] = False

                placed_clips.append(optimized_clip_data)
                for clip in sub_cluster:
                    processed_clip_ids.add(id(clip))

    # Orphan clips: didn't land in any eligible gap.
    remaining_visual_clips = [c for c in visual_clips if id(c) not in processed_clip_ids]

    if remaining_visual_clips:
        print(f"\n-- Scene {scene_number}: {len(remaining_visual_clips)} orphan clips. "
              f"Clustering near-simultaneous ones.")

        MERGE_WINDOW = 1.0
        remaining_visual_clips.sort(key=lambda x: x['start_time'])

        clusters = []
        current_cluster = [remaining_visual_clips[0]]
        for clip in remaining_visual_clips[1:]:
            if clip['start_time'] - current_cluster[0]['start_time'] <= MERGE_WINDOW:
                current_cluster.append(clip)
            else:
                clusters.append(current_cluster)
                current_cluster = [clip]
        clusters.append(current_cluster)

        for cluster in clusters:
            cluster_start_abs = cluster[0]['start_time'] + scene_start_abs

            if len(cluster) == 1:
                clip = cluster[0]
                placed_clips.append({
                    'scene_number': scene_number,
                    'start_time': cluster_start_abs,
                    'end_time': cluster_start_abs + clip['duration'],
                    'duration': clip['duration'],
                    'type': 'Visual',
                    'text': clip['text'],
                    'fits_in_gap': False,
                })
                continue

            cluster_span = cluster[-1]['start_time'] - cluster[0]['start_time']
            target_duration = max(cluster_span, MERGE_WINDOW)

            print(f"  - Merging cluster of {len(cluster)} orphan clips at {cluster_start_abs:.2f}s "
                  f"(target {target_duration:.2f}s)")
            merged, _fits = optimize_combined_clips(
                optimizer_client, optimizer_model_type,
                cluster, target_duration,
                gap_start=cluster[0]['start_time'],
                scene_number_for_logging=scene_number,
            )

            if merged:
                placed_clips.append({
                    'scene_number': scene_number,
                    'start_time': cluster_start_abs,
                    'end_time': cluster_start_abs + merged['duration'],
                    'duration': merged['duration'],
                    'type': 'Visual',
                    'text': merged['text'],
                    'fits_in_gap': False,
                    'original_texts': merged.get('original_texts', []),
                })
            else:
                for clip in cluster:
                    clip_start = clip['start_time'] + scene_start_abs
                    placed_clips.append({
                        'scene_number': scene_number,
                        'start_time': clip_start,
                        'end_time': clip_start + clip['duration'],
                        'duration': clip['duration'],
                        'type': 'Visual',
                        'text': clip['text'],
                        'fits_in_gap': False,
                    })

    placed_clips.sort(key=lambda x: x['start_time'])

    start_times = [c['start_time'] for c in placed_clips]
    if len(start_times) != len(set(start_times)):
        print(f"NOTE: Scene {scene_number} has clips with identical start times.")

    return placed_clips


def main():
    parser = argparse.ArgumentParser(description="Place and merge audio descriptions on the scene timeline.")
    parser.add_argument("video_folder", help="Path to the video folder")
    parser.add_argument("--output", help="Output JSON file name", required=True)
    parser.add_argument("--optimizer_model", type=str, choices=[MODEL_GEMINI, MODEL_QWEN, MODEL_GPT4],
                        default=MODEL_GPT4,
                        help="Choose the model for merging descriptions: 'gemini', 'qwen', or 'gpt'.")
    parser.add_argument("--min_gap", type=float, default=2.0,
                        help="Minimum gap duration in seconds to consider for placing descriptions.")
    args = parser.parse_args()

    video_id = os.path.basename(os.path.normpath(args.video_folder))
    scenes_folder = os.path.join(args.video_folder, f"{video_id}_scenes")

    # Prefer the FILTERED file from clip_analyze.py (production pipeline).
    # Fall back to the unfiltered scene_info as a convenience for ad-hoc runs.
    candidate_paths = [
        os.path.join(scenes_folder, f"scene_info_{args.optimizer_model}_filtered.json"),
        os.path.join(scenes_folder, f"scene_info_{args.optimizer_model}.json"),
        os.path.join(scenes_folder, "scene_info.json"),
    ]

    scenes_path = next((p for p in candidate_paths if os.path.exists(p)), None)
    if not scenes_path:
        print(f"Error: No scene_info file found in {scenes_folder}.")
        print("  Looked for (in order):")
        for p in candidate_paths:
            print(f"    - {os.path.basename(p)}")
        return

    print(f"Using input scene file: {scenes_path}")
    if not scenes_path.endswith("_filtered.json"):
        print("WARNING: Reading an UNFILTERED scene_info file. For production quality, "
              "run clip_analyze.py first to filter out hallucinated/unnecessary clips.")

    with open(scenes_path, "r", encoding="utf-8") as f:
        scenes = json.load(f)

    optimizer_client = None
    if args.optimizer_model == MODEL_QWEN:
        print("Initializing LOCAL Qwen model with 4-bit quantization...")
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
        optimizer_client = {'model': model, 'processor': processor}
    elif args.optimizer_model == MODEL_GEMINI:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            print("Error: GEMINI_API_KEY environment variable not set.")
            return
        print("Initializing Gemini API client...")
        client = genai.Client(api_key=gemini_api_key)
        optimizer_client = {"client": client, "model_name": "gemini-3-flash-preview"}
    elif args.optimizer_model == MODEL_GPT4:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("Error: OPENAI_API_KEY environment variable not set.")
            return
        print("Initializing OpenAI API client...")
        optimizer_client = openai.OpenAI(api_key=openai_api_key)

    if not optimizer_client:
        print("Error: Optimizer client could not be initialized.")
        return

    all_clips = []
    for scene in scenes:
        scene_clips = process_scene(scene, optimizer_client, args.optimizer_model, args.min_gap)
        all_clips.extend(scene_clips)

    output_file_path = os.path.join(scenes_folder, args.output)
    with open(output_file_path, 'w', encoding="utf-8") as f:
        json.dump(all_clips, f, indent=2)

    print(f"\nResults saved to: {output_file_path}")
    print(f"Total audio clips generated: {len(all_clips)}")


if __name__ == "__main__":
    main()