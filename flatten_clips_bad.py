"""
Minimal bridge for the INTENTIONALLY-BAD pipeline.

The normal pipeline runs description_optimize_inline.py, which places clips in
dialogue gaps, merges nearby beats, and compresses overflowing text to fit the
available narration window — all the work that makes descriptions land well.

The bad pipeline SKIPS that step. This script does only the bare mechanical
conversion that prepare_final_data.py needs and nothing else:
  - flatten the per-scene `audio_clips` into a single flat list,
  - convert each clip's scene-relative `start_time` to an ABSOLUTE video time
    (scene start_time + relative start_time),
  - attach an approximate TTS `duration` / `end_time`.

There is deliberately NO gap-fitting, merging, or compression, so descriptions
overlap dialogue and one another — a faithfully worse result.
"""

import argparse
import json
import os


def get_tts_duration(text: str, speaking_rate: float = 1.25) -> float:
    """Approximate TTS speaking duration. ~150 wpm * speaking_rate.

    Mirrors description_optimize_inline.get_tts_duration so end_times are
    comparable to the good pipeline's output.
    """
    if not text or text.isspace():
        return 0.0
    words = len(text.split())
    return max(0.5, (words / (150 * speaking_rate)) * 60)


def flatten(scenes):
    all_clips = []
    for scene in scenes:
        scene_start = scene.get("start_time", 0) or 0
        scene_number = scene.get("scene_number", "N/A")
        for c in scene.get("audio_clips", []):
            text = c.get("text")
            if not text or c.get("type") not in ("Visual", "Text on Screen"):
                continue
            try:
                rel_start = float(c.get("start_time", 0))
            except (TypeError, ValueError):
                rel_start = 0.0
            abs_start = scene_start + rel_start
            dur = get_tts_duration(text)
            all_clips.append({
                "scene_number": scene_number,
                "text": text,
                "type": c["type"],
                "duration": dur,
                "fits_in_gap": False,
                "start_time": abs_start,
                "end_time": abs_start + dur,
            })
    all_clips.sort(key=lambda x: x["start_time"])
    return all_clips


def main():
    p = argparse.ArgumentParser(
        description="Flatten per-scene audio_clips to a flat, absolute-time clip "
                    "list (bad-pipeline replacement for description_optimize_inline.py)."
    )
    p.add_argument("input", help="Path to a scene_info*.json file (clips nested per scene).")
    p.add_argument("--output", required=True, help="Path to write the flat clip list JSON.")
    args = p.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}")
        raise SystemExit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        scenes = json.load(f)

    all_clips = flatten(scenes)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_clips, f, indent=2)

    print(f"Flattened {len(all_clips)} clips (no optimization) -> {args.output}")


if __name__ == "__main__":
    main()
