import json
import difflib
from jiwer import wer
import re
import subprocess
import argparse
import whisper_timestamped
import os
import onnxruntime
from collections import Counter

onnxruntime.set_default_logger_severity(3)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

import nltk
from nltk.corpus import words as nltk_words

from google.cloud import speech
from dotenv import load_dotenv
load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Constants
WHISPER_MODEL = "large-v3-turbo"

# Load English dictionary for garbage detection
try:
    _ENGLISH_WORDS = set(w.lower() for w in nltk_words.words())
except LookupError:
    nltk.download('words', quiet=True)
    _ENGLISH_WORDS = set(w.lower() for w in nltk_words.words())

# The NLTK words corpus is a lemma-only dictionary: it has no contractions,
# speech fillers, or inflected forms, all of which dominate real dialogue.
# normalize() strips apostrophes, so contractions are matched in their
# apostrophe-less form ("I'm" -> "im").
_ENGLISH_WORDS.update({
    # contractions (apostrophes already stripped by normalize)
    "im", "ive", "ill", "id", "youre", "youve", "youll", "youd",
    "hes", "shes", "weve", "theyre", "theyve", "theyll", "theyd",
    "isnt", "arent", "wasnt", "werent", "dont", "doesnt", "didnt",
    "cant", "couldnt", "wont", "wouldnt", "shouldnt", "hasnt",
    "havent", "hadnt", "thats", "whats", "whos", "wheres", "theres",
    "heres", "lets", "aint",
    # informal speech / fillers
    "gonna", "wanna", "gotta", "kinda", "sorta", "cmon", "yall",
    "uh", "um", "hmm", "hm", "mhm", "huh", "ooh", "whoa", "yeah",
    "yep", "yup", "nah", "nope", "ok", "okay", "alright", "hey",
    "bye", "oops", "ugh", "eh", "er", "erm",
})
print(f"[INIT] Loaded {len(_ENGLISH_WORDS)} English words for garbage detection")


def _is_english_word(token: str) -> bool:
    """Dictionary lookup with a fallback for inflected forms, which the
    lemma-only NLTK corpus lacks ("wanted"/"asked"/"looking" are not in it)."""
    if token in _ENGLISH_WORDS:
        return True
    for suffix in ("ing", "ed", "es", "s", "d"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            stem = token[: len(token) - len(suffix)]
            # want-ed, ask-ed, look-ing; mak-ing -> make; stopp-ed -> stop
            if stem in _ENGLISH_WORDS or stem + "e" in _ENGLISH_WORDS:
                return True
            if stem[-1] == stem[-2] and stem[:-1] in _ENGLISH_WORDS:
                return True
    return False


def match_captions(scene_start, scene_end, scene_duration, captions):
    scene_captions = []
    for caption in captions:
        cap_start = caption.get("start", 0)
        cap_duration = caption.get("duration", 0)
        cap_end = cap_start + cap_duration
        if cap_start < scene_end and cap_end > scene_start:
            overlap_start = max(cap_start, scene_start)
            overlap_end = min(cap_end, scene_end)
            if (overlap_end - overlap_start) >= (cap_duration * 0.5):
                scene_captions.append({
                    "text": caption.get("text", ""),
                    "start": max(cap_start - scene_start, 0),
                    "end": min(cap_end - scene_start, scene_duration)
                })
    return scene_captions


def extract_audio(scene_video_path, output_audio_path):
    if os.path.exists(output_audio_path):
        print(f"Audio already exists: {output_audio_path}, skipping extraction.")
        return

    command = [
        "ffmpeg", "-y",
        "-i", scene_video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_audio_path
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Extracted audio: {output_audio_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio from {scene_video_path}: {e.stderr.decode() if e.stderr else 'Unknown error'}")


def transcribe_whisper(model, wav_path):
    print(f"Transcribing with Whisper on audio: {wav_path}")
    try:
        result = whisper_timestamped.transcribe(
            model,
            wav_path,
            vad=True,
            beam_size=5,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8)
        )
        detected_lang = result.get("language", "en")
        transcripts = []
        for segment in result["segments"]:
            transcripts.append({
                "text": segment["text"].strip(),
                "start": segment["start"],
                "end": segment["end"],
                "confidence": segment["confidence"],
                "language": detected_lang,
            })
        print(f"Whisper transcription complete: {len(transcripts)} segments (lang={detected_lang})")
        return transcripts
    except Exception as e:
        print(f"Error transcribing with Whisper: {str(e)}")
        return []


def transcribe_google_speech(client, wav_path, language_code="en-US"):
    print(f"Transcribing with Google Speech-to-Text on audio: {wav_path} (lang={language_code})")
    try:
        with open(wav_path, "rb") as audio_file:
            audio_content = audio_file.read()

        audio = speech.RecognitionAudio(content=audio_content)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code=language_code,
            enable_word_time_offsets=True,
            enable_automatic_punctuation=True,
            model="video",
        )
        response = client.recognize(config=config, audio=audio)

        transcripts = []
        for result in response.results:
            alternative = result.alternatives[0]

            if alternative.words:
                first_word = alternative.words[0]
                last_word = alternative.words[-1]
                start_time = first_word.start_time.total_seconds()
                end_time = last_word.end_time.total_seconds()

                transcripts.append({
                    "text": alternative.transcript.strip(),
                    "start": start_time,
                    "end": end_time
                })

        print(f"Google Speech-to-Text transcription complete: {len(transcripts)} segments")
        return transcripts
    except Exception as e:
        print(f"Error during Google Speech-to-Text transcription: {str(e)}")
        return []


# Map Whisper's ISO-639-1 language codes to Google Speech BCP-47 codes.
# Falls back to "{lang}-XX" pattern handled in resolve_google_language_code.
_WHISPER_TO_GOOGLE_LANG = {
    "en": "en-US",
    "vi": "vi-VN",
    "zh": "zh",       # Mandarin (simplified) – Google accepts "zh" / "cmn-Hans-CN"
    "es": "es-ES",
    "fr": "fr-FR",
    "de": "de-DE",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "it": "it-IT",
    "pt": "pt-PT",
    "ru": "ru-RU",
    "ar": "ar-XA",
    "hi": "hi-IN",
    "th": "th-TH",
    "id": "id-ID",
    "nl": "nl-NL",
}


def resolve_google_language_code(whisper_lang: str) -> str:
    """Translate Whisper's detected language code into a Google Speech language_code."""
    if not whisper_lang:
        return "en-US"
    code = _WHISPER_TO_GOOGLE_LANG.get(whisper_lang.lower())
    if code:
        return code
    # Unknown language: default to English so the WER sanity check still runs,
    # though it will likely disagree and push to confidence-only filtering.
    print(f"  [lang] No Google mapping for Whisper lang '{whisper_lang}', defaulting to en-US")
    return "en-US"


_DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    # Google STT writes digits ("1") where Whisper writes words ("one");
    # unify so the comparison doesn't count formatting as disagreement.
    text = re.sub(r"\b\d\b", lambda m: _DIGIT_WORDS[m.group()], text)
    text = re.sub(r"\s+", " ", text)
    return text


def is_garbage(text: str, language: str = "en") -> bool:
    """Reject Whisper output that isn't real speech (repetition, non-word syllables, etc.).

    The repetition check is language-agnostic and always applies — it catches
    Whisper's most common failure mode (hallucinated repeated tokens on music or
    silence). The English-dictionary check is ENGLISH ONLY: applying it to other
    languages would falsely flag perfectly good speech, since none of those tokens
    appear in the English wordlist. For non-English segments we skip it and rely on
    the repetition check plus Whisper's own confidence score downstream.
    """
    norm = normalize(text)
    tokens = norm.split()
    print(f"  [is_garbage] lang={language}, {len(tokens)} tokens, first 5: {tokens[:5]}, text snippet: \"{text[:60]}\"")

    if not tokens:
        print(f"  [is_garbage] -> True (empty)")
        return True

    # Heavy repetition: one token dominates the segment (language-agnostic)
    if len(tokens) >= 6:
        most_common_count = Counter(tokens).most_common(1)[0][1]
        ratio = most_common_count / len(tokens)
        print(f"  [is_garbage] repetition ratio: {ratio:.2f}")
        if ratio >= 0.6:
            print(f"  [is_garbage] -> True (repetition)")
            return True

    # Dictionary check: ENGLISH ONLY. Skip for other languages.
    # Also skip very short segments — with few tokens the ratio is too noisy
    # to distinguish clipped-but-real speech from hallucination.
    if language == "en" and len(tokens) >= 6:
        real_word_ratio = sum(1 for t in tokens if _is_english_word(t)) / len(tokens)
        print(f"  [is_garbage] real word ratio: {real_word_ratio:.2f}")
        if real_word_ratio < 0.4:
            print(f"  [is_garbage] -> True (low real-word ratio)")
            return True
    elif language == "en":
        print(f"  [is_garbage] skipping dictionary check (short segment: {len(tokens)} tokens)")
    else:
        print(f"  [is_garbage] skipping dictionary check (non-English: {language})")

    print(f"  [is_garbage] -> False")
    return False


def verify_transcriptions(whisper_transcripts, google_transcripts, wer_threshold=0.35,
                          confidence_threshold=0.80, char_sim_threshold=0.70,
                          containment_threshold=0.85):
    """
    Decide which Whisper segments to keep, using Google as a sanity check.

    Strategy:
      1. Drop garbage Whisper segments (repetition / non-words).
      2. If Google agrees with Whisper globally, keep all surviving Whisper segments.
         Agreement is any of:
           - WER <= wer_threshold
           - character similarity >= char_sim_threshold: WER counts variant forms
             ("turn round" vs "turn around") and dropped filler words as full errors,
             so on short conversational clips it can read as "disagreement" when both
             engines heard the same speech.
           - containment >= containment_threshold: the shorter transcript is
             essentially a substring of the longer one. Google routinely transcribes
             only part of the audio (it silently drops quiet/overlapping speech), which
             makes symmetric metrics explode even though everything Google DID hear
             matches Whisper. A contained transcript is corroboration, not conflict.
      3. Otherwise, fall back to keeping only high-confidence Whisper segments.
    """
    # Step 1: filter garbage from Whisper up front
    clean_whisper = []
    for w in whisper_transcripts:
        raw_w = w.get("text", "").strip()
        if not raw_w:
            continue
        if is_garbage(raw_w, w.get("language", "en")):
            print(f"Discarded Whisper (not real speech): \"{raw_w[:80]}\"")
            continue
        clean_whisper.append(w)

    if not clean_whisper:
        print("No usable Whisper segments after garbage filter.")
        return []

    def to_segment(w):
        return {
            "text":  w.get("text", "").strip(),
            "start": w.get("start"),
            "end":   w.get("end"),
        }

    # Step 2: no Google -> fall back to confidence filter on Whisper alone
    if not google_transcripts:
        verified = []
        for w in clean_whisper:
            conf = w.get("confidence", 0)
            if conf >= confidence_threshold:
                seg = to_segment(w)
                verified.append(seg)
                print(f"Added WHISPER_HIGH_CONFIDENCE: \"{seg['text']}\" "
                      f"({seg['start']}–{seg['end']}) conf={conf:.2f}")
            else:
                print(f"Discarded Whisper (no Google, conf {conf:.2f} < {confidence_threshold}): \"{w.get('text','').strip()}\"")
        verified.sort(key=lambda s: s.get("start", 0))
        return verified

    # Step 3: compare Whisper and Google globally
    whisper_combined = " ".join(w.get("text", "").strip() for w in clean_whisper)
    google_combined  = " ".join(g.get("text", "").strip() for g in google_transcripts if g.get("text", "").strip())

    norm_whisper = normalize(whisper_combined)
    norm_google  = normalize(google_combined)
    print(f"NORM WHISPER, {norm_whisper}")
    print(f"NORM GOOGLE,  {norm_google}")

    if not norm_google:
        print("Google transcript empty after normalization; falling back to confidence filter.")
        verified = []
        for w in clean_whisper:
            conf = w.get("confidence", 0)
            if conf >= confidence_threshold:
                verified.append(to_segment(w))
        verified.sort(key=lambda s: s.get("start", 0))
        return verified

    overall_wer = wer(norm_google, norm_whisper)
    char_sim = difflib.SequenceMatcher(None, norm_google, norm_whisper).ratio()

    # How much of the shorter transcript is found (in order) inside the longer one.
    shorter, longer = sorted([norm_google, norm_whisper], key=len)
    sm = difflib.SequenceMatcher(None, shorter, longer)
    containment = sum(b.size for b in sm.get_matching_blocks()) / len(shorter)
    print(f"Overall WER between combined transcripts: {overall_wer:.4f}, "
          f"char similarity: {char_sim:.4f}, containment: {containment:.4f}")

    verified = []

    if (overall_wer <= wer_threshold or char_sim >= char_sim_threshold
            or containment >= containment_threshold):
        # Whisper and Google substantially agree -> trust all surviving Whisper segments
        print(f"Transcripts agree (WER={overall_wer:.4f}, char_sim={char_sim:.4f}, "
              f"containment={containment:.4f}); keeping all Whisper segments.")
        for w in clean_whisper:
            seg = to_segment(w)
            verified.append(seg)
            print(f"Added VERIFIED_WHISPER: \"{seg['text']}\" ({seg['start']}–{seg['end']})")
    else:
        # Disagreement -> only trust Whisper where the model is itself confident
        print(f"Transcripts disagree (WER={overall_wer:.4f} > {wer_threshold}, "
              f"char_sim={char_sim:.4f} < {char_sim_threshold}, "
              f"containment={containment:.4f} < {containment_threshold}); keeping only high-confidence Whisper.")
        for w in clean_whisper:
            conf = w.get("confidence", 0)
            if conf >= confidence_threshold:
                seg = to_segment(w)
                verified.append(seg)
                print(f"Added WHISPER_HIGH_CONFIDENCE: \"{seg['text']}\" "
                      f"({seg['start']}–{seg['end']}) conf={conf:.2f}")
            else:
                print(f"Discarded Whisper (low conf {conf:.2f} amid disagreement): \"{w.get('text','').strip()}\"")

    verified.sort(key=lambda s: s.get("start", 0))
    print(f"Verification complete: {len(verified)} segments added.")
    return verified


def should_discard_captions(global_transcript_text, global_caption_text, threshold=0.8):
    similarity = difflib.SequenceMatcher(None, global_transcript_text, global_caption_text).ratio()
    print(f"Global transcript vs captions similarity: {similarity:.2f}")
    return similarity >= threshold


def update_scene_transcripts(video_folder, device="cuda", global_caption_threshold=0.8):
    video_id = os.path.basename(os.path.normpath(video_folder))
    scene_json_path = os.path.join(video_folder, f"{video_id}_scenes", "scene_info.json")

    if not os.path.exists(scene_json_path):
        print(f"Scene JSON file not found: {scene_json_path}")
        return

    with open(scene_json_path, "r") as f:
        scenes = json.load(f)

    # Load models once for all scenes
    print("Loading Whisper model...")
    whisper_model = whisper_timestamped.load_model(WHISPER_MODEL, device=device)
    print("Initializing Google Speech client...")
    google_client = speech.SpeechClient()

    # Process scenes sequentially
    updated_scenes = []
    for i, scene in enumerate(scenes):
        scene_number = scene.get('scene_number', i+1)
        print(f"\n{'='*50}")
        print(f"Processing scene {scene_number} ({i+1}/{len(scenes)})...")
        print(f"{'='*50}")

        scene_path = scene.get("scene_path")
        if not scene_path or not os.path.exists(scene_path):
            print(f"Scene path not found, skipping: {scene_path}")
            updated_scenes.append(scene)
            continue

        audio_path = scene_path.replace(".mp4", ".wav")
        extract_audio(scene_path, audio_path)

        if not os.path.exists(audio_path):
            print(f"Audio file not created, skipping transcription")
            scene["transcript"] = []
            updated_scenes.append(scene)
            continue

        # Transcribe with Whisper first so we know the detected language,
        # then run Google Speech with a matching language_code.
        whisper_trans = transcribe_whisper(whisper_model, audio_path)

        detected_lang = whisper_trans[0].get("language", "en") if whisper_trans else "en"
        google_lang_code = resolve_google_language_code(detected_lang)
        google_trans = transcribe_google_speech(google_client, audio_path, language_code=google_lang_code)

        # Verify and combine transcriptions
        scene["transcript"] = verify_transcriptions(whisper_trans, google_trans)

        # Clean up audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"Cleaned up audio file: {audio_path}")

        updated_scenes.append(scene)

    # Combine all transcripts for global comparison
    global_transcript_text = " ".join(
        " ".join(seg["text"] for seg in scene.get("transcript", [])) for scene in updated_scenes
    ).strip()

    # Load and process captions
    captions = None
    captions_path = os.path.join(video_folder, f"{video_id}.json")
    if os.path.exists(captions_path):
        try:
            with open(captions_path, "r") as f:
                cap_data = json.load(f)
                captions = cap_data.get("captions", [])
                print(f"\nLoaded {len(captions)} captions from {captions_path}")
        except Exception as e:
            print(f"Error loading captions: {str(e)}")

    # Match captions to scenes
    if captions and global_transcript_text:
        global_caption_text = " ".join([cap["text"] for cap in captions])
        if should_discard_captions(global_transcript_text, global_caption_text, global_caption_threshold):
            print("Global captions are very similar to the transcript. Discarding captions for all scenes.")
            for scene in updated_scenes:
                scene["captions"] = []
        else:
            print("Matching captions to individual scenes...")
            for scene in updated_scenes:
                scene_start = scene.get("start", 0)
                scene_end = scene.get("end", 0)
                if scene_end > scene_start:
                    scene_duration = scene_end - scene_start
                    scene_captions = match_captions(scene_start, scene_end, scene_duration, captions)
                    scene["captions"] = scene_captions
                    print(f"Scene {scene.get('scene_number')}: matched {len(scene_captions)} captions")
                else:
                    scene["captions"] = []
    else:
        print("No captions available or no transcript generated. Setting empty captions for all scenes.")
        for scene in updated_scenes:
            scene["captions"] = []

    # Save updated scene information
    with open(scene_json_path, "w") as out_f:
        json.dump(updated_scenes, out_f, indent=2)
    print(f"\n{'='*50}")
    print(f"Updated scene JSON with transcripts saved to: {scene_json_path}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe scene audio using Whisper and Google Speech-to-Text, verify transcripts, and update scene JSON with optional captions."
    )
    parser.add_argument("video_folder", type=str,
                        help="Path to the video folder (e.g., videos/video_id). The scene_info.json file is expected at videos/video_id/video_id_scenes/scene_info.json")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for Whisper transcription (default: cuda)")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Similarity threshold for transcription verification (default: 0.8)")

    args = parser.parse_args()
    update_scene_transcripts(args.video_folder, args.device, args.threshold)


if __name__ == '__main__':
    main()