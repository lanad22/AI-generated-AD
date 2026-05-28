import json
import difflib
from jiwer import wer
import re
import subprocess
import argparse
import whisper_timestamped
import os
import onnxruntime

onnxruntime.set_default_logger_severity(3)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

from google.cloud import speech
from dotenv import load_dotenv
load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Constants
WHISPER_MODEL = "large-v3"

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
        transcripts = []
        for segment in result["segments"]:
            transcripts.append({
                "text": segment["text"].strip(),
                "start": segment["start"],
                "end": segment["end"],
                "confidence": segment["confidence"]
            })
        print(f"Whisper transcription complete: {len(transcripts)} segments")
        return transcripts
    except Exception as e:
        print(f"Error transcribing with Whisper: {str(e)}")
        return []

def transcribe_google_speech(client, wav_path):
    print(f"Transcribing with Google Speech-to-Text on audio: {wav_path}")
    try:
        with open(wav_path, "rb") as audio_file:
            audio_content = audio_file.read()

        audio = speech.RecognitionAudio(content=audio_content)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
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

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def verify_transcriptions(whisper_transcripts, google_transcripts, wer_threshold=0.30, confidence_threshold=0.85):
    verified = []

    # If no Google transcript is available, filter Whisper by confidence
    if not google_transcripts:
        for w in whisper_transcripts:
            conf = w.get("confidence", 0)
            if conf >= confidence_threshold:
                segment = {
                    "text":       w["text"].strip(),
                    "start":      w.get("start"),
                    "end":        w.get("end")
                }
                verified.append(segment)
                print(f"Added WHISPER_HIGH_CONFIDENCE: \"{segment['text']}\" "
                      f"({segment['start']}–{segment['end']}) conf={conf:.2f}")
            else:
                print(f"Discarded Whisper (conf {conf:.2f} < {confidence_threshold}): "
                      f"\"{w['text'].strip()}\"")
        verified.sort(key=lambda s: s.get("start", 0))
        return verified

    whisper_combined = " ".join([w.get("text", "").strip() for w in whisper_transcripts if w.get("text", "").strip()])
    norm_whisper_combined = normalize(whisper_combined)
    print(f'NORM WHISPER, {norm_whisper_combined}')
    
    google_combined = " ".join([g.get("text", "").strip() for g in google_transcripts if g.get("text", "").strip()])
    norm_google_combined = normalize(google_combined)
    print(f'NORM GOOGLE, {norm_google_combined}')
    
    overall_wer = wer(norm_google_combined, norm_whisper_combined)
    print(f"Overall WER between combined transcripts: {overall_wer:.4f}")

    if overall_wer <= wer_threshold:
        print(f"Combined transcripts are similar (WER={overall_wer:.4f} <= {wer_threshold})")
        print(f"Using all Whisper segments (verified)")
        
        for w in whisper_transcripts:
            raw_w = w.get("text", "").strip()
            if not raw_w:
                continue
            segment = {
                "text": raw_w,
                "start": w.get("start"),
                "end": w.get("end")
            }
            verified.append(segment)
            print(f"Added VERIFIED_WHISPER: \"{segment['text']}\" "
                  f"({segment['start']}–{segment['end']})")
    else:
        print(f"Combined transcripts differ significantly (WER={overall_wer:.4f} > {wer_threshold})")
        print(f"Checking individual segments")
        
        for w in whisper_transcripts:
            raw_w = w.get("text", "").strip()
            if not raw_w:
                continue
            
            conf_w = w.get("confidence", 0)
            w_start = w.get("start", 0)
            w_end = w.get("end", 0)
            
            if conf_w >= 0.60:
                segment = {
                    "text": raw_w,
                    "start": w_start,
                    "end": w_end
                }
                verified.append(segment)
                print(f"Added WHISPER (high conf): \"{segment['text']}\" "
                      f"({segment['start']}–{segment['end']}) conf={conf_w:.2f}")
            else:
                matching_g = next((g for g in google_transcripts if w_start <= g.get("start", 0) < w_end), None)
                
                if matching_g:
                    segment = {
                        "text": matching_g["text"].strip(),
                        "start": matching_g["start"],
                        "end": matching_g["end"]
                    }
                    verified.append(segment)
                    print(f"Added GOOGLE (low whisper conf): \"{segment['text']}\" "
                          f"({segment['start']}–{segment['end']}) whisper_conf={conf_w:.2f}")
                else:
                    print(f"Discarded low confidence Whisper (no matching Google segment): \"{raw_w}\" "
                          f"({w_start}–{w_end}) conf={conf_w:.2f}")

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

        # Transcribe with both models
        whisper_trans = transcribe_whisper(whisper_model, audio_path)
        google_trans = transcribe_google_speech(google_client, audio_path)
        
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