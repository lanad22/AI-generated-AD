import json
import difflib
from jiwer import wer
import re
import subprocess
import argparse
import whisper_timestamped
import os
import onnxruntime
import multiprocessing as mp

onnxruntime.set_default_logger_severity(3)
os.environ["OMP_NUM_THREADS"] = "1"

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

import os
import subprocess

def extract_audio(scene_video_path, output_audio_path):
    if os.path.exists(output_audio_path):
        print(f"Audio already exists: {output_audio_path}, skipping extraction.")
        return

    command = [
        "ffmpeg", "-y",
        "-i", scene_video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "1",
        output_audio_path
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Extracted audio: {output_audio_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio from {scene_video_path}: {e.stderr.decode()}")


def transcribe_whisper(wav_path, device="cuda"):
    """
    Transcribe the audio using Whisper via whisper_timestamped.
    """
    print(f"Transcribing with Whisper on audio: {wav_path}")
    try:
        model = whisper_timestamped.load_model(WHISPER_MODEL, device=device)
        result = whisper_timestamped.transcribe(
            model,
            wav_path,
            vad="silero:v3.1",
            beam_size=10,
            best_of=5,
            temperature=(0.0, 0.1, 0.2, 0.4, 0.6, 0.8)
        )
        print('RAW RESPONSE WHISPER: {result}')
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

def transcribe_google_speech(wav_path):
    print(f"Transcribing with Google Speech-to-Text on audio: {wav_path}")
    try:
        client = speech.SpeechClient()

        # Load audio data
        with open(wav_path, "rb") as audio_file:
            audio_content = audio_file.read()

        audio = speech.RecognitionAudio(content=audio_content)

        # Configure speech recognition request
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            language_code="en-US",
            enable_word_time_offsets=True,
            enable_automatic_punctuation=True,
            model="video",  # Using video model for better results
        )
         # For longer audio, use long_running_recognize
        file_size = os.path.getsize(wav_path) / (1024 * 1024)  # Size in MB

        if file_size > 1:  # If file is larger than 1MB, use long-running recognition
            print("Using long-running recognition due to file size...")
            operation = client.long_running_recognize(config=config, audio=audio)
            response = operation.result(timeout=90)
        else:
            response = client.recognize(config=config, audio=audio)

        transcripts = []
        for result in response.results:
            alternative = result.alternatives[0]

            # Get start and end time from first and last word if available
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
        import traceback
        traceback.print_exc()
        return []

def normalize(text: str) -> str:
    # lowercase, trim, remove punctuation, collapse spaces
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)     # drop punctuation
    text = re.sub(r"\s+", " ", text)        # collapse whitespace
    return text

def verify_transcriptions(whisper_transcripts, google_transcripts, wer_threshold=0.30, confidence_threshold=0.75):
    """
    Workflow:
    1. If no Google transcripts:
         • Keep only Whisper segments with confidence ≥ confidence_threshold.
    
    2. If both Whisper and Google have transcripts:
         • Compare combined text from both transcriptions for overall similarity
         • If similar (WER ≤ wer_threshold):
             - Use all Whisper segments
         • If not similar:
             - For each Whisper segment:
                 - If high confidence (≥ 0.65), keep as is
                 - If low confidence (< 0.65):
                     - Find Google segment whose start time falls within this Whisper segment
                     - If found, use Google's segment
                     - If not found, discard the Whisper segment
    """
    verified = []

    # 1) No Google: only high‑confidence Whisper
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

    # 2) Both Whisper and Google have transcripts
    # Combine all whisper text
    whisper_combined = " ".join([w.get("text", "").strip() for w in whisper_transcripts if w.get("text", "").strip()])
    norm_whisper_combined = normalize(whisper_combined)
    print(f'NORM WHISPER, {norm_whisper_combined}')
    
    # Combine all google text
    google_combined = " ".join([g.get("text", "").strip() for g in google_transcripts if g.get("text", "").strip()])
    norm_google_combined = normalize(google_combined)
    print(f'NORM GOOGLE, {norm_google_combined}')
    
    # Calculate overall WER
    overall_wer = wer(norm_google_combined, norm_whisper_combined)
    print(f"Overall WER between combined transcripts: {overall_wer:.4f}")

    # If combined texts are similar enough, use all Whisper segments
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
        
        # Process each Whisper segment
        for w in whisper_transcripts:
            raw_w = w.get("text", "").strip()
            if not raw_w:
                continue
                
            conf_w = w.get("confidence", 0)
            w_start = w.get("start", 0)
            w_end = w.get("end", 0)
            
            # High confidence Whisper - keep regardless
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
                # Low confidence Whisper - find Google segment whose start time is within this segment
                matching_g = None
                
                for g in google_transcripts:
                    g_start = g.get("start", 0)
                    
                    # Check if Google segment's start time falls within Whisper segment's timespan
                    if w_start <= g_start < w_end:
                        matching_g = g
                        break
                
                # If matching Google segment found, use it
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

    # sort and return
    verified.sort(key=lambda s: s.get("start", 0))
    print(f"Verification complete: {len(verified)} segments added.")
    return verified

def should_discard_captions(global_transcript_text, global_caption_text, threshold=0.8):
    """
    Compare the global transcript and caption texts.
    Returns True if the similarity is above the threshold, indicating that captions should be discarded.
    """
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

    # Load global captions if available (expects a JSON file named <video_id>.json in video_folder)
    captions = None
    captions_path = os.path.join(video_folder, f"{video_id}.json")
    if os.path.exists(captions_path):
        try:
            with open(captions_path, "r") as f:
                cap_data = json.load(f)
                captions = cap_data.get("captions", [])
                print(f"Loaded {len(captions)} captions from {captions_path}")
        except Exception as e:
            print(f"Error loading captions: {str(e)}")

    global_transcript_text = ""
    # Process each scene to update transcripts and build global transcript text
    for scene in scenes:
        scene_path = scene.get("scene_path", "")
        if not scene_path or not os.path.exists(scene_path):
            print(f"Scene video not found for scene {scene.get('scene_number')}, skipping transcription.")
            continue

        audio_path = scene_path.replace(".mp4", ".wav")
        print(f"Extracting audio for scene {scene.get('scene_number')}")
        extract_audio(scene_path, audio_path)

        # Transcribe with both models and verify
        whisper_trans = transcribe_whisper(audio_path, device)
        print(f"WHISPER TRANS: {whisper_trans}")
        # Use Google Speech-to-Text instead of Wav2Vec2
        google_trans = transcribe_google_speech(audio_path)
        print(f"GOOGLE SPEECH TRANS: {google_trans}")
        verified_trans = verify_transcriptions(whisper_trans, google_trans)
        print(f"VERIFIED: {verified_trans}")
        scene["transcript"] = verified_trans

        # Append the scene transcript text to the global transcript text
        scene_transcript_text = " ".join([seg["text"] for seg in verified_trans])
        global_transcript_text += scene_transcript_text + " "
        '''
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            print(f"Removed temporary audio file: {temp_audio_path}")'''

    # Global captions check: compare the entire transcript with the full captions text
    if captions and global_transcript_text.strip():
        global_caption_text = " ".join([cap["text"] for cap in captions])
        if should_discard_captions(global_transcript_text, global_caption_text, global_caption_threshold):
            print("Global captions are very similar to the transcript. Discarding captions for all scenes.")
            for scene in scenes:
                scene["captions"] = []
        else:
            # If captions are not similar, match captions to each scene based on scene boundaries
            for scene in scenes:
                scene_start = scene.get("start", 0)
                scene_end = scene.get("end", 0)
                if scene_end > scene_start:
                    scene_duration = scene_end - scene_start
                    scene_captions = match_captions(scene_start, scene_end, scene_duration, captions)
                    scene["captions"] = scene_captions
                    print(f"Scene {scene.get('scene_number')}: {len(scene_captions)} captions matched.")
                else:
                    scene["captions"] = []
    else:
        # No captions available or transcript text is empty
        for scene in scenes:
            scene["captions"] = []

    with open(scene_json_path, "w") as out_f:
        json.dump(scenes, out_f, indent=2)
    print(f"Updated scene JSON with transcripts saved to: {scene_json_path}")
    
def main():
    parser = argparse.ArgumentParser(
        description="Transcribe scene audio using Whisper and Google Speech-to-Text, verify transcripts, and update scene JSON with optional captions."
    )
    parser.add_argument("video_folder", type=str,
                        help="Path to the video folder (e.g., videos/video_id). The scene_info.json file is expected at videos/video_id/video_id_scenes/scene_info.json")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for Whisper transcription (default: cuda)")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Similarity threshold for transcription verification (default: 0.6)")

    args = parser.parse_args()
    update_scene_transcripts(args.video_folder, args.device, args.threshold)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()



    