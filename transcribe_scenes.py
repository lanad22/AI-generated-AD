import json
import difflib
import subprocess
import argparse
import whisper_timestamped
import os
import onnxruntime
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

def extract_audio(scene_video_path, output_audio_path):
    """
    Extract audio from a video file (scene) using ffmpeg.
    Audio is extracted as a WAV file (mono, 16kHz).
    """
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
        transcripts = []
        for segment in result["segments"]:
            transcripts.append({
                "text": segment["text"].strip(),
                "start": segment["start"],
                "end": segment["end"]
            })
        print(f"Whisper transcription complete: {len(transcripts)} segments")
        return transcripts
    except Exception as e:
        print(f"Error transcribing with Whisper: {str(e)}")
        return []

def transcribe_google_speech(wav_path):
    """
    Transcribe the given WAV file using Google Speech-to-Text API.
    """
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
def verify_transcriptions(whisper_transcripts, google_transcripts):
    """
    Verify Whisper transcripts using Google Speech-to-Text output.
    """
    if not google_transcripts:
        print("No Google Speech-to-Text segments found, discarding all Whisper segments")
        return []

    # Check if any Google segment has empty text
    for google_seg in google_transcripts:
        if not google_seg["text"]:
            print("Empty text found in Google segments, discarding all Whisper segments")
            return []

    verified = []

    # add Whisper segments with Google overlap
    for w_seg in whisper_transcripts:
        has_overlap = any(
            w_seg["start"] < g_seg["end"] and w_seg["end"] > g_seg["start"]
            for g_seg in google_transcripts
        )
        if has_overlap:
            verified.append(w_seg)

    # Track which Google segments were used to verify Whisper segments
    google_used = [False] * len(google_transcripts)

    for i, g_seg in enumerate(google_transcripts):
        for w_seg in whisper_transcripts:
            if (w_seg["start"] < g_seg["end"] and w_seg["end"] > g_seg["start"]):
                google_used[i] = True
                break

    # Add high-confidence Google segments that don't overlap with any Whisper segment
    high_confidence_count = 0
    for i, g_seg in enumerate(google_transcripts):
        if not google_used[i] and g_seg.get("confidence", 0) > 0.9:  # 90% confidence threshold
            high_confidence_segment = {
                "text": g_seg["text"],
                "start": g_seg["start"],
                "end": g_seg["end"],
            }
            verified.append(high_confidence_segment)
            high_confidence_count += 1

    # If no Whisper segments but there are high-confidence Google segments
    for g_seg in google_transcripts:
        if g_seg.get("confidence", 0) > 0.9:  # 90% confidence threshold
            high_confidence_segment = {
                "text": g_seg["text"],
                "start": g_seg["start"],
                "end": g_seg["end"],
            }
            verified.append(high_confidence_segment)
            high_confidence_count += 1

    # Sort verified segments by start time
    verified.sort(key=lambda x: x["start"])

    print(f"Verification complete: {len(verified)} segments total")
    return verified

def should_discard_captions(global_transcript_text, global_caption_text, threshold=0.8):
    """
    Compare the global transcript and caption texts.
    Returns True if the similarity is above the threshold, indicating that captions should be discarded.
    """
    similarity = difflib.SequenceMatcher(None, global_transcript_text, global_caption_text).ratio()
    print(f"Global transcript vs captions similarity: {similarity:.2f}")
    return similarity >= threshold

def update_scene_transcripts(video_folder, device="cuda", verification_threshold=0.6, global_caption_threshold=0.8):
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

        temp_audio_path = scene_path.replace(".mp4", ".wav")
        print(f"Extracting audio for scene {scene.get('scene_number')}")
        extract_audio(scene_path, temp_audio_path)

        # Transcribe with both models and verify
        whisper_trans = transcribe_whisper(temp_audio_path, device)
        print(f"WHISPER TRANS: {whisper_trans}")
        # Use Google Speech-to-Text instead of Wav2Vec2
        google_trans = transcribe_google_speech(temp_audio_path)
        print(f"GOOGLE SPEECH TRANS: {google_trans}")
        verified_trans = verify_transcriptions(whisper_trans, google_trans)
        scene["transcript"] = verified_trans

        # Append the scene transcript text to the global transcript text
        scene_transcript_text = " ".join([seg["text"] for seg in verified_trans])
        global_transcript_text += scene_transcript_text + " "

        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            print(f"Removed temporary audio file: {temp_audio_path}")

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
    
if __name__ == "__main__":
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
                
    





    