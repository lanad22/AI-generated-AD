import json
import tempfile
import subprocess
import os
import re
import argparse
from gtts import gTTS
from typing import Dict
from dotenv import load_dotenv
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import google.generativeai as genai
import openai # Added import

load_dotenv()

# Added new model constant
MODEL_QWEN = "qwen"
MODEL_GEMINI = "gemini"
MODEL_GPT = "gpt"

def get_tts_duration(text):
    if not text or text.isspace():
        return 0.0
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=True) as temp_file:
        try:
            tts = gTTS(text=text, lang='en')
            tts.save(temp_file.name)
            cmd = (f'ffprobe -v error -select_streams a:0 -show_entries format=duration '
                   f'-of csv="p=0" "{temp_file.name}"')
            duration = float(subprocess.check_output(cmd, shell=True).decode().strip())
            return duration
        except Exception as e:
            print(f"Could not get TTS duration for text '{text}': {e}")
            words = len(text.split())
            estimated_duration = words / 2.5
            return max(1.0, estimated_duration)

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
        do_sample=True
    )

    input_token_len = inputs.input_ids.shape[1]
    generated_ids = output_ids[:, input_token_len:]
    response_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    return response_text.strip()


def evaluate_clip_necessity(client, model_name, clip, transcript_data, previous_descriptions):
    scene_number = clip.get('scene_number', 0)
    scene_transcript = [t for t in transcript_data if t.get('scene_number') == scene_number]
    
    transcript_text = " ".join([segment.get('text', '') for segment in scene_transcript])
    cumulative_transcript = " ".join([seg.get('text', '') for seg in transcript_data if seg.get('scene_number') <= scene_number])
    previous_desc_text = " ".join([f"[Scene {desc.get('scene_number')}] {desc.get('type')}: {desc.get('text')}" for desc in previous_descriptions])
    
    prompt = f"""
            You are an accessibility expert selecting ONE visual description per scene to convert to audio description for blind and low-vision users.

            ### CONTEXT
            - IMPORTANT: You must select ONLY ONE description per scene - the most important one.
            - Audio descriptions interrupt the natural flow of content and should be MINIMAL.
            - The video's spoken audio (transcript) is the primary source of information.
            - Audio descriptions should be used SPARINGLY - only for truly critical visual information.

            ### INPUT
            CURRENT SCENE TRANSCRIPT:
            {transcript_text}

            CUMULATIVE TRANSCRIPT SO FAR:
            {cumulative_transcript}
            
            CUMULATIVE DESCRIPTION SO FAR:
            {previous_desc_text}

            VISUAL DESCRIPTIONS TO EVALUATE:
            {clip['text']}

            ### EVALUATION CRITERIA
            Include this visual description (necessary = true) if it meets **any** one of these essential conditions:
            - Important Visual Information: Conveys visual details not in the audio (new actions, expressions, settings).
            - Unspoken Actions & Key Events: Describes important silent actions or events (e.g. a character’s meaningful gesture, a key object movement).
            - Scene Context & Characters: Identifies who or where when audio alone is ambiguous (e.g. new character entry, location change).
            - Novelty & Variation: Introduces a distinct visual element or scene detail that has not been described before (e.g. a flowing stream, a perched butterfly).
            - Scene Changes & Time Jumps: Notes unannounced transitions (e.g. “cut to: a hospital corridor, later that night”).

            ### OUTPUT FORMAT
            Return a JSON object with the following keys:
            - "necessary": boolean (true or false).
            - "reason": A clear explanation of why this was selected or why none were necessary.
            """

    try:
        result = ""
        if model_name.startswith('gemini'):
            prompt += '\n\nIMPORTANT: Respond with ONLY the raw JSON object, without using markdown ```json ... ``` wrappers.'
            response = client.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 300,
                    "response_mime_type": "application/json",
                }
            )
            result = response.text
        elif model_name == 'local_qwen':
            result = _generate_with_qwen(client, prompt, 300, 0.7)
        else: # This block handles GPT models
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an accessibility expert evaluating audio descriptions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            result = response.choices[0].message.content.strip()
        
        print(f"MODEL RESPONSE: {result}")
        
        try:
            matches = re.search(r'\{.*\}', result, re.DOTALL)
            json_str = matches.group(0) if matches else result
            analysis = json.loads(json_str)
            return analysis.get('necessary', False), analysis.get('reason', "No reason provided")
        except json.JSONDecodeError:
            print(f"Failed to parse JSON for clip in scene {scene_number}.")
            return False, "Failed to parse model response"
            
    except Exception as e:
        print(f"Error evaluating necessity with model {model_name}: {e}")
        return False, f"Error: {str(e)}"

def optimize_description(client, model_name, clip):
    if not clip:
        return None
    
    prompt = f"""
            TASK: Create an extremely concise version of this visual description for an audio description track.

            ORIGINAL DESCRIPTION:
            {clip['text']}

            GUIDELINES:
            - Focus ONLY on the most essential visual elements.
            - Make it significantly more concise while keeping the most critical information.
            - Use natural, conversational language.
            - Use clear, vivid language suitable for audio description.
            - Maintain a flowing sentence structure.
            - Start with the most important element.
            - Be extremely concise - every word must earn its place.

            OUTPUT:
            Provide only the optimized description text, with no extra commentary or quotation marks.
            """
    
    try:
        optimized_text = ""
        if model_name.startswith('gemini'):
            response = client.generate_content(
                prompt,
                generation_config={"temperature": 1.0, "max_output_tokens": 100}
            )
            optimized_text = response.text.strip()
        elif model_name == 'local_qwen':
            optimized_text = _generate_with_qwen(client, prompt, 100, 1.0)
        else: # This block handles GPT models
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert audio describer who writes concise text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=1.0,
                max_tokens=100
            )
            optimized_text = response.choices[0].message.content.strip()
        
        optimized_text = optimized_text.strip('"')

        tts_duration = get_tts_duration(optimized_text)
        
        optimized_clip = clip.copy()
        optimized_clip['text'] = optimized_text
        optimized_clip['duration'] = tts_duration
        optimized_clip['end_time'] = clip['start_time'] + tts_duration
        optimized_clip['original_text'] = clip['text']
        
        return optimized_clip
        
    except Exception as e:
        print(f"Error optimizing clip with model {model_name}: {e}")
        return clip  

def main():
    parser = argparse.ArgumentParser(description="Analyze and optimize visual descriptions for accessibility")
    parser.add_argument("video_folder", help="Path to the video folder containing relevant JSON files")
    parser.add_argument("--model", type=str, choices=[MODEL_GEMINI, MODEL_QWEN, MODEL_GPT], default=MODEL_GPT,
                         help="Choose the model for optimizing descriptions: 'gemini', 'qwen', or 'gpt-4'.")
    parser.add_argument("--no-analyze-necessity", action="store_true", 
                        help="Skip analyzing whether descriptions are necessary (default is to analyze)")
    
    args = parser.parse_args()
    
    client = None
    model_to_use = ""
    if args.model == MODEL_GEMINI:
        try:
            model_to_use = "gemini-1.5-pro-latest"
            print(f"\nSetting up Google Gemini client for model: {model_to_use}...")
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("Error: GEMINI_API_KEY environment variable not set. This is required for Gemini models.")
                return
            genai.configure(api_key=api_key)
            client = genai.GenerativeModel(model_to_use)
        except ImportError:
            print("Error: 'google.generativeai' package not found. Please install it using 'pip install google-generativeai'")
            return
    elif args.model == MODEL_QWEN:
        model_to_use = "local_qwen"
        print(f"\nSetting up LOCAL Qwen model...")           
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-72B-Instruct", 
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2", 
            device_map="auto", 
            quantization_config=quantization_config,
            cache_dir="../.cache")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")
        client = {'model': model, 'processor': processor}
    # Added block to initialize the GPT client
    elif args.model == MODEL_GPT:
        try:
            model_to_use = "gpt-4o" # Specify your desired GPT model here
            print(f"\nSetting up OpenAI GPT client for model: {model_to_use}...")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("Error: OPENAI_API_KEY environment variable not set. This is required for GPT models.")
                return
            client = openai.OpenAI(api_key=api_key)
        except ImportError:
            print("Error: 'openai' package not found. Please install it using 'pip install openai'")
            return

    if not client:
        print("Client setup failed. Exiting.")
        return

    video_id = os.path.basename(os.path.normpath(args.video_folder))  
    scenes_folder = os.path.join(args.video_folder, f"{video_id}_scenes")
    
    # File path logic now correctly handles the 'gpt' suffix
    audio_clips_path = os.path.join(scenes_folder, f"audio_clips_optimized_{args.model}.json")
    preferred_scene_info_path = os.path.join(scenes_folder, f"scene_info_{args.model}.json")
    fallback_scene_info_path = os.path.join(scenes_folder, "scene_info.json")

    scene_info_path = ""
    if os.path.exists(preferred_scene_info_path):
        scene_info_path = preferred_scene_info_path
    elif os.path.exists(fallback_scene_info_path):
        scene_info_path = fallback_scene_info_path
    else:
        print(f"Error: No suitable scene_info file found in {scenes_folder}.")
        print(f"  - Looked for: {os.path.basename(preferred_scene_info_path)}")
        print(f"  - And: {os.path.basename(fallback_scene_info_path)}")
        return

    print(f"\nUsing scene info file: {os.path.basename(scene_info_path)}")
    print(f"Processing audio clips file: {os.path.basename(audio_clips_path)}")
    
    if not os.path.exists(scene_info_path) or not os.path.exists(audio_clips_path):
        print(f"Error: Required input files not found.")
        if not os.path.exists(scene_info_path):
            print(f"  - Missing: {scene_info_path}")
        if not os.path.exists(audio_clips_path):
            print(f"  - Missing: {audio_clips_path}")
        return
    
    with open(scene_info_path, "r") as f:
        scene_info = json.load(f)
    with open(audio_clips_path, "r") as f:
        audio_clips = json.load(f)
    
    transcript_data = []
    for scene in scene_info:
        scene_number = scene.get('scene_number', 0)
        for segment in scene.get('transcript', []):
            transcript_segment = segment.copy()
            transcript_segment['scene_number'] = scene_number
            transcript_data.append(transcript_segment)
    
    print(f"Loaded transcript with {len(transcript_data)} segments")
    print(f"Loaded {len(audio_clips)} descriptions from {os.path.basename(audio_clips_path)}")
    
    non_gap_visuals = [desc for desc in audio_clips if not desc.get('fits_in_gap', True) and desc.get('type') == 'Visual']
    print(f"\nFound {len(non_gap_visuals)} Visual descriptions where fits_in_gap is false")
    
    if not non_gap_visuals and not args.no_analyze_necessity:
        print("No Visual descriptions with fits_in_gap=false to process.")
        return

    audio_clips.sort(key=lambda x: (x.get('scene_number', 0), x.get('start_time', 0)))
    final_clips, previous_descriptions = [], []
    clips_kept, clips_removed = 0, 0
    
    for clip in audio_clips:
        is_non_gap_visual = clip.get('type') == 'Visual' and not clip.get('fits_in_gap', True)
        
        if not args.no_analyze_necessity and is_non_gap_visual:
            print(f"\n===== EVALUATING CLIP IN SCENE {clip['scene_number']} =====")
            print(f"Description: \"{clip['text']}\"")
            
            is_necessary, reason = evaluate_clip_necessity(client, model_to_use, clip, transcript_data, previous_descriptions)
            print(f"REASON: {reason}")
            
            if is_necessary:
                clips_kept += 1
                print("STATUS: Kept. Optimizing description...")
                optimized_clip = optimize_description(client, model_to_use, clip)
                print(f"Original ({len(clip['text'])} chars): {clip['text']}")
                print(f"Optimized ({len(optimized_clip['text'])} chars): {optimized_clip['text']}")
                final_clips.append(optimized_clip)
                previous_descriptions.append(optimized_clip)
            else:
                clips_removed += 1
                print("STATUS: Removed as unnecessary.")
        else:
            final_clips.append(clip)
            previous_descriptions.append(clip)

    final_clips.sort(key=lambda x: (x.get('scene_number', 0), x.get('start_time', 0)))

    with open(audio_clips_path, 'w') as f:
        json.dump(final_clips, f, indent=2)

    print(f"\nResults saved back to: {audio_clips_path}")
    print(f"Final output: {len(final_clips)} clips total")
    
    if not args.no_analyze_necessity:
        print(f"Non-gap visual descriptions evaluated: {len(non_gap_visuals)}")
        print(f"  - Kept and optimized: {clips_kept}")
        print(f"  - Removed as unnecessary: {clips_removed}")

if __name__ == "__main__":
    main()