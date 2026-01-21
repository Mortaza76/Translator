"""
Multi-language Speech-to-Text conversion.
Supports Urdu (fine-tuned model) and other languages (standard Whisper).
"""

import os
import sys


def transcribe_audio_multi(audio_file: str, source_language: str = "urdu"):
    """
    Transcribe audio to text for multiple languages.
    
    For Urdu: Uses fine-tuned whisper-small-urdu model
    For other languages: Uses standard Whisper model (multilingual)
    
    Args:
        audio_file: Path to the audio file
        source_language: Source language ("urdu", "french", "english", etc.)
    
    Returns:
        Transcribed text in the source language
    """
    try:
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        import soundfile as sf
        import numpy as np
    except ImportError as e:
        print(f"✗ Error: Required packages not installed: {e}")
        return None
    
    if not os.path.exists(audio_file):
        print(f"✗ Error: Audio file '{audio_file}' not found.")
        return None
    
    # Load audio
    try:
        audio_input, samplerate = sf.read(audio_file)
        
        # Convert to mono if stereo
        if audio_input.ndim > 1:
            audio_input = np.mean(audio_input, axis=1)
        
    except Exception as e:
        print(f"✗ Error loading audio: {e}")
        return None
    
    # Choose model based on language
    if source_language.lower() in ["urdu", "ur"]:
        # Use fine-tuned Urdu model
        model_name = "khawajaaliarshad/whisper-small-urdu"
    else:
        # Use standard multilingual Whisper model
        model_name = "openai/whisper-small"
    
    try:
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        
        # Use MPS (Apple GPU) if available, otherwise CPU
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model.to(device)
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None
    
    # Transcribe
    try:
        inputs = processor(audio_input, sampling_rate=samplerate, return_tensors="pt").to(device)
        
        with torch.no_grad():
            # Set language for multilingual model
            if source_language.lower() not in ["urdu", "ur"]:
                # Map language names to Whisper language codes
                # Handle both full names and common variations
                lang_map = {
                    "french": "fr",
                    "français": "fr",
                    "english": "en",
                    "spanish": "es",
                    "español": "es",
                    "german": "de",
                    "deutsch": "de",
                    "italian": "it",
                    "italiano": "it",
                    "portuguese": "pt",
                    "português": "pt",
                    "japanese": "ja",
                    "日本語": "ja",
                    "korean": "ko",
                    "한국어": "ko",
                    "chinese": "zh",
                    "中文": "zh",
                    "arabic": "ar",
                    "العربية": "ar",
                    "hindi": "hi",
                    "हिन्दी": "hi"
                }
                lang_code = lang_map.get(source_language.lower(), "en")
                # Force decoder to use specific language
                forced_decoder_ids = processor.get_decoder_prompt_ids(language=lang_code, task="transcribe")
            else:
                forced_decoder_ids = None
            
            if forced_decoder_ids:
                predicted_ids = model.generate(
                    inputs["input_features"],
                    forced_decoder_ids=forced_decoder_ids
                )
            else:
                predicted_ids = model.generate(inputs["input_features"])
            
            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return text.strip()
        
    except Exception as e:
        print(f"✗ Error during transcription: {e}")
        import traceback
        traceback.print_exc()
        return None
