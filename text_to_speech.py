"""
Text-to-Speech conversion using Piper TTS.
Converts translated text to audio in the target language.
Memory-efficient for M1 MacBook with 8GB RAM.
"""

import os
import sys
import wave
import numpy as np
from pathlib import Path


def get_piper_voice_for_language(language: str) -> str:
    """
    Get appropriate Piper voice model for a given language.
    
    Piper uses voice models in format: language_voice.onnx
    This function maps languages to available Piper voices.
    
    Args:
        language: Language name or code (e.g., "English", "en", "Spanish", "es")
    
    Returns:
        Voice model name/path for Piper
    """
    # Map common languages to Piper voice models
    # Piper voices are typically named like: en_US-lessac-medium, es_ES-sharvard-medium, etc.
    language_lower = language.lower()
    
    voice_mapping = {
        # English voices
        "english": "en_US-lessac-medium",
        "en": "en_US-lessac-medium",
        "en-us": "en_US-lessac-medium",
        "en-gb": "en_GB-alba-medium",
        
        # Spanish voices
        "spanish": "es_ES-sharvard-medium",
        "es": "es_ES-sharvard-medium",
        "es-es": "es_ES-sharvard-medium",
        "es-mx": "es_MX-ald-medium",
        
        # French voices
        "french": "fr_FR-siwis-medium",
        "fr": "fr_FR-siwis-medium",
        "fr-fr": "fr_FR-siwis-medium",
        
        # German voices
        "german": "de_DE-thorsten-medium",
        "de": "de_DE-thorsten-medium",
        "de-de": "de_DE-thorsten-medium",
        
        # Italian
        "italian": "it_IT-riccardo-medium",
        "it": "it_IT-riccardo-medium",
        
        # Portuguese
        "portuguese": "pt_BR-edresson-medium",
        "pt": "pt_BR-edresson-medium",
        
        # Polish
        "polish": "pl_PL-darkman-medium",
        "pl": "pl_PL-darkman-medium",
        
        # Russian
        "russian": "ru_RU-ruslan-medium",
        "ru": "ru_RU-ruslan-medium",
        
        # Chinese
        "chinese": "zh_CN-huayan-medium",
        "zh": "zh_CN-huayan-medium",
        
        # Japanese
        "japanese": "ja_JP-natsume-medium",
        "ja": "ja_JP-natsume-medium",
        
        # Korean
        "korean": "ko_KR-kyungho-medium",
        "ko": "ko_KR-kyungho-medium",
    }
    
    # Try to find matching voice
    voice = voice_mapping.get(language_lower)
    
    if not voice:
        # Default to English if language not found
        print(f"⚠️  Voice for '{language}' not found, using English default")
        voice = "en_US-lessac-medium"
    
    return voice


def text_to_speech_piper(text: str, target_language: str = "English", output_file: str = "output.wav"):
    """
    Convert text to speech using Piper TTS.
    
    This function uses Piper TTS to generate audio from text.
    Piper is a fast, local neural TTS system that runs efficiently on M1 MacBooks.
    
    Args:
        text: Text to convert to speech (translated text from Step 3)
        target_language: Target language for voice selection (default: "English")
        output_file: Output WAV file path (default: "output.wav")
    
    Returns:
        Path to the generated audio file
    
    Note:
        Piper models need to be downloaded first. The function will attempt to
        download the model automatically if not found locally.
    """
    print("=" * 70)
    print("🔊 Text-to-Speech: Converting Text to Audio")
    print("=" * 70)
    print()
    
    # Step 1: Check if text is provided
    if not text or not text.strip():
        print("✗ Error: No text provided for speech synthesis")
        return None
    
    print(f"Input text: {text}")
    print(f"Target language: {target_language}")
    print()
    
    # Step 2: Get appropriate voice for the language
    voice_model = get_piper_voice_for_language(target_language)
    print(f"Selected voice: {voice_model}")
    print()
    
    # Step 3: Use Piper TTS
    try:
        from piper.voice import PiperVoice
        from piper.config import SynthesisConfig
        
        print("Loading Piper TTS model...")
        print("(Voice models need to be downloaded first)")
        print()
        
        # Step 4: Find or download voice model
        # Piper voices are typically stored in ~/.local/share/piper/voices/
        # Format: voice_name.onnx and voice_name.onnx.json
        import os
        home_dir = os.path.expanduser("~")
        voice_dir = os.path.join(home_dir, ".local", "share", "piper", "voices")
        
        # Create voice directory if it doesn't exist
        os.makedirs(voice_dir, exist_ok=True)
        
        voice_path = os.path.join(voice_dir, f"{voice_model}.onnx")
        config_path = os.path.join(voice_dir, f"{voice_model}.onnx.json")
        
        # Check if voice model exists, if not, try to download it
        if not os.path.exists(voice_path):
            print(f"Voice model '{voice_model}' not found locally")
            print("Attempting to download...")
            print()
            
            try:
                # Try to download the voice model
                from piper.download_voices import download_voice
                from pathlib import Path
                
                print(f"Downloading voice model: {voice_model}")
                print("(This may take a minute - model is ~10-50MB)")
                print()
                
                # Download voice to the voice directory (needs Path object)
                download_voice(voice_model, Path(voice_dir))
                
                # Verify download
                if os.path.exists(voice_path):
                    print("✓ Voice model downloaded successfully")
                else:
                    print("✗ Download failed - voice file not found")
                    return None
                    
            except Exception as e:
                print(f"✗ Could not download voice automatically: {e}")
                print()
                print("Please download manually:")
                print("  1. Visit: https://huggingface.co/rhasspy/piper-voices")
                print(f"  2. Download: {voice_model}.onnx and {voice_model}.onnx.json")
                print(f"  3. Place in: {voice_dir}")
                print()
                return None
        
        print(f"✓ Found voice model: {voice_path}")
        
        # Step 5: Load the voice model
        # PiperVoice.load() loads the ONNX model (memory-efficient for M1)
        if config_path:
            voice = PiperVoice.load(voice_path, config_path=config_path)
        else:
            # Try loading without explicit config (Piper may find it automatically)
            voice = PiperVoice.load(voice_path)
        
        print("✓ Voice model loaded")
        print(f"  Model: {voice_model}")
        print(f"  Sample rate: {voice.config.sample_rate} Hz")
        print()
        
        # Step 6: Generate speech from text
        # Piper uses synthesize() method which returns an iterable of AudioChunk objects
        print("Generating speech from text...")
        
        # Synthesize text to audio chunks
        # synthesize() returns an iterable of AudioChunk objects
        audio_chunks = voice.synthesize(text)
        
        # Step 7: Collect audio data from chunks
        # Process in chunks to be memory-efficient
        audio_data = []
        sample_rate = voice.config.sample_rate
        
        for audio_chunk in audio_chunks:
            # Piper returns AudioChunk objects
            # AudioChunk has 'audio_int16_array' attribute which is a numpy array (int16)
            if hasattr(audio_chunk, 'audio_int16_array') and audio_chunk.audio_int16_array is not None:
                # Direct access to int16 numpy array
                audio_array = audio_chunk.audio_int16_array
                audio_data.append(audio_array)
            elif hasattr(audio_chunk, 'audio_int16_bytes') and audio_chunk.audio_int16_bytes is not None:
                # If audio_int16_bytes exists, convert to numpy array
                audio_bytes = audio_chunk.audio_int16_bytes
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_data.append(audio_array)
            elif hasattr(audio_chunk, 'audio_float_array'):
                # Convert float array to int16
                audio_float = audio_chunk.audio_float_array
                audio_array = (audio_float * 32767).astype(np.int16)
                audio_data.append(audio_array)
        
        # Step 8: Concatenate all audio chunks into single array
        if audio_data:
            full_audio = np.concatenate(audio_data)
            
            # Step 9: Save to WAV file
            # WAV format: 16-bit PCM, mono, at model's sample rate
            print("Saving audio file...")
            with wave.open(output_file, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono audio
                wav_file.setsampwidth(2)  # 16-bit (2 bytes per sample)
                wav_file.setframerate(sample_rate)  # Sample rate from model
                wav_file.writeframes(full_audio.tobytes())  # Write audio data
            
            # Calculate duration
            duration = len(full_audio) / sample_rate
            
            print(f"✓ Audio saved to: {output_file}")
            print(f"  Sample rate: {sample_rate} Hz")
            print(f"  Duration: {duration:.2f} seconds")
            print(f"  Format: 16-bit PCM, Mono")
            print()
            
            return output_file
        else:
            print("✗ No audio data generated")
            return None
            
    except ImportError as e:
        print(f"✗ Error: Piper TTS package not found: {e}")
        print()
        print("Please install piper:")
        print("  pip install piper-tts")
        print()
        return None
    except Exception as e:
        print(f"✗ Error during text-to-speech: {e}")
        import traceback
        traceback.print_exc()
        return None
            
    except Exception as e:
        print(f"✗ Error during text-to-speech conversion: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function to run text-to-speech conversion."""
    print("=" * 70)
    print("🔊 Text-to-Speech Conversion")
    print("=" * 70)
    print()
    
    # Get text input
    if len(sys.argv) > 1:
        text = sys.argv[1]
    else:
        text = input("Enter text to convert to speech: ").strip()
        if not text:
            print("✗ No text provided")
            return
    
    # Get target language
    if len(sys.argv) > 2:
        target_language = sys.argv[2]
    else:
        target_language = input("Target language (default: English): ").strip() or "English"
    
    # Get output file
    output_file = "output.wav"
    if len(sys.argv) > 3:
        output_file = sys.argv[3]
    
    print()
    
    # Convert to speech
    result = text_to_speech_piper(text, target_language, output_file)
    
    if result:
        print("=" * 70)
        print("✅ Text-to-Speech Complete!")
        print("=" * 70)
        print()
        print(f"✓ Audio file: {result}")
        print("  You can now play this file to hear the speech!")
    else:
        print()
        print("✗ Text-to-speech conversion failed")


if __name__ == "__main__":
    main()
