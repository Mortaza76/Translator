"""
Speech-to-Text conversion using fine-tuned whisper-small-urdu model.
Converts recorded audio (input.wav) to Urdu text.
Uses GPU acceleration (MPS) on Apple Silicon for faster processing.
Memory-efficient for M1 MacBook with 8GB RAM.
"""

import os
import sys


def transcribe_urdu_audio(audio_file: str = "input.wav", chunk_duration: int = 30):
    """
    Transcribe Urdu audio to text using fine-tuned whisper-small-urdu model.
    
    This uses the transformers library with a model fine-tuned specifically for Urdu,
    providing better accuracy than standard Whisper models. Uses GPU acceleration
    (MPS) on Apple Silicon for faster processing.
    
    Args:
        audio_file: Path to the audio file (default: input.wav)
        chunk_duration: Duration in seconds per chunk for long audio (default: 30)
    
    Returns:
        Transcribed Urdu text
    """
    try:
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        import soundfile as sf
        import numpy as np
    except ImportError as e:
        print(f"✗ Error: Required packages not installed: {e}")
        print()
        print("Please install:")
        print("  pip install torch transformers soundfile")
        return None
    
    if not os.path.exists(audio_file):
        print(f"✗ Error: Audio file '{audio_file}' not found.")
        print(f"   Please record audio first using: python3 record_clean_voice.py")
        return None
    
    print("=" * 70)
    print("🎤 Speech-to-Text: Urdu Audio Transcription")
    print("=" * 70)
    print()
    print("Using: whisper-small-urdu (fine-tuned for Urdu)")
    print("GPU acceleration: MPS (Apple Silicon)")
    print()
    
    # Model settings
    model_name = "khawajaaliarshad/whisper-small-urdu"
    
    # Load model
    print("Loading fine-tuned Urdu model...")
    print("(This may take a moment on first run - model downloads automatically)")
    print()
    
    try:
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        
        # Use MPS (Apple GPU) if available, otherwise CPU
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model.to(device)
        
        if device == "mps":
            print(f"✓ Model loaded on Apple GPU (MPS)")
        else:
            print(f"✓ Model loaded on CPU")
        print()
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Load audio
    print(f"Loading audio file: {audio_file}")
    try:
        audio_input, samplerate = sf.read(audio_file)
        
        # Convert to mono if stereo
        if audio_input.ndim > 1:
            audio_input = np.mean(audio_input, axis=1)
        
        print(f"✓ Audio loaded: {len(audio_input)/samplerate:.2f} seconds at {samplerate} Hz")
        print()
        
    except Exception as e:
        print(f"✗ Error loading audio: {e}")
        return None
    
    # Split audio into chunks for long audio
    chunk_size = chunk_duration * samplerate
    chunks = [audio_input[i:i+chunk_size] for i in range(0, len(audio_input), chunk_size)]
    
    print(f"Processing {len(chunks)} chunk(s)...")
    print()
    
    # Transcribe each chunk
    transcriptions = []
    for i, chunk in enumerate(chunks):
        try:
            print(f"Transcribing chunk {i+1}/{len(chunks)}...")
            
            inputs = processor(chunk, sampling_rate=samplerate, return_tensors="pt").to(device)
            
            with torch.no_grad():
                # Generate transcription
                # Remove forced_decoder_ids from generation config to avoid conflict
                generation_config = model.generation_config
                if hasattr(generation_config, 'forced_decoder_ids'):
                    generation_config.forced_decoder_ids = None
                
                predicted_ids = model.generate(
                    inputs["input_features"],
                    generation_config=generation_config
                )
                text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                transcriptions.append(text)
            
            print(f"✓ Chunk {i+1} transcribed")
            
        except Exception as e:
            print(f"✗ Error transcribing chunk {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Combine results
    if not transcriptions:
        print("✗ No transcriptions generated")
        return None
    
    full_transcription = " ".join(transcriptions).strip()
    
    print()
    print("=" * 70)
    print("✅ Transcription Complete!")
    print("=" * 70)
    print()
    print("Transcribed Urdu Text:")
    print("-" * 70)
    print(full_transcription)
    print("-" * 70)
    print()
    
    return full_transcription


def main():
    """Main function to run speech-to-text conversion."""
    # Check command line arguments for custom audio file
    audio_file = "input.wav"
    
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    
    # Run transcription (always uses fine-tuned model)
    transcribed_text = transcribe_urdu_audio(audio_file)
    
    if transcribed_text:
        print()
        print("=" * 70)
        print("💡 Next step: Translate this Urdu text to English")
        print("=" * 70)
        return transcribed_text
    else:
        return None


if __name__ == "__main__":
    main()
