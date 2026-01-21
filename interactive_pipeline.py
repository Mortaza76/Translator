"""
Interactive Push-to-Talk Pipeline: Press Enter to Record → Process → Play
Non-real-time mode: Record when you want, then process the entire recording.
"""

import os
import sys
import threading
import sounddevice as sd
import numpy as np
import wave
from typing import Optional

from record_audio import AudioRecorder
from speech_to_text import transcribe_urdu_audio
from translate_text import translate_urdu_to_target
from text_to_speech import text_to_speech_piper
from play_audio import play_audio_file
from text_cleaner import clean_for_translation, is_valid_transcription


def interactive_record_until_enter(recorder: AudioRecorder, output_file: str = "input.wav") -> Optional[str]:
    """
    Record audio with push-to-talk: Press Enter to start, then Enter to stop.
    Uses high-quality 48 kHz recording with audio processing.
    
    Args:
        recorder: AudioRecorder instance (not used directly, but kept for compatibility)
        output_file: Path to save the recorded audio (16 kHz version for Whisper)
    
    Returns:
        Path to saved audio file, or None if recording failed
    """
    import threading
    
    print("\n" + "=" * 70)
    print("🎤 READY TO RECORD")
    print("=" * 70)
    print()
    print("Using high-quality recording (48 kHz → processed → 16 kHz for Whisper)")
    print()
    print("Press Enter to START recording...")
    input()  # Wait for Enter to start
    
    # Find best microphone
    devices = sd.query_devices()
    airpods_device = None
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            name_lower = device['name'].lower()
            if 'airpod' in name_lower or 'bluetooth' in name_lower:
                airpods_device = i
                break
    
    if airpods_device is None:
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                if 'macbook' in device['name'].lower() and 'microphone' in device['name'].lower():
                    airpods_device = i
                    break
        
        if airpods_device is None:
            airpods_device = sd.default.device[0]
    
    print()
    print("=" * 70)
    print("🔴 RECORDING NOW - Speak in Urdu!")
    print("=" * 70)
    print()
    print("Press Enter again to STOP recording...")
    
    # Record at high sample rate (48 kHz)
    sample_rate = 48000
    recording_data = []
    stop_event = threading.Event()  # Use Event instead of boolean for better thread safety
    
    def audio_callback(indata, frames, time_info, status):
        """Callback to capture audio."""
        if status:
            print(f"Audio status: {status}")
        if not stop_event.is_set():  # Only append if not stopped
            recording_data.append(indata.copy())
    
    def stop_on_enter():
        """Wait for Enter key to stop recording."""
        try:
            import sys
            # Flush stdin to ensure it's ready
            sys.stdin.flush()
            # Read a line (waits for Enter)
            line = sys.stdin.readline()
            stop_event.set()  # Set the event to signal stop
            print("\n✓ Stop signal received - stopping recording...")
        except Exception as e:
            print(f"\n⚠️  Error reading input: {e}")
            stop_event.set()
    
    print("(Press Enter in this terminal to stop)")
    print()
    
    # Start the stop thread BEFORE opening the stream
    stop_thread = threading.Thread(target=stop_on_enter, daemon=False)  # Not daemon so it can read stdin
    stop_thread.start()
    
    # Give thread a moment to start and be ready
    import time
    time.sleep(0.2)
    
    stream = None
    try:
        # Record at high sample rate
        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype='float32',
            device=airpods_device,
            callback=audio_callback,
            blocksize=int(sample_rate * 0.1)
        )
        
        stream.start()
        print("✓ Recording started - speak now!")
        
        # Keep recording until Enter is pressed (check stop_event)
        loop_count = 0
        while not stop_event.is_set():
            time.sleep(0.1)  # Check every 100ms
            loop_count += 1
            # Safety check - if loop runs too long, something's wrong
            if loop_count > 600:  # 60 seconds max
                print("\n⚠️  Recording timeout - stopping automatically")
                stop_event.set()
                break
        
        print("\nStopping stream...")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        stop_event.set()
    except Exception as e:
        print(f"\n✗ Error during recording: {e}")
        import traceback
        traceback.print_exc()
        stop_event.set()
    finally:
        # Always stop the stream, even if there was an error
        if stream is not None:
            try:
                stream.stop()
                stream.close()
                print("✓ Stream stopped")
            except:
                pass
        
        # Small buffer to finish
        time.sleep(0.1)
    
    # Process audio after recording stops
    if not recording_data:
        print("✗ No audio data recorded")
        return None
    
    try:
        # Concatenate and process audio
        audio = np.concatenate(recording_data, axis=0).flatten()
        
        print("Processing audio for better quality...")
        
        # Process audio (DC offset removal, filtering, normalization)
        audio = audio - np.mean(audio)  # Remove DC offset
        
        from scipy import signal
        nyquist = sample_rate / 2
        b, a = signal.butter(2, 80 / nyquist, btype='high')
        audio = signal.filtfilt(b, a, audio)
        
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio * (0.9 / max_val)  # Normalize to 90%
        
        # Resample to 16 kHz for Whisper
        num_samples_16k = int(len(audio) * 16000 / sample_rate)
        audio_16k = signal.resample(audio, num_samples_16k)
        audio_16k_int16 = (audio_16k * 32767).astype(np.int16)
        
        # Save 16 kHz version (for Whisper)
        with wave.open(output_file, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(audio_16k_int16.tobytes())
        
        print(f"✓ Recording saved: {output_file} (16 kHz, processed)")
        return output_file
        
    except Exception as e:
        print(f"✗ Error processing audio: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_recording(
    audio_file: str,
    target_language: str = "English"
) -> Optional[str]:
    """
    Process a recorded audio file: Transcribe → Translate → TTS → Play.
    
    Args:
        audio_file: Path to the audio file
        target_language: Target language for translation
    
    Returns:
        Path to output audio file, or None if processing failed
    """
    print()
    print("=" * 70)
    print("🔄 PROCESSING RECORDING")
    print("=" * 70)
    print()
    
    # Step 1: Transcribe
    print("Step 1: Transcribing Urdu speech to text...")
    print("-" * 70)
    raw_urdu_text = transcribe_audio(audio_file)
    
    if not raw_urdu_text:
        print("✗ Transcription failed")
        return None
    # I have no idea how this is working but it is working, do not change anything do not try to add modularity inside this code
    # it will break the system architecture and it will take a lot of time to fix it, so PLEASE DO NOT CHAMGE ANYTHING HERE.
    
    # Clean transcription to remove artifacts
    print("Cleaning transcription...")
    urdu_text = clean_for_translation(raw_urdu_text)
    
    if not is_valid_transcription(urdu_text):
        print("⚠️  Transcription appears to be noise/garbled - skipping")
        print(f"   Raw output: {raw_urdu_text[:100]}...")
        return None
    
    if urdu_text != raw_urdu_text:
        print(f"   Original: {raw_urdu_text[:60]}...")
        print(f"   Cleaned:  {urdu_text[:60]}...")
    
    print()
    
    # Step 2: Translate
    print(f"Step 2: Translating to {target_language}...")
    print("-" * 70)
    translated_text = translate_text(urdu_text, target_language)
    
    if not translated_text:
        print("✗ Translation failed")
        return None
    
    print()
    
    # Step 3: Text-to-Speech
    print(f"Step 3: Converting to speech...")
    print("-" * 70)
    output_file = synthesize_speech(translated_text, target_language, "output.wav")
    
    if not output_file:
        print("✗ Text-to-speech failed")
        return None
    
    print()
    
    # Step 4: Play audio
    print("Step 4: Playing translated audio...")
    print("-" * 70)
    play_audio(output_file, blocking=True)
    
    print()
    print("=" * 70)
    print("✅ PROCESSING COMPLETE")
    print("=" * 70)
    print()
    print("Results:")
    print(f"  Urdu Text: {urdu_text}")
    print(f"  {target_language} Translation: {translated_text}")
    print(f"  Audio Output: {output_file}")
    print()
    
    return output_file


def run_interactive_pipeline(
    target_language: str = "English",
    sample_rate: int = 16000
):
    """
    Run interactive push-to-talk pipeline in a loop.
    
    Workflow:
    1. Press Enter to start recording
    2. Speak in Urdu
    3. Press Enter to stop recording
    4. Process: Transcribe → Translate → TTS → Play
    5. Repeat from step 1
    
    Press Ctrl+C to exit.
    
    Args:
        target_language: Target language for translation (default: "English")
        sample_rate: Audio sample rate in Hz (default: 16000)
    """
    print("=" * 70)
    print("🎙️  INTERACTIVE PUSH-TO-TALK PIPELINE")
    print("=" * 70)
    print()
    print("How it works:")
    print("  1. Press Enter to START recording")
    print("  2. Speak in Urdu")
    print("  3. Press Enter to STOP recording")
    print("  4. Wait for processing (Transcribe → Translate → TTS)")
    print("  5. Listen to the translated audio")
    print("  6. Repeat from step 1")
    print()
    print("Press Ctrl+C to exit")
    print()
    print(f"Target Language: {target_language}")
    print(f"Transcription Model: whisper-small-urdu (fine-tuned for Urdu, GPU accelerated)")
    print()
    print("=" * 70)
    print()
    
    # Initialize audio recorder
    # Find best microphone (AirPods/Bluetooth, then MacBook mic)
    recorder = AudioRecorder(
        sample_rate=sample_rate,
        channels=1,
        dtype='int16'
    )
    
    # Find and use best microphone
    device = recorder._find_best_microphone()
    if device is not None:
        recorder.device = device
        devices = sd.query_devices()
        print(f"✓ Using microphone: {devices[device]['name']}")
        print()
    
    cycle = 1
    
    try:
        while True:
            print()
            print("=" * 70)
            print(f"📝 CYCLE {cycle}")
            print("=" * 70)
            print()
            
            # Record audio
            audio_file = interactive_record_until_enter(recorder, "input.wav")
            
            if not audio_file or not os.path.exists(audio_file):
                print("⚠️  No audio recorded, skipping processing...")
                print()
                continue
            
            # Process recording
            output_file = process_recording(
                audio_file,
                target_language=target_language
            )
            
            if output_file:
                print(f"✓ Cycle {cycle} completed successfully!")
            else:
                print(f"⚠️  Cycle {cycle} had processing errors")
            
            cycle += 1
            print()
            print("Ready for next recording...")
            print()
    
    except KeyboardInterrupt:
        print()
        print()
        print("=" * 70)
        print("🛑 PIPELINE STOPPED")
        print("=" * 70)
        print()
        print("Thank you for using the Interactive Pipeline!")
        print()


# Helper functions (wrappers for cleaner code)
def transcribe_audio(audio_file: str = "input.wav"):
    """Transcribe audio to Urdu text using fine-tuned whisper-small-urdu model."""
    return transcribe_urdu_audio(audio_file)


def translate_text(urdu_text: str, target_language: str = "English"):
    """Translate Urdu text to target language."""
    return translate_urdu_to_target(urdu_text, target_language)


def synthesize_speech(text: str, target_language: str = "English", output_file: str = "output.wav"):
    """Synthesize text to speech."""
    return text_to_speech_piper(text, target_language, output_file)


def play_audio(audio_file: str = "output.wav", blocking: bool = True):
    """Play audio file."""
    return play_audio_file(audio_file, blocking=blocking)


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Interactive Push-to-Talk Urdu Speech Translation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start interactive pipeline with default settings
  python3 interactive_pipeline.py
  
  # Translate to Spanish
  python3 interactive_pipeline.py --target Spanish
  
  # Translate to French
  python3 interactive_pipeline.py --target French
        """
    )
    
    parser.add_argument(
        "--target", "-t",
        type=str,
        default="English",
        help="Target language for translation and TTS (default: English)"
    )
    
    parser.add_argument(
        "--sample-rate", "-r",
        type=int,
        default=16000,
        help="Audio sample rate in Hz (default: 16000)"
    )
    
    args = parser.parse_args()
    
    # Run interactive pipeline (always uses fine-tuned whisper-small-urdu model)
    run_interactive_pipeline(
        target_language=args.target,
        sample_rate=args.sample_rate
    )


if __name__ == "__main__":
    main()
