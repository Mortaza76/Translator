"""
Complete End-to-End Pipeline: Urdu Speech → Any Language Audio
Combines all 5 steps into a seamless workflow with sentence-level buffering.

Pipeline Flow:
1. Record audio in Urdu from microphone
2. Convert Urdu speech to text with Whisper (tiny/base)
3. Translate text to user-selected target language with TranslateGemma
4. Convert translated text to audio with Piper TTS (sentence-level buffering)
5. Play the audio output

Optimized for M1 MacBook with 8GB RAM.
"""

import os
import sys
import re
import time
from typing import Optional, Tuple

# Import modular functions from each step
from record_clean_voice import record_clean_voice
from speech_to_text import transcribe_urdu_audio
from translate_text import translate_urdu_to_target
from text_to_speech import text_to_speech_piper
from play_audio import play_audio_file
from text_cleaner import clean_for_translation, is_valid_transcription


# ============================================================================
# MODULAR FUNCTIONS FOR EACH STEP
# ============================================================================

def record_audio(output_file: str = "input.wav", duration: Optional[float] = None) -> bool:
    """
    Step 1: Record audio in Urdu from microphone.
    
    This function records audio from the microphone (prioritizing AirPods/Bluetooth,
    then MacBook mic) and saves it as a WAV file. Uses 48 kHz sample rate for high
    quality (similar to iMessage).
    
    Args:
        output_file: Path to save the recorded audio (default: "input.wav")
        duration: Recording duration in seconds. If None, uses interactive mode.
    
    Returns:
        True if recording successful, False otherwise
    
    Example:
        >>> record_audio("input.wav", duration=5.0)
        Recording audio for 5.0 seconds...
        ✓ Recording saved to input.wav
    """
    print("=" * 70)
    print("STEP 1: Recording Audio (Urdu)")
    print("=" * 70)
    print()
    
    try:
        # Use the record_clean_voice function which handles device selection
        # and high-quality recording (48 kHz)
        if duration:
            print(f"Recording for {duration} seconds...")
            # For fixed duration, we'd need to modify record_clean_voice
            # For now, use interactive mode
            print("Using interactive recording mode...")
        
        # Call the recording function
        # Note: record_clean_voice() saves to "input.wav" by default
        # We'll handle custom output_file if needed
        record_clean_voice()
        
        # Check if file was created
        if os.path.exists("input.wav"):
            # If custom output file requested, copy it
            if output_file != "input.wav" and os.path.exists("input.wav"):
                import shutil
                shutil.copy("input.wav", output_file)
                print(f"✓ Recording saved to: {output_file}")
            else:
                print(f"✓ Recording saved to: input.wav")
            print()
            return True
        else:
            print("✗ Recording failed - no audio file created")
            return False
            
    except KeyboardInterrupt:
        print("\n✗ Recording cancelled by user")
        return False
    except Exception as e:
        print(f"✗ Error during recording: {e}")
        import traceback
        traceback.print_exc()
        return False


def transcribe_audio(audio_file: str = "input.wav") -> Optional[str]:
    """
    Step 2: Convert Urdu speech to text with fine-tuned whisper-small-urdu model.
    
    This function uses the fine-tuned whisper-small-urdu model to transcribe Urdu audio to text.
    The model is fine-tuned specifically for Urdu and uses GPU acceleration (MPS) on Apple Silicon.
    
    Args:
        audio_file: Path to the audio file to transcribe (default: "input.wav")
    
    Returns:
        Transcribed Urdu text, or None if transcription failed
    
    Example:
        >>> urdu_text = transcribe_audio("input.wav")
        Loading fine-tuned Urdu model...
        ✓ Transcription: "آپ کیسے ہیں؟"
    """
    print("=" * 70)
    print("STEP 2: Speech-to-Text (Urdu Audio → Urdu Text)")
    print("=" * 70)
    print()
    
    # Check if audio file exists
    if not os.path.exists(audio_file):
        print(f"✗ Error: Audio file '{audio_file}' not found")
        print(f"   Please record audio first using Step 1")
        return None
    
    try:
        # Use the modular transcription function (always uses fine-tuned model)
        raw_urdu_text = transcribe_urdu_audio(audio_file)
        
        if not raw_urdu_text:
            print("✗ Transcription failed")
            return None
        
        # Clean transcription to remove artifacts
        urdu_text = clean_for_translation(raw_urdu_text)
        
        if not is_valid_transcription(urdu_text):
            print("⚠️  Transcription appears to be noise/garbled")
            return None
        
        print(f"✓ Transcription successful")
        print()
        return urdu_text
            
    except Exception as e:
        print(f"✗ Error during transcription: {e}")
        import traceback
        traceback.print_exc()
        return None


def translate_text(urdu_text: str, target_language: str = "English") -> Optional[str]:
    """
    Step 3: Translate Urdu text to target language with TranslateGemma.
    
    This function uses the TranslateGemma model via Ollama to translate Urdu text
    to any target language. Supports dynamic language selection.
    
    Args:
        urdu_text: Urdu text to translate (from Step 2)
        target_language: Target language name, code, or locale (default: "English")
                        Examples: "English", "en", "en-US", "Spanish", "es", "es-ES"
    
    Returns:
        Translated text in target language, or None if translation failed
    
    Example:
        >>> translated = translate_text("آپ کیسے ہیں؟", "English")
        Translating from Urdu (ur-PK) to English...
        ✓ Translation: "How are you?"
    """
    print("=" * 70)
    print(f"STEP 3: Translation (Urdu Text → {target_language})")
    print("=" * 70)
    print()
    
    if not urdu_text or not urdu_text.strip():
        print("✗ Error: No Urdu text provided for translation")
        return None
    
    try:
        # Use the modular translation function
        translated_text = translate_urdu_to_target(urdu_text, target_language)
        
        if translated_text:
            print(f"✓ Translation successful")
            print()
            return translated_text
        else:
            print("✗ Translation failed")
            return None
            
    except Exception as e:
        print(f"✗ Error during translation: {e}")
        import traceback
        traceback.print_exc()
        return None


def synthesize_speech(text: str, target_language: str = "English", 
                     output_file: str = "output.wav",
                     sentence_buffering: bool = True) -> Optional[str]:
    """
    Step 4: Convert translated text to audio with Piper TTS.
    
    This function uses Piper TTS to generate audio from text. Supports sentence-level
    buffering for smoother playback and better memory efficiency.
    
    Args:
        text: Translated text to convert to speech (from Step 3)
        target_language: Target language for voice selection (default: "English")
        output_file: Output WAV file path (default: "output.wav")
        sentence_buffering: If True, process text sentence-by-sentence for smoother TTS
    
    Returns:
        Path to generated audio file, or None if synthesis failed
    
    Example:
        >>> audio_file = synthesize_speech("How are you?", "English", sentence_buffering=True)
        Processing 1 sentence(s)...
        ✓ Audio saved to: output.wav
    """
    print("=" * 70)
    print(f"STEP 4: Text-to-Speech ({target_language} Text → Audio)")
    print("=" * 70)
    print()
    
    if not text or not text.strip():
        print("✗ Error: No text provided for speech synthesis")
        return None
    
    try:
        if sentence_buffering:
            # Step 4a: Split text into sentences for smoother processing
            # This helps with memory efficiency and can enable streaming playback
            sentences = split_into_sentences(text)
            print(f"Processing {len(sentences)} sentence(s) with sentence-level buffering...")
            print()
            
            # For now, we'll synthesize all sentences and concatenate
            # In a more advanced implementation, we could stream each sentence
            # as it's generated for even smoother playback
            all_audio_chunks = []
            
            for i, sentence in enumerate(sentences, 1):
                if not sentence or not sentence.strip():
                    continue
                
                sentence_preview = sentence.strip()[:50] if len(sentence.strip()) > 50 else sentence.strip()
                print(f"  Synthesizing sentence {i}/{len(sentences)}: {sentence_preview}...")
                
                # Generate temporary file for this sentence
                temp_file = f"temp_sentence_{i}.wav"
                sentence_audio = text_to_speech_piper(
                    sentence.strip(),
                    target_language,
                    temp_file
                )
                
                if sentence_audio and os.path.exists(temp_file):
                    all_audio_chunks.append(temp_file)
                else:
                    print(f"    ⚠️  Warning: Failed to synthesize sentence {i}")
            
            # Step 4b: Concatenate all sentence audio files
            if all_audio_chunks:
                print()
                print("Concatenating sentence audio...")
                concatenate_audio_files(all_audio_chunks, output_file)
                
                # Clean up temporary files
                for temp_file in all_audio_chunks:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                
                print(f"✓ Audio saved to: {output_file}")
                print()
                return output_file
            else:
                print("✗ No audio generated from sentences")
                return None
        else:
            # Standard synthesis without sentence buffering
            output = text_to_speech_piper(text, target_language, output_file)
            if output:
                print(f"✓ Audio saved to: {output_file}")
                print()
            return output
            
    except Exception as e:
        print(f"✗ Error during speech synthesis: {e}")
        import traceback
        traceback.print_exc()
        return None


def play_audio(audio_file: str = "output.wav", blocking: bool = True) -> bool:
    """
    Step 5: Play the translated audio through speakers.
    
    This function plays the generated audio file through the system's default
    audio output device (MacBook speakers, AirPods, etc.).
    
    Args:
        audio_file: Path to the audio file to play (default: "output.wav")
        blocking: If True, wait until playback finishes. If False, return immediately.
    
    Returns:
        True if playback successful, False otherwise
    
    Example:
        >>> play_audio("output.wav", blocking=True)
        Playing output.wav...
        ✓ Playback complete!
    """
    print("=" * 70)
    print("STEP 5: Audio Playback")
    print("=" * 70)
    print()
    
    if not os.path.exists(audio_file):
        print(f"✗ Error: Audio file '{audio_file}' not found")
        print(f"   Please generate audio first using Step 4")
        return False
    
    try:
        # Use the modular playback function
        success = play_audio_file(audio_file, blocking=blocking)
        
        if success:
            print(f"✓ Playback complete!")
            print()
        return success
        
    except Exception as e:
        print(f"✗ Error during audio playback: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def split_into_sentences(text: str) -> list:
    """
    Split text into sentences for sentence-level buffering.
    
    This function splits text by sentence boundaries (periods, exclamation marks,
    question marks) to enable sentence-by-sentence TTS processing.
    
    Args:
        text: Text to split into sentences
    
    Returns:
        List of sentences (strings)
    """
    if not text or not isinstance(text, str):
        return []
    
    # Simple sentence splitting by punctuation
    # More sophisticated NLP could be used here
    sentences = re.split(r'([.!?]+)', text)
    
    # Combine punctuation with preceding sentence
    result = []
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i].strip()
        if i + 1 < len(sentences):
            sentence += sentences[i + 1]
        if sentence:
            result.append(sentence)
    
    # Handle last sentence if odd number of splits
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        result.append(sentences[-1].strip())
    
    # Filter out empty sentences
    result = [s.strip() for s in result if s.strip()]
    
    return result if result else [text.strip()]  # Return original text if no sentences found


def concatenate_audio_files(audio_files: list, output_file: str) -> bool:
    """
    Concatenate multiple audio files into a single file.
    
    This function combines multiple WAV files (e.g., from sentence-level synthesis)
    into a single output file.
    
    Args:
        audio_files: List of paths to audio files to concatenate
        output_file: Path to save the concatenated audio
    
    Returns:
        True if concatenation successful, False otherwise
    """
    import wave
    import numpy as np
    
    try:
        # Read first file to get format
        with wave.open(audio_files[0], 'rb') as first_wav:
            sample_rate = first_wav.getframerate()
            num_channels = first_wav.getnchannels()
            sample_width = first_wav.getsampwidth()
        
        # Collect all audio data
        all_audio_data = []
        
        for audio_file in audio_files:
            if not os.path.exists(audio_file):
                continue
            
            with wave.open(audio_file, 'rb') as wav_file:
                # Verify format matches
                if (wav_file.getframerate() != sample_rate or
                    wav_file.getnchannels() != num_channels or
                    wav_file.getsampwidth() != sample_width):
                    print(f"⚠️  Warning: Format mismatch in {audio_file}, skipping")
                    continue
                
                # Read audio data
                frames = wav_file.readframes(wav_file.getnframes())
                all_audio_data.append(frames)
        
        # Write concatenated audio
        if all_audio_data:
            with wave.open(output_file, 'wb') as out_wav:
                out_wav.setnchannels(num_channels)
                out_wav.setsampwidth(sample_width)
                out_wav.setframerate(sample_rate)
                
                for audio_data in all_audio_data:
                    out_wav.writeframes(audio_data)
            
            return True
        else:
            print("✗ No audio data to concatenate")
            return False
            
    except Exception as e:
        print(f"✗ Error concatenating audio files: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# COMPLETE PIPELINE FUNCTION
# ============================================================================

def run_complete_pipeline(
    target_language: str = "English",
    record_new: bool = True,
    audio_file: str = "input.wav",
    generate_speech: bool = True,
    play_audio_output: bool = True,
    sentence_buffering: bool = True
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Run the complete end-to-end pipeline: Record → Transcribe → Translate → TTS → Play.
    
    This is the main pipeline function that orchestrates all 5 steps:
    1. Record audio in Urdu from microphone
    2. Convert Urdu speech to text with Whisper
    3. Translate text to target language with TranslateGemma
    4. Convert translated text to audio with Piper TTS (with sentence-level buffering)
    5. Play the audio output
    
    Args:
        target_language: Target language for translation and TTS (default: "English")
                        Examples: "English", "Spanish", "French", "en", "es", "fr"
        record_new: If True, record new audio. If False, use existing audio file.
        audio_file: Path to audio file (used if record_new=False)
        generate_speech: Whether to generate speech output (default: True)
        play_audio_output: Whether to play audio after generation (default: True)
        sentence_buffering: Whether to use sentence-level buffering for TTS (default: True)
    
    Returns:
        Tuple of (urdu_text, translated_text, output_file)
        Returns (None, None, None) if pipeline fails
    
    Example:
        >>> urdu, english, audio = run_complete_pipeline(
        ...     target_language="English",
        ...     record_new=True,
        ...     sentence_buffering=True
        ... )
        Recording audio...
        Transcribing...
        Translating...
        Synthesizing speech...
        Playing audio...
        ✓ Pipeline complete!
    """
    print("=" * 70)
    print("🔄 COMPLETE PIPELINE: Urdu Speech → Any Language Audio")
    print("=" * 70)
    print()
    print("Pipeline Steps:")
    print("  1. Record audio in Urdu from microphone")
    print("  2. Convert Urdu speech to text (Whisper)")
    print("  3. Translate text to target language (TranslateGemma)")
    print("  4. Convert translated text to audio (Piper TTS)")
    print("  5. Play the audio output")
    print()
    print(f"Target Language: {target_language}")
    print(f"Transcription Model: whisper-small-urdu (fine-tuned for Urdu, GPU accelerated)")
    print(f"Sentence Buffering: {'Enabled' if sentence_buffering else 'Disabled'}")
    print()
    print("=" * 70)
    print()
    
    urdu_text = None
    translated_text = None
    output_file = None
    
    # Step 1: Record audio
    if record_new:
        print()
        if not record_audio(audio_file):
            print("✗ Pipeline failed at Step 1: Recording")
            return None, None, None
        audio_file = "input.wav"  # record_audio saves to input.wav
    else:
        if not os.path.exists(audio_file):
            print(f"✗ Error: Audio file '{audio_file}' not found")
            print("   Set record_new=True to record new audio")
            return None, None, None
        print(f"✓ Using existing audio file: {audio_file}")
        print()
    
    # Step 2: Speech-to-Text
    print()
    urdu_text = transcribe_audio(audio_file)
    if not urdu_text:
        print("✗ Pipeline failed at Step 2: Speech-to-Text")
        return None, None, None
    
    # Step 3: Translation
    print()
    translated_text = translate_text(urdu_text, target_language)
    if not translated_text:
        print("✗ Pipeline failed at Step 3: Translation")
        return urdu_text, None, None
    
    # Step 4: Text-to-Speech
    if generate_speech:
        print()
        output_file = synthesize_speech(
            translated_text,
            target_language,
            "output.wav",
            sentence_buffering=sentence_buffering
        )
        if not output_file:
            print("✗ Pipeline failed at Step 4: Text-to-Speech")
            return urdu_text, translated_text, None
    
    # Step 5: Play audio
    if play_audio_output and output_file:
        print()
        if not play_audio(output_file, blocking=True):
            print("⚠️  Pipeline completed but audio playback failed")
            # Don't fail the pipeline if playback fails
    
    # Final summary
    print()
    print("=" * 70)
    print("✅ COMPLETE PIPELINE FINISHED!")
    print("=" * 70)
    print()
    print("Results Summary:")
    print("-" * 70)
    print(f"1. Audio Input: {audio_file}")
    print(f"2. Urdu Text: {urdu_text}")
    print(f"3. {target_language} Translation: {translated_text}")
    if output_file:
        print(f"4. Speech Output: {output_file}")
    print("-" * 70)
    print()
    
    return urdu_text, translated_text, output_file


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Complete Urdu Speech → Any Language Audio Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Record new audio and translate to English
  python3 pipeline.py --target English
  
  # Use existing audio file and translate to Spanish
  python3 pipeline.py --target Spanish --no-record --audio input.wav
  
  # Translate to French
  python3 pipeline.py --target French
        """
    )
    
    parser.add_argument(
        "--target", "-t",
        type=str,
        default="English",
        help="Target language for translation and TTS (default: English)"
    )
    
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Use existing audio file instead of recording new audio"
    )
    
    parser.add_argument(
        "--audio", "-a",
        type=str,
        default="input.wav",
        help="Audio file path (used with --no-record, default: input.wav)"
    )
    
    parser.add_argument(
        "--no-speech",
        action="store_true",
        help="Skip text-to-speech generation"
    )
    
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Skip audio playback"
    )
    
    parser.add_argument(
        "--no-buffering",
        action="store_true",
        help="Disable sentence-level buffering for TTS"
    )
    
    args = parser.parse_args()
    
    # Run the complete pipeline (always uses fine-tuned whisper-small-urdu model)
    run_complete_pipeline(
        target_language=args.target,
        record_new=not args.no_record,
        audio_file=args.audio,
        generate_speech=not args.no_speech,
        play_audio_output=not args.no_play,
        sentence_buffering=not args.no_buffering
    )


if __name__ == "__main__":
    main()
