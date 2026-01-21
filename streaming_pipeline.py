"""
Real-Time Streaming Pipeline: Live Urdu Speech → Any Language Audio
Enables continuous, real-time translation with incremental playback.

Features:
- Stream audio chunks from microphone to Whisper for near real-time transcription
- Stream translated sentences immediately to Piper TTS for faster playback
- Continuous speaking with incremental translations
- Proper buffering and sentence segmentation to avoid choppy audio

Optimized for M1 MacBook with 8GB RAM.
"""

import os
import sys
import re
import time
import threading
import queue
from typing import Optional, Callable
import numpy as np
import sounddevice as sd
import wave

# Import modular functions
from speech_to_text import transcribe_urdu_audio
from translate_text import translate_urdu_to_target
from text_to_speech import text_to_speech_piper
from play_audio import play_audio_file


# ============================================================================
# STREAMING AUDIO RECORDING
# ============================================================================

class StreamingAudioRecorder:
    """
    Records audio in chunks for real-time processing.
    
    This class continuously records audio from the microphone and provides
    chunks of audio data for processing. Uses a circular buffer to handle
    streaming efficiently.
    """
    
    def __init__(self, sample_rate: int = 16000, chunk_duration: float = 4.0,
                 device: Optional[int] = None):
        """
        Initialize streaming audio recorder.
        
        Args:
            sample_rate: Audio sample rate in Hz (default: 16000 for Whisper)
            chunk_duration: Duration of each audio chunk in seconds (default: 4.0)
            device: Audio input device index (None = default)
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.device = device
        
        # Audio buffer and queue for chunks
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.recording_thread = None
        
        # Buffer for accumulating audio
        self.audio_buffer = []
    
    def _audio_callback(self, indata, frames, time_info, status):
        """
        Callback function for sounddevice streaming.
        
        This is called continuously while recording, providing audio chunks
        in real-time. We accumulate chunks and process them when ready.
        """
        if status:
            print(f"Audio callback status: {status}")
        
        # Convert float32 to int16 for storage
        audio_chunk = (indata[:, 0] * 32767).astype(np.int16)
        self.audio_buffer.append(audio_chunk)
        
        # When we have enough samples for a chunk, process it
        total_samples = sum(len(chunk) for chunk in self.audio_buffer)
        if total_samples >= self.chunk_samples:
            # Combine chunks into one
            combined_chunk = np.concatenate(self.audio_buffer)
            
            # Put chunk in queue for processing
            self.audio_queue.put(combined_chunk[:self.chunk_samples])
            
            # Keep remaining samples in buffer
            remaining = combined_chunk[self.chunk_samples:]
            self.audio_buffer = [remaining] if len(remaining) > 0 else []
    
    def start_recording(self):
        """Start streaming audio recording."""
        if self.is_recording:
            return
        
        self.is_recording = True
        self.audio_buffer = []
        
        # Start recording stream
        print(f"🎤 Starting streaming audio recording...")
        print(f"   Sample rate: {self.sample_rate} Hz")
        print(f"   Chunk duration: {self.chunk_duration} seconds")
        print()
        
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                device=self.device,
                callback=self._audio_callback,
                blocksize=int(self.sample_rate * 0.5)  # 0.5 second blocks
            )
            self.stream.start()
            print("✓ Recording started - speak now!")
            print()
        except Exception as e:
            print(f"✗ Error starting recording: {e}")
            self.is_recording = False
            raise
    
    def stop_recording(self):
        """Stop streaming audio recording."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        try:
            if hasattr(self, 'stream'):
                self.stream.stop()
                self.stream.close()
            print("✓ Recording stopped")
        except Exception as e:
            print(f"⚠️  Error stopping recording: {e}")
    
    def get_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get next audio chunk from the recording queue.
        
        Args:
            timeout: Maximum time to wait for a chunk (seconds)
        
        Returns:
            Audio chunk as numpy array, or None if timeout
        """
        try:
            chunk = self.audio_queue.get(timeout=timeout)
            return chunk
        except queue.Empty:
            return None
    
    def save_chunk_to_wav(self, chunk: np.ndarray, filename: str) -> bool:
        """
        Save an audio chunk to a WAV file for processing.
        
        Args:
            chunk: Audio data as numpy array (int16)
            filename: Output WAV file path
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(chunk.tobytes())
            return True
        except Exception as e:
            print(f"✗ Error saving chunk: {e}")
            return False


# ============================================================================
# STREAMING TRANSCRIPTION PROCESSOR
# ============================================================================

class StreamingTranscriber:
    """
    Processes audio chunks through Whisper for real-time transcription.
    
    This class takes audio chunks from the recorder and transcribes them
    using Whisper. It handles the transcription in a separate thread to
    avoid blocking the audio recording.
    """
    
    def __init__(self, chunk_file: str = "temp_chunk.wav"):
        """
        Initialize streaming transcriber with fine-tuned whisper-small-urdu model.
        
        Args:
            chunk_file: Temporary file path for audio chunks
        """
        self.chunk_file = chunk_file
        self.processor = None
        self.model = None
        self.device = None
        self.transcription_queue = queue.Queue()
        self.is_processing = False
        self.processing_thread = None
        
        # Load fine-tuned Whisper model
        print("Loading fine-tuned whisper-small-urdu model...")
        try:
            import torch
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            
            model_name = "khawajaaliarshad/whisper-small-urdu"
            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
            
            # Use MPS (Apple GPU) if available, otherwise CPU
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.model.to(self.device)
            
            if self.device == "mps":
                print("✓ Fine-tuned model loaded on Apple GPU (MPS)")
            else:
                print("✓ Fine-tuned model loaded on CPU")
        except Exception as e:
            print(f"✗ Error loading fine-tuned model: {e}")
            raise
    
    def process_chunk(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> Optional[str]:
        """
        Transcribe a single audio chunk using fine-tuned model.
        
        Args:
            audio_chunk: Audio data as numpy array
            sample_rate: Sample rate of the audio
        
        Returns:
            Transcribed text, or None if transcription failed
        """
        if self.model is None or self.processor is None:
            return None
        
        try:
            # Convert to mono if stereo
            if audio_chunk.ndim > 1:
                audio_chunk = np.mean(audio_chunk, axis=1)
            
            # Process with fine-tuned model
            import torch
            inputs = self.processor(audio_chunk, sampling_rate=sample_rate, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                # Generate transcription
                # Remove forced_decoder_ids from generation config to avoid conflict
                generation_config = self.model.generation_config
                if hasattr(generation_config, 'forced_decoder_ids'):
                    generation_config.forced_decoder_ids = None
                
                predicted_ids = self.model.generate(
                    inputs["input_features"],
                    generation_config=generation_config
                )
                text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            return text.strip() if text else None
            
        except Exception as e:
            print(f"✗ Error transcribing chunk: {e}")
            return None
    
    def start_processing(self, audio_queue: queue.Queue, sample_rate: int = 16000):
        """
        Start processing audio chunks in a background thread.
        
        Args:
            audio_queue: Queue containing audio chunks
            sample_rate: Sample rate of the audio
        """
        if self.is_processing:
            return
        
        self.is_processing = True
        
        def process_loop():
            """Background processing loop."""
            while self.is_processing:
                try:
                    # Get chunk from queue (non-blocking)
                    audio_chunk = audio_queue.get(timeout=0.5)
                    
                    if audio_chunk is not None:
                        # Transcribe chunk
                        transcribed = self.process_chunk(audio_chunk, sample_rate)
                        
                        if transcribed:
                            print(f"  📝 Transcribed: {transcribed}")
                            # Put transcription in output queue
                            self.transcription_queue.put(transcribed)
                        else:
                            print(f"  ⚠️  No speech detected in chunk (silent/empty)")
                            
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"✗ Error in processing loop: {e}")
                    continue
        
        self.processing_thread = threading.Thread(target=process_loop, daemon=True)
        self.processing_thread.start()
        print("✓ Transcription processing started")
    
    def stop_processing(self):
        """Stop processing audio chunks."""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        print("✓ Transcription processing stopped")
    
    def get_transcription(self, timeout: float = 1.0) -> Optional[str]:
        """
        Get next transcription from the queue.
        
        Args:
            timeout: Maximum time to wait (seconds)
        
        Returns:
            Transcribed text, or None if timeout
        """
        try:
            return self.transcription_queue.get(timeout=timeout)
        except queue.Empty:
            return None


# ============================================================================
# STREAMING TRANSLATION AND TTS
# ============================================================================

class StreamingTranslator:
    """
    Translates text chunks and synthesizes speech in real-time.
    
    This class handles translation and TTS for streaming text, processing
    sentences incrementally for faster playback.
    """
    
    def __init__(self, target_language: str = "English"):
        """
        Initialize streaming translator.
        
        Args:
            target_language: Target language for translation
        """
        self.target_language = target_language
        self.translation_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.is_processing = False
        self.processing_thread = None
        
        # Initialize Ollama client for translation
        from ollama_integration import OllamaClient
        self.translation_client = OllamaClient()
        print(f"✓ Translation client initialized for {target_language}")
    
    def process_text(self, urdu_text: str) -> Optional[str]:
        """
        Translate a text chunk to target language.
        
        Args:
            urdu_text: Urdu text to translate
        
        Returns:
            Translated text, or None if translation failed
        """
        if not urdu_text or not urdu_text.strip():
            return None
        
        try:
            translated = translate_urdu_to_target(urdu_text, self.target_language)
            return translated.strip() if translated else None
        except Exception as e:
            print(f"✗ Error translating: {e}")
            return None
    
    def synthesize_sentence(self, text: str) -> Optional[str]:
        """
        Synthesize a sentence to audio.
        
        Args:
            text: Text to synthesize
        
        Returns:
            Path to audio file, or None if synthesis failed
        """
        if not text or not text.strip():
            return None
        
        try:
            # Generate unique filename for this sentence
            timestamp = int(time.time() * 1000)
            audio_file = f"temp_stream_{timestamp}.wav"
            
            # Synthesize speech
            result = text_to_speech_piper(text, self.target_language, audio_file)
            
            if result and os.path.exists(audio_file):
                return audio_file
            return None
        except Exception as e:
            print(f"✗ Error synthesizing: {e}")
            return None
    
    def start_processing(self, transcription_queue: queue.Queue):
        """
        Start processing transcriptions in a background thread.
        
        Args:
            transcription_queue: Queue containing transcribed text
        """
        if self.is_processing:
            return
        
        self.is_processing = True
        
        def process_loop():
            """Background processing loop for translation and TTS."""
            accumulated_text = ""
            
            while self.is_processing:
                try:
                    # Get transcription from queue
                    transcribed = transcription_queue.get(timeout=0.5)
                    
                    if transcribed:
                        print(f"  🌐 Processing translation for: {transcribed[:50]}...")
                        # Accumulate text
                        accumulated_text += " " + transcribed
                        accumulated_text = accumulated_text.strip()
                        
                        # Check if we have a complete sentence
                        sentences = self._split_into_sentences(accumulated_text)
                        
                        # Process complete sentences
                        # If we have multiple sentences, process all but the last
                        # If we only have one sentence, process it if it's long enough (likely complete)
                        sentences_to_process = sentences[:-1] if len(sentences) > 1 else []
                        
                        # If only one sentence and it's substantial (likely complete), process it
                        if len(sentences) == 1 and len(sentences[0].strip()) > 10:
                            sentences_to_process = sentences
                            accumulated_text = ""  # Clear since we're processing it
                        
                        for sentence in sentences_to_process:
                            if sentence.strip():
                                # Translate sentence
                                translated = self.process_text(sentence)
                                
                                if translated:
                                    print(f"  ✓ Translated: {translated[:50]}...")
                                    # Synthesize audio
                                    audio_file = self.synthesize_sentence(translated)
                                    
                                    if audio_file:
                                        # Put audio in queue for playback
                                        self.audio_queue.put(audio_file)
                                        print(f"  🔊 Ready to play: {translated[:50]}...")
                                else:
                                    print(f"  ⚠️  Translation failed for: {sentence[:50]}...")
                        
                        # Keep incomplete sentence for next iteration (only if we didn't process it)
                        if len(sentences) > 1:
                            accumulated_text = sentences[-1] if sentences else ""
                        elif len(sentences) == 1 and len(sentences[0].strip()) <= 10:
                            # Keep short single sentences to accumulate
                            accumulated_text = sentences[0] if sentences else ""
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"✗ Error in translation/TTS loop: {e}")
                    continue
        
        self.processing_thread = threading.Thread(target=process_loop, daemon=True)
        self.processing_thread.start()
        print("✓ Translation and TTS processing started")
    
    def stop_processing(self):
        """Stop processing transcriptions."""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        print("✓ Translation and TTS processing stopped")
    
    def _split_into_sentences(self, text: str) -> list:
        """Split text into sentences."""
        if not text:
            return []
        
        # Simple sentence splitting
        sentences = re.split(r'([.!?]+)', text)
        result = []
        
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i].strip()
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]
            if sentence:
                result.append(sentence)
        
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1].strip())
        
        return [s.strip() for s in result if s.strip()] if result else [text.strip()]
    
    def get_audio_file(self, timeout: float = 1.0) -> Optional[str]:
        """
        Get next audio file from the queue.
        
        Args:
            timeout: Maximum time to wait (seconds)
        
        Returns:
            Path to audio file, or None if timeout
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None


# ============================================================================
# STREAMING AUDIO PLAYBACK
# ============================================================================

class StreamingAudioPlayer:
    """
    Plays audio files in sequence for streaming playback.
    
    This class handles playing audio files as they become available,
    creating a continuous playback experience.
    """
    
    def __init__(self):
        """Initialize streaming audio player."""
        self.is_playing = False
        self.playback_thread = None
    
    def start_playback(self, audio_queue: queue.Queue):
        """
        Start playing audio files from queue in a background thread.
        
        Args:
            audio_queue: Queue containing paths to audio files
        """
        if self.is_playing:
            return
        
        self.is_playing = True
        
        def playback_loop():
            """Background playback loop."""
            while self.is_playing:
                try:
                    # Get audio file from queue
                    audio_file = audio_queue.get(timeout=0.5)
                    
                    if audio_file and os.path.exists(audio_file):
                        # Play audio
                        play_audio_file(audio_file, blocking=True)
                        
                        # Clean up temporary file
                        try:
                            os.remove(audio_file)
                        except:
                            pass
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"✗ Error in playback loop: {e}")
                    continue
        
        self.playback_thread = threading.Thread(target=playback_loop, daemon=True)
        self.playback_thread.start()
        print("✓ Audio playback started")
    
    def stop_playback(self):
        """Stop playing audio."""
        self.is_playing = False
        if self.playback_thread:
            self.playback_thread.join(timeout=2.0)
        print("✓ Audio playback stopped")


# ============================================================================
# COMPLETE STREAMING PIPELINE
# ============================================================================

def run_streaming_pipeline(
    target_language: str = "English",
    chunk_duration: float = 4.0,
    sample_rate: int = 16000
):
    """
    Run the complete real-time streaming pipeline.
    
    This function orchestrates all components for real-time translation:
    1. Stream audio chunks from microphone
    2. Transcribe chunks with fine-tuned whisper-small-urdu (near real-time)
    3. Translate sentences incrementally
    4. Synthesize speech for each sentence
    5. Play audio as it becomes available
    
    Args:
        target_language: Target language for translation (default: "English")
        chunk_duration: Duration of each audio chunk in seconds (default: 4.0)
        sample_rate: Audio sample rate in Hz (default: 16000)
    """
    print("=" * 70)
    print("🔄 REAL-TIME STREAMING PIPELINE")
    print("=" * 70)
    print()
    print("Features:")
    print("  • Continuous audio recording")
    print("  • Near real-time transcription")
    print("  • Incremental translation and TTS")
    print("  • Streaming audio playback")
    print()
    print(f"Target Language: {target_language}")
    print(f"Transcription Model: whisper-small-urdu (fine-tuned for Urdu, GPU accelerated)")
    print(f"Chunk Duration: {chunk_duration} seconds")
    print()
    print("=" * 70)
    print()
    print("Press Ctrl+C to stop...")
    print()
    
    # Initialize components
    recorder = StreamingAudioRecorder(
        sample_rate=sample_rate,
        chunk_duration=chunk_duration
    )
    
    transcriber = StreamingTranscriber()
    
    translator = StreamingTranslator(target_language=target_language)
    
    player = StreamingAudioPlayer()
    
    try:
        # Start all components
        recorder.start_recording()
        transcriber.start_processing(recorder.audio_queue, sample_rate)
        translator.start_processing(transcriber.transcription_queue)
        player.start_playback(translator.audio_queue)
        
        print()
        print("=" * 70)
        print("🎤 LIVE TRANSLATION ACTIVE - Speak in Urdu now!")
        print("=" * 70)
        print()
        print("You will hear translations as you speak...")
        print()
        
        # Main loop - keep running until interrupted
        while True:
            time.sleep(0.1)
            
            # Optional: Print status updates
            # (Can be removed or made optional)
    
    except KeyboardInterrupt:
        print()
        print()
        print("=" * 70)
        print("🛑 Stopping streaming pipeline...")
        print("=" * 70)
        print()
    
    finally:
        # Stop all components
        recorder.stop_recording()
        transcriber.stop_processing()
        translator.stop_processing()
        player.stop_playback()
        
        # Clean up any remaining temporary files
        import glob
        for temp_file in glob.glob("temp_stream_*.wav"):
            try:
                os.remove(temp_file)
            except:
                pass
        
        print()
        print("=" * 70)
        print("✅ Streaming pipeline stopped")
        print("=" * 70)


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Real-Time Streaming Urdu Speech → Any Language Audio Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start real-time translation to English
  python3 streaming_pipeline.py --target English
  
  # Translate to Spanish
  python3 streaming_pipeline.py --target Spanish
  
  # Custom chunk duration (longer = less frequent updates)
  python3 streaming_pipeline.py --target French --chunk-duration 4.0
        """
    )
    
    parser.add_argument(
        "--target", "-t",
        type=str,
        default="English",
        help="Target language for translation and TTS (default: English)"
    )
    
    parser.add_argument(
        "--chunk-duration", "-d",
        type=float,
        default=4.0,
        help="Duration of each audio chunk in seconds (default: 4.0)"
    )
    
    parser.add_argument(
        "--sample-rate", "-r",
        type=int,
        default=16000,
        help="Audio sample rate in Hz (default: 16000)"
    )
    
    args = parser.parse_args()
    
    # Run streaming pipeline (always uses fine-tuned whisper-small-urdu model)
    run_streaming_pipeline(
        target_language=args.target,
        chunk_duration=args.chunk_duration,
        sample_rate=args.sample_rate
    )


if __name__ == "__main__":
    main()
#this is the pipeline which will run everything in real time, sending audio chunks for 2 second time, and then feeding the recorded audio to the pipeline.
#the pipeline will translate the audio to the target language and will give a near to real-time translation. 
#will do its due diligence testing to make sure the pipeline is working as expected. 
#will start testing the pipeline now
#after the testing has produced satisfactory results, will start integrating the pipeline with a UI and possibly a 
#mobile app. 