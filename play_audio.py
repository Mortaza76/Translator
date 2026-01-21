"""
Play audio file using sounddevice.
Plays the translated audio output from the TTS pipeline.
"""

import sounddevice as sd
import wave
import numpy as np
import sys
import os


def play_audio_file(audio_file: str = "output.wav", blocking: bool = True):
    """
    Play an audio WAV file through speakers.
    
    This function loads a WAV file and plays it through the system's
    default audio output device (MacBook speakers, AirPods, etc.).
    
    Args:
        audio_file: Path to the WAV file to play (default: "output.wav")
        blocking: If True, wait until playback finishes. If False, return immediately.
    
    Returns:
        True if playback successful, False otherwise
    
    Example:
        >>> play_audio_file("output.wav")
        Playing output.wav...
        ✓ Playback complete!
    """
    # Step 1: Check if audio file exists
    if not os.path.exists(audio_file):
        print(f"✗ Error: Audio file '{audio_file}' not found")
        print(f"   Please generate audio first using: python3 text_to_speech.py")
        return False
    
    print("=" * 70)
    print("🔊 Playing Audio")
    print("=" * 70)
    print()
    
    # Step 2: Load WAV file
    # Open the WAV file and read its properties
    print(f"Loading audio file: {audio_file}")
    try:
        with wave.open(audio_file, 'rb') as wav_file:
            # Get audio properties
            sample_rate = wav_file.getframerate()
            num_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            num_frames = wav_file.getnframes()
            duration = num_frames / sample_rate
            
            print(f"✓ File loaded")
            print(f"  Sample rate: {sample_rate} Hz")
            print(f"  Channels: {num_channels} ({'Mono' if num_channels == 1 else 'Stereo'})")
            print(f"  Sample width: {sample_width * 8} bits")
            print(f"  Duration: {duration:.2f} seconds")
            print()
            
            # Step 3: Read audio data
            # Read all frames from the WAV file
            print("Reading audio data...")
            frames = wav_file.readframes(num_frames)
            
            # Step 4: Convert audio data to numpy array
            # WAV files store audio as bytes, need to convert to numpy array
            # Based on sample width, convert to appropriate dtype
            if sample_width == 1:
                # 8-bit audio (unsigned)
                audio_data = np.frombuffer(frames, dtype=np.uint8)
                # Convert to float32 range [-1.0, 1.0]
                audio_float = (audio_data.astype(np.float32) - 128) / 128.0
            elif sample_width == 2:
                # 16-bit audio (signed)
                audio_data = np.frombuffer(frames, dtype=np.int16)
                # Convert to float32 range [-1.0, 1.0]
                audio_float = audio_data.astype(np.float32) / 32767.0
            elif sample_width == 4:
                # 32-bit audio (signed)
                audio_data = np.frombuffer(frames, dtype=np.int32)
                # Convert to float32 range [-1.0, 1.0]
                audio_float = audio_data.astype(np.float32) / 2147483647.0
            else:
                print(f"✗ Unsupported sample width: {sample_width * 8} bits")
                return False
            
            # Step 5: Handle mono/stereo conversion
            # sounddevice expects shape (num_samples,) for mono or (num_samples, channels) for stereo
            if num_channels == 1:
                # Mono: already in correct shape
                audio_playback = audio_float
            elif num_channels == 2:
                # Stereo: reshape to (num_samples, 2)
                audio_playback = audio_float.reshape(-1, 2)
            else:
                print(f"✗ Unsupported number of channels: {num_channels}")
                return False
            
            print("✓ Audio data prepared")
            print()
            
            # Step 6: Get output device info
            # sounddevice will use the default output device
            device_info = sd.query_devices(kind='output')
            print(f"Playing through: {device_info['name']}")
            print()
            
            # Step 7: Play audio
            # sounddevice.play() plays the audio array
            print("🔊 Playing audio...")
            print()
            
            # Play the audio
            # blocking=True means wait until playback finishes
            sd.play(audio_playback, samplerate=sample_rate, blocking=blocking)
            
            if blocking:
                # Wait for playback to finish (if not already finished)
                sd.wait()
                print("✓ Playback complete!")
            else:
                print("✓ Playback started (non-blocking)")
            
            print()
            print("=" * 70)
            
            return True
            
    except FileNotFoundError:
        print(f"✗ Error: File '{audio_file}' not found")
        return False
    except Exception as e:
        print(f"✗ Error playing audio: {e}")
        import traceback
        traceback.print_exc()
        return False


def play_audio_streaming(audio_file: str = "output.wav", chunk_size: int = 1024):
    """
    Play audio file with streaming playback (for large files).
    
    This function reads and plays audio in chunks, which is more
    memory-efficient for large audio files.
    
    Args:
        audio_file: Path to the WAV file to play
        chunk_size: Number of frames to read per chunk (default: 1024)
    
    Returns:
        True if playback successful, False otherwise
    """
    if not os.path.exists(audio_file):
        print(f"✗ Error: Audio file '{audio_file}' not found")
        return False
    
    print("=" * 70)
    print("🔊 Streaming Audio Playback")
    print("=" * 70)
    print()
    
    try:
        with wave.open(audio_file, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            num_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            
            print(f"Streaming: {audio_file}")
            print(f"  Sample rate: {sample_rate} Hz")
            print(f"  Channels: {num_channels}")
            print()
            
            # Open audio stream
            print("🔊 Playing audio (streaming)...")
            print()
            
            with sd.OutputStream(samplerate=sample_rate, channels=num_channels) as stream:
                # Read and play in chunks
                while True:
                    # Read chunk of frames
                    frames = wav_file.readframes(chunk_size)
                    
                    if not frames:
                        # End of file
                        break
                    
                    # Convert to numpy array
                    if sample_width == 2:
                        audio_chunk = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
                    else:
                        # Handle other sample widths if needed
                        audio_chunk = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
                    
                    # Reshape for stereo if needed
                    if num_channels == 2:
                        audio_chunk = audio_chunk.reshape(-1, 2)
                    
                    # Write chunk to audio stream
                    stream.write(audio_chunk)
            
            print("✓ Streaming playback complete!")
            print()
            print("=" * 70)
            
            return True
            
    except Exception as e:
        print(f"✗ Error in streaming playback: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to play audio file."""
    # Get audio file from command line or use default
    audio_file = "output.wav"
    streaming = False
    
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    
    if len(sys.argv) > 2 and sys.argv[2] == "--stream":
        streaming = True
    
    # Play audio
    if streaming:
        play_audio_streaming(audio_file)
    else:
        play_audio_file(audio_file, blocking=True)


if __name__ == "__main__":
    main()
