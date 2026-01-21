"""
Audio Recording Script for MacBook Microphone
Records audio in 16-bit PCM format at 16 kHz sample rate.
Supports both fixed duration and interactive (press Enter to stop) recording modes.
"""

import sounddevice as sd
import numpy as np
import wave
import sys
import threading
import queue
from typing import Optional


class AudioRecorder:
    """Class to handle audio recording from microphone."""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1, dtype: str = 'int16', device: Optional[int] = None):
        """
        Initialize the audio recorder.
        
        Args:
            sample_rate: Sample rate in Hz (default: 16000 for 16 kHz)
            channels: Number of audio channels (1 for mono, 2 for stereo)
            dtype: Data type for audio samples ('int16' for 16-bit PCM)
            device: Audio device index (None for default, or specify device number)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.recording = False
        self.audio_queue = queue.Queue()
        self.device = device
        
        # Auto-detect best microphone if device not specified
        # Priority: AirPods/Bluetooth > MacBook Pro Microphone > Default
        if self.device is None:
            self.device = self._find_best_microphone()
        
    def _find_best_microphone(self) -> Optional[int]:
        """
        Try to find the best available microphone device.
        Priority: AirPods/Bluetooth devices > MacBook Pro Microphone > Default input
        
        Returns:
            Device index if found, None otherwise
        """
        devices = sd.query_devices()
        airpods_device = None
        macbook_mic = None
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                name = device['name'].lower()
                
                # Check for AirPods or other Bluetooth audio devices
                if 'airpod' in name or 'bluetooth' in name or 'bt' in name:
                    airpods_device = i
                
                # Check for MacBook microphone
                if 'macbook' in name and 'microphone' in name:
                    macbook_mic = i
        
        # Return AirPods if available, otherwise MacBook mic, otherwise default
        if airpods_device is not None:
            return airpods_device
        elif macbook_mic is not None:
            return macbook_mic
        else:
            return sd.default.device[0] if sd.default.device[0] is not None else None
    
    def list_audio_devices(self):
        """List all available audio input devices."""
        print("\nAvailable audio input devices:")
        print("-" * 60)
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                markers = []
                if i == self.device:
                    markers.append("SELECTED")
                # Identify device types
                name_lower = device['name'].lower()
                if 'airpod' in name_lower or 'bluetooth' in name_lower:
                    markers.append("Bluetooth")
                elif 'macbook' in name_lower and 'microphone' in name_lower:
                    markers.append("Built-in")
                
                marker_str = f" ({', '.join(markers)})" if markers else ""
                print(f"Device {i}: {device['name']} "
                      f"(Input channels: {device['max_input_channels']}, "
                      f"Sample rate: {device['default_samplerate']} Hz){marker_str}")
        print("-" * 60)
    
    def audio_callback(self, indata, frames, time, status):
        """
        Callback function called by sounddevice for each audio block.
        
        Args:
            indata: Input audio data (numpy array)
            frames: Number of frames in this block
            time: Current time information
            status: Status flags
        """
        if status:
            print(f"Warning: {status}", file=sys.stderr)
        
        # Convert float32 to int16 if needed and add to queue
        if self.dtype == 'int16':
            # Convert from float32 [-1.0, 1.0] to int16 [-32768, 32767]
            audio_data = (indata * 32767).astype(np.int16)
        else:
            audio_data = indata
        
        # Add audio data to queue for processing
        self.audio_queue.put(audio_data.copy())
    
    def record_with_duration(self, duration: float, output_file: str = "input.wav") -> str:
        """
        Record audio for a fixed duration.
        
        Args:
            duration: Recording duration in seconds
            output_file: Output WAV file path
            
        Returns:
            Path to the saved audio file
        """
        print(f"\nRecording for {duration} seconds...")
        print("Recording started. Speak now...")
        
        # Check if device is available
        if self.device is not None:
            device_info = sd.query_devices(self.device)
            print(f"Using device: {device_info['name']}")
        
        # List to store all audio chunks
        audio_chunks = []
        
        try:
            # Start recording stream with specified device
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32',  # sounddevice uses float32 internally
                callback=self.audio_callback,
                blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
                device=self.device  # Use specified device
            ):
                # Record for the specified duration
                sd.sleep(int(duration * 1000))  # Convert seconds to milliseconds
            
            # Collect all audio data from queue
            print("Processing recorded audio...")
            while not self.audio_queue.empty():
                chunk = self.audio_queue.get()
                audio_chunks.append(chunk)
            
            # Concatenate all chunks into a single array
            if audio_chunks:
                audio_data = np.concatenate(audio_chunks, axis=0)
                
                # Convert from float32 to int16 for 16-bit PCM
                audio_data = (audio_data * 32767).astype(np.int16)
                
                # Save to WAV file
                self._save_wav(audio_data, output_file)
                print(f"✓ Recording saved to: {output_file}")
                return output_file
            else:
                print("✗ No audio data recorded.")
                return None
                
        except KeyboardInterrupt:
            print("\n✗ Recording interrupted by user.")
            return None
        except Exception as e:
            print(f"✗ Error during recording: {e}")
            return None
    
    def record_until_enter(self, output_file: str = "input.wav") -> str:
        """
        Record audio until user presses Enter.
        
        Args:
            output_file: Output WAV file path
            
        Returns:
            Path to the saved audio file
        """
        print("\nRecording started. Press Enter to stop recording...")
        print("Speak now...")
        
        # List to store all audio chunks
        audio_chunks = []
        self.recording = True
        
        def stop_recording():
            """Wait for user to press Enter, then stop recording."""
            input()  # Wait for Enter key
            self.recording = False
            print("\nStopping recording...")
        
        # Start thread to wait for Enter key
        stop_thread = threading.Thread(target=stop_recording, daemon=True)
        stop_thread.start()
        
        try:
            # Start recording stream with specified device
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32',  # sounddevice uses float32 internally
                callback=self.audio_callback,
                blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
                device=self.device  # Use specified device
            ):
                # Keep recording until user presses Enter
                while self.recording:
                    sd.sleep(100)  # Sleep for 100ms and check again
                
                # Give a small buffer to finish current block
                sd.sleep(200)
            
            # Collect all audio data from queue
            print("Processing recorded audio...")
            while not self.audio_queue.empty():
                chunk = self.audio_queue.get()
                audio_chunks.append(chunk)
            
            # Concatenate all chunks into a single array
            if audio_chunks:
                audio_data = np.concatenate(audio_chunks, axis=0)
                
                # Convert from float32 to int16 for 16-bit PCM
                audio_data = (audio_data * 32767).astype(np.int16)
                
                # Save to WAV file
                self._save_wav(audio_data, output_file)
                print(f"✓ Recording saved to: {output_file}")
                return output_file
            else:
                print("✗ No audio data recorded.")
                return None
                
        except KeyboardInterrupt:
            print("\n✗ Recording interrupted by user.")
            self.recording = False
            return None
        except Exception as e:
            print(f"✗ Error during recording: {e}")
            self.recording = False
            return None
    
    def _save_wav(self, audio_data: np.ndarray, filename: str):
        """
        Save audio data to a WAV file in 16-bit PCM format.
        
        Args:
            audio_data: Audio data as numpy array (int16)
            filename: Output filename
        """
        # Ensure audio_data is 1D for mono, or 2D for stereo
        if len(audio_data.shape) == 1:
            # Mono audio
            audio_data = audio_data.reshape(-1, 1)
        elif len(audio_data.shape) == 2 and audio_data.shape[1] == 1:
            # Already in correct shape
            pass
        else:
            # Convert to mono if stereo
            audio_data = np.mean(audio_data, axis=1).reshape(-1, 1)
        
        # Open WAV file for writing
        with wave.open(filename, 'wb') as wav_file:
            # Set WAV file parameters
            wav_file.setnchannels(self.channels)  # Number of channels (1 = mono)
            wav_file.setsampwidth(2)  # Sample width in bytes (2 bytes = 16-bit)
            wav_file.setframerate(self.sample_rate)  # Sample rate (16000 Hz)
            
            # Write audio data to file
            # Convert int16 array to bytes
            wav_file.writeframes(audio_data.tobytes())


def main():
    """Main function to run the audio recorder."""
    print("=" * 60)
    print("Audio Recorder - MacBook Microphone")
    print("=" * 60)
    
    # Initialize recorder with 16-bit PCM, 16 kHz sample rate
    recorder = AudioRecorder(
        sample_rate=16000,  # 16 kHz sample rate
        channels=1,          # Mono audio
        dtype='int16'       # 16-bit PCM
    )
    
    # List available audio devices
    recorder.list_audio_devices()
    
    # Ask user for recording mode
    print("\nSelect recording mode:")
    print("1. Record for a fixed duration (in seconds)")
    print("2. Record until Enter is pressed")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        # Fixed duration mode
        try:
            duration = float(input("Enter recording duration in seconds: "))
            if duration <= 0:
                print("✗ Duration must be positive.")
                return
            recorder.record_with_duration(duration, "input.wav")
        except ValueError:
            print("✗ Invalid duration. Please enter a number.")
    elif choice == "2":
        # Press Enter to stop mode
        recorder.record_until_enter("input.wav")
    else:
        print("✗ Invalid choice. Please run the script again and select 1 or 2.")
    
    print("\n" + "=" * 60)
    print("Recording session completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
