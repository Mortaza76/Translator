"""
Record audio specifically using AirPods microphone.
"""

from record_audio import AudioRecorder
from audio_processor import AudioProcessor
import sounddevice as sd
import numpy as np
import time
import sys


def record_with_airpods():
    """Record audio using AirPods microphone."""
    print("=" * 70)
    print("🎤 Recording with AirPods")
    print("=" * 70)
    print()
    
    # Find AirPods device
    devices = sd.query_devices()
    airpods_device = None
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            name_lower = device['name'].lower()
            if 'airpod' in name_lower:
                airpods_device = i
                print(f"✓ Found: {device['name']} (Device {i})")
                break
    
    if airpods_device is None:
        print("✗ AirPods not found or not connected")
        print("   Please make sure your AirPods are connected to your MacBook")
        return
    
    device_name = devices[airpods_device]['name']
    print(f"Using: {device_name}")
    print()
    
    # Initialize recorder with AirPods
    # Use 48 kHz for better quality (like iMessage)
    recorder = AudioRecorder(
        sample_rate=48000,
        channels=1,
        dtype='int16',
        device=airpods_device
    )
    
    output_file = "input.wav"
    duration = 5.0
    
    print("Ready to record with AirPods!")
    print()
    print("💡 Tips:")
    print("   - Speak clearly into your AirPods")
    print("   - Make sure AirPods are properly positioned")
    print("   - Speak at normal volume")
    print()
    
    # Wait for user to press Enter
    input("Press Enter when you're ready to start recording...")
    print()
    
    print("Recording will start in 3 seconds...")
    print("Get ready to speak!")
    print()
    
    # Countdown with clear messages
    for i in range(3, 0, -1):
        print(f"⏰ Starting in {i}...", end='\r')
        time.sleep(1)
    print()  # New line after countdown
    
    print()
    print()
    print("=" * 70)
    print("🔴 RECORDING NOW - Speak into your AirPods!")
    print("=" * 70)
    print()
    
    try:
        # Record
        result = recorder.record_with_duration(
            duration=duration,
            output_file=output_file
        )
        
        if result:
            print()
            print("=" * 70)
            print("✅ Recording Complete!")
            print("=" * 70)
            print(f"\n✓ Audio saved to: {output_file}")
            print()
            
            # Minimal processing for clean voice (like iMessage)
            print("Applying minimal processing for clean voice...")
            try:
                import sounddevice as sd
                import wave
                from scipy import signal
                
                # Load the recorded file
                with wave.open(output_file, 'rb') as wav:
                    frames = wav.readframes(-1)
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
                    sample_rate = wav.getframerate()
                
                # Minimal processing: just remove DC offset and gentle filtering
                audio = audio - np.mean(audio)  # Remove DC offset
                
                # Gentle high-pass to remove rumble (less aggressive)
                nyquist = sample_rate / 2
                b, a = signal.butter(2, 80 / nyquist, btype='high')
                audio = signal.filtfilt(b, a, audio)
                
                # Normalize gently
                max_val = np.max(np.abs(audio))
                if max_val > 0:
                    audio = audio * (0.9 / max_val)
                
                # Save clean version
                audio_int16 = (audio * 32767).astype(np.int16)
                with wave.open("input_cleaned.wav", 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_int16.tobytes())
                
                print()
                print("✓ Cleaned audio saved to: input_cleaned.wav")
                print()
                print("=" * 70)
                print("Files created:")
                print(f"  - {output_file} (48 kHz - high quality)")
                print("  - input_cleaned.wav (minimally processed)")
                print("=" * 70)
                print()
                print("💡 The 48 kHz recording should sound clean and clear!")
                
            except Exception as e:
                print(f"⚠️  Could not clean audio: {e}")
                print("   Original recording is still available.")
            
        else:
            print("\n✗ Recording failed. Please try again.")
            
    except KeyboardInterrupt:
        print("\n\n✗ Recording cancelled.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    record_with_airpods()
