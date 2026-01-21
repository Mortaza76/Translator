"""
Clean voice recording using optimized settings similar to iMessage.
Uses higher sample rate and better audio processing.
"""

from record_audio import AudioRecorder
import sounddevice as sd
import numpy as np
import wave
import time
import sys


def record_clean_voice():
    """Record with settings optimized for clean voice (like iMessage)."""
    print("=" * 70)
    print("🎤 Clean Voice Recording (iMessage-style)")
    print("=" * 70)
    print()
    
    # Find AirPods
    devices = sd.query_devices()
    airpods_device = None
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            name_lower = device['name'].lower()
            if 'airpod' in name_lower:
                airpods_device = i
                device_name = device['name']
                device_sample_rate = device['default_samplerate']
                print(f"✓ Found: {device_name}")
                print(f"  Native sample rate: {device_sample_rate} Hz")
                break
    
    if airpods_device is None:
        print("✗ AirPods not found")
        return
    
    print()
    print("Using optimized settings for clean voice:")
    print("  - Higher sample rate (48 kHz - like iMessage)")
    print("  - Better audio processing")
    print("  - Automatic gain control")
    print()
    
    # Use higher sample rate (iMessage uses 48 kHz)
    sample_rate = 48000
    duration = 5.0
    output_file = "input.wav"
    
    print("Ready to record!")
    print()
    input("Press Enter when you're ready to start recording...")
    print()
    
    print("Recording will start in 3 seconds...")
    print("Get ready to speak!")
    print()
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"⏰ Starting in {i}...", end='\r')
        time.sleep(1)
    print()
    print()
    
    print("=" * 70)
    print("🔴 RECORDING NOW - Speak clearly!")
    print("=" * 70)
    print()
    
    try:
        # Record at higher sample rate (48 kHz like iMessage)
        print(f"Recording at {sample_rate} Hz (high quality)...")
        
        # Record directly with sounddevice for better control
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32',
            device=airpods_device
        )
        sd.wait()  # Wait until recording is finished
        
        print("✓ Recording complete!")
        print()
        
        # Process audio for cleaner output
        print("Processing audio...")
        audio = recording.flatten()
        
        # Remove DC offset
        audio = audio - np.mean(audio)
        
        # Gentle high-pass to remove rumble (less aggressive)
        from scipy import signal
        nyquist = sample_rate / 2
        b, a = signal.butter(2, 80 / nyquist, btype='high')
        audio = signal.filtfilt(b, a, audio)
        
        # Normalize (but not too aggressive)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            # Normalize to 90% to avoid clipping
            audio = audio * (0.9 / max_val)
        
        # Convert to int16 for WAV
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Save at original high sample rate
        with wave.open(output_file, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        print(f"✓ Saved to: {output_file}")
        print(f"  Sample rate: {sample_rate} Hz (high quality)")
        print(f"  Format: 16-bit PCM, Mono")
        print()
        
        # Also create a 16 kHz version for compatibility
        print("Creating 16 kHz version for compatibility...")
        # Resample to 16 kHz if needed for speech processing
        from scipy import signal as scipy_signal
        num_samples_16k = int(len(audio) * 16000 / sample_rate)
        audio_16k = scipy_signal.resample(audio, num_samples_16k)
        audio_16k_int16 = (audio_16k * 32767).astype(np.int16)
        
        output_16k = "input_16k.wav"
        with wave.open(output_16k, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(audio_16k_int16.tobytes())
        
        print(f"✓ Also saved: {output_16k} (16 kHz for compatibility)")
        print()
        print("=" * 70)
        print("✅ Recording Complete!")
        print("=" * 70)
        print()
        print("Files created:")
        print(f"  - {output_file} (48 kHz - high quality, like iMessage)")
        print(f"  - {output_16k} (16 kHz - for speech processing)")
        print()
        print("💡 The 48 kHz version should sound much cleaner!")
        print("   Play it and compare with your iMessage recording.")
        
    except KeyboardInterrupt:
        print("\n\n✗ Recording cancelled.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    record_clean_voice()
