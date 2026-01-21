"""
Audio processing utilities for noise reduction and audio enhancement.
"""

import numpy as np
import wave
from scipy import signal
from typing import Tuple
try:
    import noisereduce as nr
    NOISE_REDUCE_AVAILABLE = True
except ImportError:
    NOISE_REDUCE_AVAILABLE = False
    print("Warning: noisereduce not available. Using basic noise reduction.")


class AudioProcessor:
    """Class for processing and cleaning audio recordings."""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Sample rate in Hz (default: 16000)
        """
        self.sample_rate = sample_rate
    
    def load_wav(self, filename: str) -> Tuple[np.ndarray, int]:
        """
        Load a WAV file.
        
        Args:
            filename: Path to WAV file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        with wave.open(filename, 'rb') as wav_file:
            frames = wav_file.readframes(-1)
            audio_data = np.frombuffer(frames, dtype=np.int16)
            sample_rate = wav_file.getframerate()
            
            # Convert to float32
            audio_float = audio_data.astype(np.float32) / 32767.0
            
            return audio_float, sample_rate
    
    def save_wav(self, audio_data: np.ndarray, filename: str, sample_rate: int = None):
        """
        Save audio data to WAV file.
        
        Args:
            audio_data: Audio data as float32 array
            filename: Output filename
            sample_rate: Sample rate (defaults to self.sample_rate)
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        # Ensure audio is in correct range
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        # Convert to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Ensure mono
        if len(audio_int16.shape) > 1:
            audio_int16 = np.mean(audio_int16, axis=1)
        
        # Save to WAV
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
    
    def reduce_noise_advanced(self, audio: np.ndarray) -> np.ndarray:
        """
        Advanced noise reduction using spectral gating and filtering.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Processed audio
        """
        if NOISE_REDUCE_AVAILABLE:
            # Use noisereduce library for better noise reduction
            try:
                # Estimate noise from first 0.3 seconds
                noise_sample_length = int(self.sample_rate * 0.3)
                if len(audio) > noise_sample_length:
                    noise_clip = audio[:noise_sample_length]
                else:
                    noise_clip = audio[:len(audio)//4]
                
                # Apply noise reduction with aggressive settings
                reduced_noise = nr.reduce_noise(
                    y=audio,
                    sr=self.sample_rate,
                    stationary=False,
                    prop_decrease=0.9  # Reduce 90% of noise
                )
                return reduced_noise
            except Exception as e:
                print(f"Warning: Advanced noise reduction failed: {e}")
                return self.reduce_noise_aggressive(audio)
        else:
            return self.reduce_noise_aggressive(audio)
    
    def reduce_noise_aggressive(self, audio: np.ndarray) -> np.ndarray:
        """
        Aggressive noise reduction using multiple techniques.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Processed audio
        """
        # Step 1: Estimate noise from quiet parts
        noise_sample_length = int(self.sample_rate * 0.3)
        if len(audio) > noise_sample_length:
            # Use first 0.3 seconds and quietest 10% of audio
            noise_samples = audio[:noise_sample_length]
            quiet_threshold = np.percentile(np.abs(audio), 10)
            quiet_samples = audio[np.abs(audio) < quiet_threshold]
            noise_estimate = np.mean(np.abs(np.concatenate([noise_samples, quiet_samples])))
        else:
            noise_estimate = np.percentile(np.abs(audio), 5)
        
        # Step 2: Apply aggressive noise gate
        threshold = noise_estimate * 2.5
        audio_cleaned = audio.copy()
        
        # Strongly reduce noise in quiet parts
        mask = np.abs(audio_cleaned) < threshold
        audio_cleaned[mask] = audio_cleaned[mask] * 0.1  # Reduce to 10% in quiet parts
        
        # Step 3: Spectral subtraction for remaining noise
        # Use FFT for frequency-domain processing
        fft = np.fft.rfft(audio_cleaned)
        freqs = np.fft.rfftfreq(len(audio_cleaned), 1/self.sample_rate)
        
        # Reduce frequencies that are likely noise
        # Reduce very low frequencies (below 100 Hz)
        low_freq_mask = freqs < 100
        fft[low_freq_mask] = fft[low_freq_mask] * 0.3
        
        # Reduce very high frequencies (above 6000 Hz)
        high_freq_mask = freqs > 6000
        fft[high_freq_mask] = fft[high_freq_mask] * 0.5
        
        # Convert back to time domain
        audio_cleaned = np.fft.irfft(fft, len(audio_cleaned))
        
        return audio_cleaned
    
    def reduce_noise_simple(self, audio: np.ndarray, noise_reduction_factor: float = 0.7) -> np.ndarray:
        """
        Simple noise reduction using spectral subtraction.
        
        Args:
            audio: Input audio signal
            noise_reduction_factor: How much noise to reduce (0.0 to 1.0)
            
        Returns:
            Processed audio
        """
        # Estimate noise from the first 0.2 seconds (assuming it's mostly silence/noise)
        noise_sample_length = int(self.sample_rate * 0.2)
        if len(audio) > noise_sample_length:
            noise_estimate = np.mean(np.abs(audio[:noise_sample_length]))
        else:
            noise_estimate = np.mean(np.abs(audio)) * 0.1
        
        # Apply noise gate - reduce samples below threshold
        threshold = noise_estimate * (1.5 + noise_reduction_factor)
        audio_cleaned = audio.copy()
        
        # More aggressive noise reduction in quiet parts
        mask = np.abs(audio_cleaned) < threshold
        audio_cleaned[mask] = audio_cleaned[mask] * (1 - noise_reduction_factor * 0.8)
        
        return audio_cleaned
    
    def apply_high_pass_filter(self, audio: np.ndarray, cutoff: float = 100.0) -> np.ndarray:
        """
        Apply high-pass filter to remove low-frequency noise.
        
        Args:
            audio: Input audio signal
            cutoff: Cutoff frequency in Hz (default: 80 Hz)
            
        Returns:
            Filtered audio
        """
        # Design high-pass filter
        nyquist = self.sample_rate / 2
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
        
        # Apply filter
        filtered_audio = signal.filtfilt(b, a, audio)
        
        return filtered_audio
    
    def apply_low_pass_filter(self, audio: np.ndarray, cutoff: float = 7000.0) -> np.ndarray:
        """
        Apply low-pass filter to remove high-frequency noise.
        
        Args:
            audio: Input audio signal
            cutoff: Cutoff frequency in Hz (default: 7000 Hz, must be < Nyquist)
            
        Returns:
            Filtered audio
        """
        # Design low-pass filter
        nyquist = self.sample_rate / 2
        # Ensure cutoff is below Nyquist
        cutoff = min(cutoff, nyquist * 0.95)
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
        
        # Apply filter
        filtered_audio = signal.filtfilt(b, a, audio)
        
        return filtered_audio
    
    def normalize_audio(self, audio: np.ndarray, target_level: float = 0.8) -> np.ndarray:
        """
        Normalize audio to a target level.
        
        Args:
            audio: Input audio signal
            target_level: Target peak level (0.0 to 1.0)
            
        Returns:
            Normalized audio
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            normalization_factor = target_level / max_val
            audio_normalized = audio * normalization_factor
        else:
            audio_normalized = audio
        
        return audio_normalized
    
    def clean_audio(self, audio: np.ndarray, 
                   apply_high_pass: bool = True,
                   apply_low_pass: bool = True,
                   reduce_noise: bool = True,
                   normalize: bool = True) -> np.ndarray:
        """
        Apply multiple cleaning steps to audio.
        
        Args:
            audio: Input audio signal
            apply_high_pass: Apply high-pass filter to remove low-frequency noise
            apply_low_pass: Apply low-pass filter to remove high-frequency noise
            reduce_noise: Apply noise reduction
            normalize: Normalize audio levels
            
        Returns:
            Cleaned audio
        """
        cleaned = audio.copy()
        
        # Step 1: High-pass filter (remove low-frequency noise like rumble)
        if apply_high_pass:
            cleaned = self.apply_high_pass_filter(cleaned, cutoff=100.0)
        
        # Step 2: Low-pass filter (remove high-frequency noise)
        if apply_low_pass:
            cleaned = self.apply_low_pass_filter(cleaned, cutoff=6000.0)
        
        # Step 3: Advanced noise reduction
        if reduce_noise:
            cleaned = self.reduce_noise_advanced(cleaned)
        
        # Step 4: Normalize
        if normalize:
            cleaned = self.normalize_audio(cleaned, target_level=0.8)
        
        return cleaned
    
    def process_file(self, input_file: str, output_file: str, **kwargs):
        """
        Process an audio file and save the cleaned version.
        
        Args:
            input_file: Input WAV file path
            output_file: Output WAV file path
            **kwargs: Additional arguments for clean_audio()
        """
        # Load audio
        audio, sample_rate = self.load_wav(input_file)
        
        # Clean audio
        cleaned_audio = self.clean_audio(audio, **kwargs)
        
        # Save cleaned audio
        self.save_wav(cleaned_audio, output_file, sample_rate)
        
        return cleaned_audio
