import numpy as np
from scipy.io import wavfile
import os

def generate_realistic_stutter():
    sample_rate = 16000
    duration = 6.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # 1. Base signal: mix of 3 sine waves at 200Hz, 400Hz, 800Hz
    f1, f2, f3 = 200, 400, 800
    base_signal = (np.sin(2 * np.pi * f1 * t) + 
                   0.5 * np.sin(2 * np.pi * f2 * t) + 
                   0.25 * np.sin(2 * np.pi * f3 * t))

    envelope = np.zeros_like(t)
    chunk_samples = int(0.15 * sample_rate)
    
    # Generate random syllables (on/off every 0.15s)
    # Speech from 0.0 to 1.0
    for i in range(0, int(1.0 * sample_rate), chunk_samples * 2):
        end_idx = min(i + chunk_samples, len(t))
        envelope[i:end_idx] = 1.0

    # Stutter at 1.0s to 1.6s
    # Repeat a 0.15s syllable chunk 3 times in a row at 1.0s mark
    # To make sure they are identical, we don't just use envelope, we actually copy the signal segment later.
    # But for now let's just make the envelope 1.0
    for i in range(3):
        start = int(1.0 * sample_rate) + i * int(0.20 * sample_rate) # 0.15 chunk + 0.05 gap
        end = start + chunk_samples
        envelope[start:end] = 1.0
        
    # Speech from 1.6s to 2.1s
    for i in range(int(1.6 * sample_rate), int(2.1 * sample_rate), chunk_samples * 2):
        end_idx = min(i + chunk_samples, len(t))
        envelope[i:end_idx] = 1.0
        
    # Pause 1: 2.1s to 2.6s (0.5s)
    # Envelope remains 0
    
    # Speech from 2.6s to 3.5s
    for i in range(int(2.6 * sample_rate), int(3.5 * sample_rate), chunk_samples * 2):
        end_idx = min(i + chunk_samples, len(t))
        envelope[i:end_idx] = 1.0
        
    # Pause 2: 3.5s to 4.0s (0.5s)
    
    # Speech from 4.0s to 4.9s
    for i in range(int(4.0 * sample_rate), int(4.9 * sample_rate), chunk_samples * 2):
        end_idx = min(i + chunk_samples, len(t))
        envelope[i:end_idx] = 1.0
        
    # Pause 3: 4.9s to 5.4s (0.5s)
    
    # Speech from 5.4s to 6.0s
    for i in range(int(5.4 * sample_rate), int(6.0 * sample_rate), chunk_samples * 2):
        end_idx = min(i + chunk_samples, len(t))
        envelope[i:end_idx] = 1.0

    signal = base_signal * envelope

    # Now let's enforce exact repetitions at 1.0s
    stutter_chunk = signal[0:chunk_samples].copy()
    for i in range(3):
        start = int(1.0 * sample_rate) + i * int(0.20 * sample_rate)
        end = start + chunk_samples
        signal[start:end] = stutter_chunk

    # Add Gaussian noise (SNR 20dB)
    signal_power = np.mean(signal**2)
    signal_power = max(signal_power, 1e-10)
    noise_power = signal_power / (10 ** (20 / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    
    signal_with_noise = signal + noise

    # Normalize to int16 range
    max_val = np.max(np.abs(signal_with_noise))
    if max_val > 0:
        signal_normalized = np.int16((signal_with_noise / max_val) * 32767)
    else:
        signal_normalized = np.int16(signal_with_noise)

    os.makedirs("sample_audio", exist_ok=True)
    wavfile.write("sample_audio/realistic_test.wav", sample_rate, signal_normalized)
    print("Successfully generated sample_audio/realistic_test.wav")

if __name__ == "__main__":
    generate_realistic_stutter()
