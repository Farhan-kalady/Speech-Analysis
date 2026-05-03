import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
from .preprocessor import AudioPreprocessor
from .pause_detector import PauseDetector
from .repetition_detector import RepetitionDetector

def plot_analysis(audio, sr, result):
    """
    Plots the waveform and RMS energy, shading the pause regions.
    Saves the plot to 'output/analysis.png'.
    """
    os.makedirs("output", exist_ok=True)
    
    time_wave = np.linspace(0, len(audio) / sr, num=len(audio))
    
    frame_length = 512
    hop_length = 256
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    frames = range(len(rms))
    time_rms = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(time_wave, audio, color='b', alpha=0.6)
    ax1.set_title(f"Waveform - {os.path.basename(result['file'])}")
    ax1.set_ylabel("Amplitude")
    
    for p in result["pauses"]["pause_list"]:
        ax1.axvspan(p["start"], p["end"], color='red', alpha=0.3)
        
    ax2.plot(time_rms, rms, color='g')
    ax2.set_title("RMS Energy")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Energy")
    
    ax2.axhline(y=0.02, color='r', linestyle='--', label='Threshold (0.02)')
    ax2.legend()
    
    for p in result["pauses"]["pause_list"]:
        ax2.axvspan(p["start"], p["end"], color='red', alpha=0.3)
        
    plt.tight_layout()
    plt.savefig("output/analysis.png", dpi=300)
    plt.close()


class SpeechAnalyzer:
    """
    Orchestrator for speech analysis pipeline.
    """

    def __init__(self, filepath):
        """
        Initializes the SpeechAnalyzer.

        Args:
            filepath (str): Path to the audio file.
        """
        self.filepath = filepath
        self.preprocessor = AudioPreprocessor()
        self.pause_detector = PauseDetector()
        self.repetition_detector = RepetitionDetector()

    def analyze(self):
        """
        Runs the full analysis pipeline.

        Returns:
            dict: Structured result containing pause and repetition statistics.
        """
        # 1. Preprocess
        audio, sr = self.preprocessor.preprocess(self.filepath)
        duration_seconds = float(librosa.get_duration(y=audio, sr=sr))
        
        # 2. Detect Pauses
        pauses = self.pause_detector.detect(audio, sr)
        pause_summary = self.pause_detector.summarize(pauses)
        
        # 3. Detect Repetitions
        repetitions = self.repetition_detector.detect(audio, sr)
        repetition_summary = self.repetition_detector.summarize(repetitions)
        
        # 4. Construct Result
        result = {
            "file": self.filepath,
            "duration_seconds": duration_seconds,
            "pauses": pause_summary,
            "repetitions": repetition_summary
        }
        
        # 5. Plot
        plot_analysis(audio, sr, result)
        
        return result
