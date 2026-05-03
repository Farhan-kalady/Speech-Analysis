import numpy as np
from .feature_extractor import FeatureExtractor

class PauseDetector:
    """
    Detects pauses in speech audio based on RMS energy thresholding.
    """

    def __init__(self, energy_threshold=0.02, min_pause_duration=0.3):
        """
        Initializes the PauseDetector.

        Args:
            energy_threshold (float): RMS energy threshold below which a frame is considered silent.
            min_pause_duration (float): Minimum duration in seconds to be considered a pause.
        """
        self.energy_threshold = energy_threshold
        self.min_pause_duration = min_pause_duration
        self.feature_extractor = FeatureExtractor()
        self.total_audio_duration = 0.0

    def auto_calibrate(self, audio, sr):
        frame_length = 512
        hop_length = 256
        rms = self.feature_extractor.extract_rms_energy(audio, frame_length=frame_length, hop_length=hop_length)
        threshold = np.percentile(rms, 5) * 1.5
        return threshold, rms

    def detect(self, audio, sr):
        """
        Detects pause segments in the audio.

        Args:
            audio (np.ndarray): Audio time series.
            sr (int): Sample rate.

        Returns:
            list: List of dictionaries containing start, end, and duration of pauses.
        """
        hop_length = 256
        frame_length = 512
        
        self.energy_threshold, rms = self.auto_calibrate(audio, sr)
        self.min_pause_duration = 0.3
        max_pause_duration = 3.0
        
        self.total_audio_duration = len(audio) / sr
        
        silent_frames = rms < self.energy_threshold
        
        pauses = []
        in_pause = False
        start_frame = 0
        
        for i, is_silent in enumerate(silent_frames):
            if is_silent and not in_pause:
                in_pause = True
                start_frame = i
            elif not is_silent and in_pause:
                in_pause = False
                end_frame = i
                
                start_time = float(self.feature_extractor.frames_to_time(start_frame, sr, hop_length=hop_length))
                end_time = float(self.feature_extractor.frames_to_time(end_frame, sr, hop_length=hop_length))
                duration = end_time - start_time
                
                if self.min_pause_duration <= duration <= max_pause_duration:
                    pauses.append({
                        "start": start_time,
                        "end": end_time,
                        "duration": duration
                    })
                    
        if in_pause:
            end_frame = len(silent_frames)
            start_time = float(self.feature_extractor.frames_to_time(start_frame, sr, hop_length=hop_length))
            end_time = float(self.feature_extractor.frames_to_time(end_frame, sr, hop_length=hop_length))
            duration = end_time - start_time
            if self.min_pause_duration <= duration <= max_pause_duration:
                pauses.append({
                    "start": start_time,
                    "end": end_time,
                    "duration": duration
                })
        
        total_pause_duration = sum(p["duration"] for p in pauses)
        pause_ratio = total_pause_duration / self.total_audio_duration
        if pause_ratio > 0.60:
            print("WARNING: Over 60% of audio flagged as silence. Check audio quality.")
                
        return pauses

    def summarize(self, pauses):
        """
        Summarizes the detected pauses.

        Args:
            pauses (list): List of pause dictionaries from detect().

        Returns:
            dict: Summary statistics including total_pauses, total_duration, pause_list, and pause_ratio.
        """
        total_pauses = len(pauses)
        total_duration = sum(p["duration"] for p in pauses)
        pause_ratio = (total_duration / self.total_audio_duration) * 100 if self.total_audio_duration > 0 else 0.0
        return {
            "total_pauses": total_pauses,
            "total_duration": total_duration,
            "pause_ratio": round(pause_ratio, 1),
            "pause_list": pauses,
            "energy_threshold": float(self.energy_threshold)
        }
