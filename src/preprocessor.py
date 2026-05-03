import librosa
import numpy as np
import noisereduce as nr

class AudioPreprocessor:
    """
    Handles loading and preprocessing of audio files for speech analysis.
    """

    def load(self, filepath):
        """
        Loads an audio file and resamples it to 16000 Hz mono.

        Args:
            filepath (str): Path to the audio file.

        Returns:
            tuple: (audio_time_series, sample_rate)
        """
        audio, sr = librosa.load(filepath, sr=16000, mono=True)
        return audio, sr

    def normalize(self, audio):
        """
        Peak normalizes the audio signal to the range [-1, 1].

        Args:
            audio (np.ndarray): Audio time series.

        Returns:
            np.ndarray: Normalized audio time series.
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio

    def remove_noise(self, audio, sr):
        """
        Removes background noise using spectral gating.

        Args:
            audio (np.ndarray): Audio time series.
            sr (int): Sample rate of the audio.

        Returns:
            np.ndarray: Noise-reduced audio time series.
        """
        return nr.reduce_noise(y=audio, sr=sr, stationary=True)

    def preprocess(self, filepath):
        """
        Runs the full preprocessing pipeline: load -> normalize -> remove_noise.

        Args:
            filepath (str): Path to the audio file.

        Returns:
            tuple: (clean_audio, sample_rate)
        """
        audio, sr = self.load(filepath)
        audio_norm = self.normalize(audio)
        clean_audio = self.remove_noise(audio_norm, sr)
        return clean_audio, sr
