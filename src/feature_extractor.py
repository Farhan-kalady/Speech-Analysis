import librosa
import numpy as np

class FeatureExtractor:
    """
    Extracts acoustic features from audio signals.
    """

    def extract_mfcc(self, audio, sr, n_mfcc=13):
        """
        Extracts Mel-frequency cepstral coefficients (MFCCs).

        Args:
            audio (np.ndarray): Audio time series.
            sr (int): Sample rate.
            n_mfcc (int): Number of MFCCs to extract.

        Returns:
            np.ndarray: MFCC matrix.
        """
        return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    def extract_rms_energy(self, audio, frame_length=512, hop_length=256):
        """
        Extracts Root Mean Square (RMS) energy per frame.

        Args:
            audio (np.ndarray): Audio time series.
            frame_length (int): Length of the frame.
            hop_length (int): Number of samples between successive frames.

        Returns:
            np.ndarray: RMS energy array.
        """
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)
        return rms[0]

    def extract_spectrogram(self, audio, sr):
        """
        Extracts mel-spectrogram in decibels (dB).

        Args:
            audio (np.ndarray): Audio time series.
            sr (int): Sample rate.

        Returns:
            np.ndarray: Mel-spectrogram in dB.
        """
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return mel_spectrogram_db

    def frames_to_time(self, frames, sr, hop_length=256):
        """
        Converts frame indices to time in seconds.

        Args:
            frames (int or np.ndarray): Frame index or array of frame indices.
            sr (int): Sample rate.
            hop_length (int): Number of samples between successive frames.

        Returns:
            float or np.ndarray: Time in seconds.
        """
        return librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
