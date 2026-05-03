import unittest
import numpy as np
import os
from scipy.io import wavfile
import tempfile

from src.preprocessor import AudioPreprocessor
from src.pause_detector import PauseDetector
from src.repetition_detector import RepetitionDetector

class TestSpeechAnalysis(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_wav_path = os.path.join(self.temp_dir.name, "test_audio.wav")
        
        self.sr = 16000
        t = np.linspace(0, 1, self.sr, endpoint=False)
        self.sine_wave = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        
        wavfile.write(self.temp_wav_path, self.sr, self.sine_wave)
        
    def tearDown(self):
        self.temp_dir.cleanup()

    def test_preprocessor_loads_audio(self):
        preprocessor = AudioPreprocessor()
        audio, sr = preprocessor.load(self.temp_wav_path)
        self.assertIsInstance(audio, np.ndarray)
        self.assertIsInstance(sr, int)
        self.assertEqual(sr, 16000)

    def test_pause_detector_returns_list(self):
        detector = PauseDetector(energy_threshold=0.02, min_pause_duration=0.1)
        silence = np.zeros(16000, dtype=np.float32)
        pauses = detector.detect(silence, 16000)
        self.assertIsInstance(pauses, list)
        self.assertGreater(len(pauses), 0)

    def test_repetition_detector_returns_list(self):
        detector = RepetitionDetector(segment_duration=0.3, hop_duration=0.1, similarity_threshold=0.85)
        t = np.linspace(0, 2, 32000, endpoint=False)
        audio = (np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        repetitions = detector.detect(audio, 16000)
        self.assertIsInstance(repetitions, list)

if __name__ == "__main__":
    unittest.main()
