import librosa
from .preprocessor import AudioPreprocessor
from .pause_detector import PauseDetector
from .repetition_detector import RepetitionDetector

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
        
        return result
