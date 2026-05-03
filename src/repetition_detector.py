import numpy as np
from .feature_extractor import FeatureExtractor

class RepetitionDetector:
    """
    Detects repetitions in speech audio using MFCC cosine similarity on overlapping windows.
    """

    def __init__(self, segment_duration=0.4, hop_duration=0.2, similarity_threshold=0.92):
        """
        Initializes the RepetitionDetector.

        Args:
            segment_duration (float): Duration of each window in seconds.
            hop_duration (float): Step size between consecutive windows in seconds.
            similarity_threshold (float): Minimum cosine similarity to be considered a repetition.
        """
        self.segment_duration = segment_duration
        self.hop_duration = hop_duration
        self.similarity_threshold = similarity_threshold
        self.feature_extractor = FeatureExtractor()

    def _cosine_similarity(self, vec1, vec2):
        """
        Computes cosine similarity between two vectors.

        Args:
            vec1 (np.ndarray): First vector.
            vec2 (np.ndarray): Second vector.

        Returns:
            float: Cosine similarity score.
        """
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def detect(self, audio, sr):
        """
        Detects repetitions in the audio.

        Args:
            audio (np.ndarray): Audio time series.
            sr (int): Sample rate.

        Returns:
            list: List of dictionaries containing start, end, similarity, and count of repetitions.
        """
        segment_samples = int(self.segment_duration * sr)
        hop_samples = int(self.hop_duration * sr)
        
        if len(audio) < segment_samples:
            return []
            
        windows = []
        start_times = []
        for i in range(0, len(audio) - segment_samples + 1, hop_samples):
            window = audio[i:i + segment_samples]
            rms = np.sqrt(np.mean(window**2))
            if rms < 0.01:
                continue
            mfcc = self.feature_extractor.extract_mfcc(window, sr)
            mfcc_mean = np.mean(mfcc, axis=1)
            windows.append(mfcc_mean)
            start_times.append(i / sr)
            
        repetitions = []
        current_rep = None
        
        for i in range(len(windows) - 1):
            if abs((start_times[i+1] - start_times[i]) - self.hop_duration) > 1e-4:
                if current_rep is not None:
                    current_rep["similarity"] = current_rep["sim_sum"] / (current_rep["count"] - 1)
                    del current_rep["sim_sum"]
                    repetitions.append(current_rep)
                    current_rep = None
                continue
                
            sim = self._cosine_similarity(windows[i], windows[i+1])
            is_identical = sim == 1.0
            
            if sim >= self.similarity_threshold and not is_identical:
                if current_rep is None:
                    current_rep = {
                        "start": float(start_times[i]),
                        "end": float(start_times[i+1] + self.segment_duration),
                        "similarity": float(sim),
                        "count": 2,
                        "sim_sum": float(sim)
                    }
                else:
                    current_rep["end"] = float(start_times[i+1] + self.segment_duration)
                    current_rep["count"] += 1
                    current_rep["sim_sum"] += float(sim)
            else:
                if current_rep is not None:
                    current_rep["similarity"] = current_rep["sim_sum"] / (current_rep["count"] - 1)
                    del current_rep["sim_sum"]
                    repetitions.append(current_rep)
                    current_rep = None
                    
        if current_rep is not None:
            current_rep["similarity"] = current_rep["sim_sum"] / (current_rep["count"] - 1)
            del current_rep["sim_sum"]
            repetitions.append(current_rep)
            
        filtered_reps = [r for r in repetitions if 2 <= r["count"] <= 5]
        
        merged_reps = []
        for rep in filtered_reps:
            if not merged_reps:
                merged_reps.append(rep)
            else:
                last_rep = merged_reps[-1]
                if rep["start"] - last_rep["end"] < 0.5:
                    last_rep["end"] = max(last_rep["end"], rep["end"])
                    last_rep["count"] += rep["count"]
                    last_rep["similarity"] = (last_rep["similarity"] + rep["similarity"]) / 2.0
                else:
                    merged_reps.append(rep)
                    
        return merged_reps

    def summarize(self, repetitions):
        """
        Summarizes the detected repetitions.

        Args:
            repetitions (list): List of repetition dictionaries from detect().

        Returns:
            dict: Summary statistics including total_repetitions and repetition_list.
        """
        for rep in repetitions:
            if rep["count"] == 2:
                rep["note"] = "possible stutter"
            elif rep["count"] >= 3:
                rep["note"] = "likely stutter"
                
        return {
            "total_repetitions": len(repetitions),
            "repetition_list": repetitions
        }

