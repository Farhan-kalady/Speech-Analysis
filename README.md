# Speech Analysis: Pause & Repetition Detection

## 1. Project Overview
This project is an automated speech analysis tool designed to detect both **pauses** (silences/hesitations) and **repetitions** (stuttering) in speech audio files. The system processes raw audio and yields a structured report indicating precisely when pauses or repeated segments occur, along with their duration and similarity.

## 2. Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd speech-analysis
   ```

2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 3. Usage

Run the main analysis script by providing the path to an audio file (e.g., `.wav`, `.mp3`):
```bash
python main.py path/to/audio.wav
```
The script will output a structured report to the console and save a JSON result file in the `output/` directory.

## 4. Approach Explanation

- **Preprocessing:** Audio is loaded using `librosa`, downsampled to 16 kHz, and converted to mono. It is then peak-normalized to the range `[-1, 1]`, and stationary background noise is removed using the `noisereduce` library to improve feature extraction quality.
- **Feature Extraction:** Extracts 13 Mel-frequency cepstral coefficients (MFCCs), Root Mean Square (RMS) energy, and mel-spectrograms (in dB) using `librosa`.
- **Pause Detection:** Operates by thresholding the RMS energy on a per-frame basis. Consecutive silent frames are merged into contiguous segments. If a segment exceeds the `min_pause_duration` threshold, it is logged as a pause.
- **Repetition Detection:** Splits the audio into overlapping temporal windows. For each window, the mean MFCC vector over time is computed. Consecutive windows are compared using cosine similarity. Sequences of highly similar, consecutive windows are flagged as repetitions.

## 5. Output Format

The analyzer outputs a `result.json` file in the following schema format:
```json
{
    "file": "path/to/audio.wav",
    "duration_seconds": 5.3,
    "pauses": {
        "total_pauses": 2,
        "total_duration": 1.3,
        "pause_list": [
            {"start": 0.4, "end": 1.1, "duration": 0.7},
            {"start": 2.3, "end": 2.9, "duration": 0.6}
        ]
    },
    "repetitions": {
        "total_repetitions": 1,
        "repetition_list": [
            {"start": 1.2, "end": 1.8, "similarity": 0.91, "count": 2}
        ]
    }
}
```

## 6. Challenges Faced

- **Noise Sensitivity:** Varying levels of background noise severely impacted energy-based pause detection. Applying spectral gating with `noisereduce` was necessary.
- **Threshold Tuning:** Finding the right balance for RMS thresholds (`0.02`) and cosine similarity thresholds (`0.85`) across different speakers and recording qualities required empirical tuning.
- **Granularity (Syllable vs. Word-level):** Detecting stuttering using mean MFCCs works decently for phoneme/syllable repetitions but can sometimes yield false positives for sustained vowels or slow speech.

## 7. Dataset References

To train or tune thresholds, these datasets are commonly used in the field:
- **UCLASS (University College London Archive of Stuttered Speech):** Specialized dataset containing stuttered speech.
- **LibriSpeech:** Large corpus of read English speech, useful for establishing baseline non-stuttered speech characteristics.
