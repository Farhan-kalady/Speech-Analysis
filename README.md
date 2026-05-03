#  🎤 Speech Analysis: Pause & Repetition Detection

A Python-based system that analyzes speech audio to detect:
- ⏸️ Silent pauses in speech
- 🔁 Acoustic repetition patterns (stuttering) using MFCC similarity
- 🗣️ Word-level repetitions (e.g., "I I I") using SpeechRecognition

This project combines signal processing and speech recognition to solve a real-world speech analysis problem.

---

## 🚀 Features

- Detects pause segments with start/end timestamps
- Calculates total pause duration and pause ratio
- Identifies repeated acoustic patterns using MFCC cosine similarity
- Detects repeated words using Google SpeechRecognition API
- Auto-calibrates energy threshold per audio file
- Modular Python implementation across separate detector files

---

## 🔄 System Pipeline

```
Audio Input
    └── Preprocessor (librosa load → mono 16kHz → normalize → noise removal)
         └── Feature Extractor (MFCC, RMS Energy, Mel-Spectrogram)
              ├── Pause Detector   → Silent regions by RMS thresholding
              └── Repetition Detector → Acoustic (MFCC cosine similarity)
                                      → Word-level (SpeechRecognition transcription)
                                           └── Structured Output (console + result.json)
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/speech-analysis.git
cd speech-analysis
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python main.py sample_audio/your_file.wav
```

---

## 🎬 Demo

https://drive.google.com/file/d/1AIJTEHF9ie7RPSuR1TasgGJ5BlDymBp2/view?usp=sharing

---

## 📊 Sample Output

```
🎧 AUDIO ANALYSIS RESULTS
File: test_sample.wav

⏸️ Pause Segments:
  [0.00s – 0.26s]  duration: 0.26s
  [0.58s – 0.96s]  duration: 0.38s
⏱️  Total Pause Duration: 1.95s
    Pause Ratio: 22.4% of audio

🔁 Acoustic Repetitions:
  Detected pattern at: [0.6s – 1.2s]
  Repetition Count: 2

  Detected pattern at: [1.8s – 2.4s]
  Repetition Count: 2

🗣️ Word-Level Repetitions:
  Word: "i"   Count: 3

=== Calibration Summary ===
Energy threshold used : 0.0041
Pause ratio           : 22.4% of audio
Repetition events     : 2
Avg similarity score  : 0.93
```

---

## 🧠 Approach

### 1. Audio Preprocessing (`src/preprocessor.py`)
- Audio loaded using `librosa` at 16000 Hz mono
- Peak normalization applied to range [-1, 1]
- Noise reduction using `noisereduce` with stationary mode

### 2. Feature Extraction (`src/feature_extractor.py`)
- **MFCC** — 13 Mel-Frequency Cepstral Coefficients per frame
- **RMS Energy** — per-frame root mean square energy
- **Mel-Spectrogram** — converted to dB scale

### 3. Pause Detection (`src/pause_detector.py`)
- RMS energy computed per frame (hop=256 samples)
- Threshold auto-calibrated to 5th percentile RMS × 1.5
- Consecutive silent frames merged into pause segments
- Segments shorter than 0.3s filtered out (consonants, not silence)
- Sanity check: warns if pause ratio exceeds 60%

### 4. Acoustic Repetition Detection (`src/repetition_detector.py`)
- Audio split into 0.4s windows with 0.2s hop (50% overlap)
- Mean MFCC vector computed per window
- Cosine similarity compared between consecutive windows
- Windows with similarity ≥ 0.92 grouped into repetition events
- Energy gate: windows with RMS < 0.01 skipped
- Count filter: only events with 2–5 repeats kept

### 5. Word-Level Repetition Detection (`src/analyzer.py`)
- Audio transcribed using `SpeechRecognition` (Google API)
- Transcribed words scanned for consecutive duplicates
- Repeated words reported with count

---

## 📁 Project Structure

```
speech-analysis/
├── src/
│   ├── __init__.py
│   ├── preprocessor.py          # Audio loading, normalization, noise removal
│   ├── feature_extractor.py     # MFCC, RMS energy, spectrogram
│   ├── pause_detector.py        # RMS thresholding, auto-calibration
│   ├── repetition_detector.py   # MFCC cosine similarity, energy gate
│   └── analyzer.py              # Orchestrator + word-level detection
├── tests/
│   └── test_basic.py
├── sample_audio/
│   └── test_sample.wav
├── output/
│   ├── result.json
│   └── analysis.png
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚠️ Challenges

- Choosing optimal silence threshold — solved with auto-calibration per file
- Overlapping windows causing false repetitions — solved by reducing to 50% overlap
- Synthetic test audio giving misleading results — solved by generating realistic test signal
- Word detection accuracy depends on internet and audio clarity

---

## 🚀 Future Improvements

- Integrate OpenAI Whisper for offline, higher-accuracy transcription
- Add real-time microphone input support
- Add waveform + RMS visualization dashboard

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `librosa` | Audio loading, MFCC, mel-spectrogram |
| `noisereduce` | Stationary noise removal |
| `numpy` / `scipy` | Signal processing, cosine similarity |
| `SpeechRecognition` | Word-level transcription |
| `matplotlib` | Waveform and RMS visualization |

---

## 💡 Notes

- Works best with `.wav` files sampled at 16kHz
- Word-level detection requires an active internet connection
- Detection accuracy depends on audio quality and background noise

---

## 👤 Author

**Mohammed Farhan K**

---
