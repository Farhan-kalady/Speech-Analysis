import sys
import json
import os
from src.analyzer import SpeechAnalyzer

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_audio_file>")
        sys.exit(1)
        
    filepath = sys.argv[1]
    
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
        
    print(f"Analyzing {filepath}...")
    analyzer = SpeechAnalyzer(filepath)
    result = analyzer.analyze()
    
    # Save to JSON
    os.makedirs("output", exist_ok=True)
    with open("output/result.json", "w") as f:
        json.dump(result, f, indent=4)
        
    # Pretty print to console
    print("\n=== Speech Analysis Report ===")
    filename = os.path.basename(result["file"])
    print(f"File: {filename} | Duration: {result['duration_seconds']:.1f}s")
    
    print("\n--- Pause Segments ---")
    for p in result["pauses"]["pause_list"]:
        print(f"[{p['start']:.1f}s – {p['end']:.1f}s] duration: {p['duration']:.1f}s")
    print(f"Total Pause Duration: {result['pauses']['total_duration']:.1f}s")
    
    print("\n--- Repetition Segments ---")
    for r in result["repetitions"]["repetition_list"]:
        print(f"[{r['start']:.1f}s – {r['end']:.1f}s] similarity: {r['similarity']:.2f} | count: {r['count']}")
    print(f"Total Repetitions Detected: {result['repetitions']['total_repetitions']}")
    
if __name__ == "__main__":
    main()
