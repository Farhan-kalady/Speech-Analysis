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
    print(f"Pause ratio: {result['pauses']['pause_ratio']}% of total audio")
    
    print("\n--- Repetition Segments ---")
    for r in result["repetitions"]["repetition_list"]:
        print(f"[{r['start']:.1f}s – {r['end']:.1f}s] similarity: {r['similarity']:.2f} | count: {r['count']}")
    print(f"Total Repetitions Detected: {result['repetitions']['total_repetitions']}")
    
    # Healthy expected ranges:
    # - Pause ratio: 10% – 40% (not 94%)
    # - Rep count per event: 2 – 5 (not 6+)
    # - Rep segment length: 0.2s – 0.8s
    
    print("\n=== Calibration Summary ===")
    print(f"Energy threshold used : {result['pauses'].get('energy_threshold', 0.0):.4f}")
    print(f"Pause ratio           : {result['pauses'].get('pause_ratio', 0.0)}% of audio")
    print(f"Repetition events     : {result['repetitions']['total_repetitions']}")
    print(f"Avg similarity score  : {result['repetitions'].get('avg_similarity', 0.0):.2f}")
    
    pause_ratio = result['pauses'].get('pause_ratio', 0.0)
    any_rep_count_high = any(r['count'] > 5 for r in result['repetitions']['repetition_list'])
    
    if pause_ratio > 60.0 or any_rep_count_high:
        print("\033[91mALERT: Output looks unrealistic. Try with a different audio file or check preprocessing.\033[0m")
        
if __name__ == "__main__":
    main()
