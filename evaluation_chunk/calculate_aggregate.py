import json
import os
import argparse
from typing import Dict, Any

# Configuration matching run_evaluation.py
OUTPUT_ROOT = "/home/wangyaqi/jst/evaluation_chunk/output"
DEFAULT_REPORT_PATH = os.path.join(OUTPUT_ROOT, "evaluation_report.jsonl")

def calculate_weighted_aggregate(report_path: str = DEFAULT_REPORT_PATH):
    if not os.path.exists(report_path):
        print(f"Report file not found: {report_path}")
        return

    # Structure: { strategy_name: { weighted_score_sum: float, total_chunks: int, total_time: float, file_count: int } }
    stats = {}
    valid_lines = 0

    print(f"Reading report from: {report_path}")
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc_result = json.loads(line)
                    valid_lines += 1
                except json.JSONDecodeError:
                    continue

                strategies = doc_result.get("strategies", {})
                for name, data in strategies.items():
                    if name not in stats:
                        stats[name] = {
                            "weighted_score_sum": 0.0,
                            "total_chunks": 0,
                            "total_time": 0.0,
                            "file_count": 0
                        }
                    
                    count = data.get("chunk_count", 0)
                    score = data.get("score", 0)
                    time_sec = data.get("time_sec", 0)
                    
                    # Weight score by chunk count
                    stats[name]["weighted_score_sum"] += score * count
                    stats[name]["total_chunks"] += count
                    stats[name]["total_time"] += time_sec
                    stats[name]["file_count"] += 1
                    
    except Exception as e:
        print(f"Error reading report: {e}")
        return

    if valid_lines == 0:
        print("No valid data found in report.")
        return

    print("\n=== Weighted Aggregate Results (Weighted by Chunk Count) ===")
    print(f"{'Strategy':<15} | {'Avg Score':<12} | {'Avg Time/File':<15} | {'Total Chunks':<12} | {'Files':<5}")
    print("-" * 80)

    for name, data in stats.items():
        total_chunks = data["total_chunks"]
        if total_chunks > 0:
            avg_score = data["weighted_score_sum"] / total_chunks
        else:
            avg_score = 0.0
            
        file_count = data["file_count"]
        avg_time = data["total_time"] / file_count if file_count > 0 else 0.0
        
        print(f"{name:<15} | {avg_score:<12.4f} | {avg_time:<15.4f} | {total_chunks:<12} | {file_count:<5}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=DEFAULT_REPORT_PATH, help="Path to evaluation report jsonl")
    args = parser.parse_args()
    
    calculate_weighted_aggregate(args.path)

if __name__ == "__main__":
    main()

