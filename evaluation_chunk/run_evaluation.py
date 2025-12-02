import os
import json
import argparse
import time
from typing import List, Dict, Any
import numpy as np

from strategies import LengthStrategy, LevelStrategy, SemanticStrategy
from utils import load_document_nodes
try:
    from chunk_similarity_service import evaluate_single_file
except ImportError:
    from .chunk_similarity_service import evaluate_single_file

# Configuration
INPUT_ROOT = "/home/wangyaqi/jst/金盘上市公告_mineru解析/"
OUTPUT_ROOT = "/home/wangyaqi/jst/evaluation_chunk/output"

def find_files(root: str) -> List[str]:
    files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith("_content_list.json"):
                files.append(os.path.join(dirpath, f))
    return files

def save_chunks(chunks: List[str], strategy_name: str, doc_name: str):
    out_dir = os.path.join(OUTPUT_ROOT, strategy_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{doc_name}.jsonl")
    
    with open(out_path, "w", encoding="utf-8") as f:
        for i, c in enumerate(chunks):
            item = {
                "id": f"{doc_name}_{i}",
                "content": c
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def evaluate_chunks(chunks: List[str]) -> float:
    """Calculate score for a list of chunks. Lower is better."""
    if not chunks:
        return 0.0
    try:
        return evaluate_single_file(chunks)
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files to process")
    args = parser.parse_args()

    files = find_files(INPUT_ROOT)
    if args.limit > 0:
        files = files[:args.limit]

    print(f"Found {len(files)} files to process.")
    
    strategies = {
        "length": LengthStrategy(max_chars=800, overlap=200),
        "level": LevelStrategy(max_chars=800),
        "semantic": SemanticStrategy(max_chars=800, similarity_threshold=0.6)
    }
    
    results = []
    
    # Create report file immediately to verify write access
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    report_path = os.path.join(OUTPUT_ROOT, "evaluation_report.jsonl")
    
    for i, file_path in enumerate(files):
        print(f"Processing {i+1}/{len(files)}: {os.path.basename(file_path)}")
        doc_name = os.path.splitext(os.path.basename(file_path))[0].replace("_content_list", "")
        
        try:
            nodes = load_document_nodes(file_path)
            if not nodes:
                continue
                
            doc_result = {"doc_name": doc_name, "strategies": {}}
            
            for name, strategy in strategies.items():
                start_time = time.time()
                chunks = strategy.chunk(nodes)
                elapsed = time.time() - start_time
                
                # Save chunks
                save_chunks(chunks, name, doc_name)
                
                # Evaluate
                score = evaluate_chunks(chunks)
                
                doc_result["strategies"][name] = {
                    "chunk_count": len(chunks),
                    "avg_chunk_len": np.mean([len(c) for c in chunks]) if chunks else 0,
                    "score": score, # Lower is better (distance)
                    "time_sec": elapsed
                }
            
            # Append to results and write to file incrementally
            results.append(doc_result)
            with open(report_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(doc_result, ensure_ascii=False) + "\n")
                
        except Exception as e:
            print(f"Error processing {doc_name}: {e}")
            import traceback
            traceback.print_exc()

    # Aggregate results using the external module
    try:
        from calculate_aggregate import calculate_weighted_aggregate
        calculate_weighted_aggregate(report_path)
    except ImportError:
        # Fallback if running from different dir
        try:
            from .calculate_aggregate import calculate_weighted_aggregate
            calculate_weighted_aggregate(report_path)
        except ImportError:
            print("Could not import calculate_weighted_aggregate for final summary.")

if __name__ == "__main__":
    main()

