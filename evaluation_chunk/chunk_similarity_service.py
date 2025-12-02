import os
import numpy as np
from typing import List
from openai import OpenAI

# Configuration
# Try to get from env, otherwise use defaults (from user snippet)
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "sk-6a44d15e56dd4007945ccc41b97b499c")
QWEN_API_BASE = os.environ.get("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
# Using text-embedding-v3 as v4 is not standard yet or might be specific, ensuring fallback
QWEN_EMBED_MODEL = os.environ.get("QWEN_EMBED_MODEL", "text-embedding-v3") 

def _embed_texts_remote(texts: List[str]) -> np.ndarray:
    """
    Batch embedding using OpenAI compatible client (Qwen/Dashscope).
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
        
    client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_API_BASE)
    
    # Batch size limit for API
    max_batch = 10 # Safe batch size for some APIs
    all_embs: List[np.ndarray] = []
    
    for i in range(0, len(texts), max_batch):
        batch = texts[i : i + max_batch]
        try:
            # Ensure all texts are strings and not empty
            batch = [str(t) if t else " " for t in batch]
            resp = client.embeddings.create(
                model=QWEN_EMBED_MODEL, 
                input=batch, 
                encoding_format="float"
            )
            # Sort by index to ensure order (OpenAI API usually preserves order but good to be safe if they return index)
            data = sorted(resp.data, key=lambda x: x.index)
            all_embs.append(
                np.array([d.embedding for d in data], dtype=np.float32)
            )
        except Exception as e:
            print(f"Embedding failed for batch {i}: {e}")
            # Return zero vectors or retry? For now, raise to notice
            raise e

    if all_embs:
        return np.vstack(all_embs)
    return np.empty((0, 0), dtype=np.float32)

def _no_semantic_similarity_avg(segments: List[str]) -> float:
    """
    Calculate average (1 - cosine_similarity) between adjacent segments.
    Lower score is better (means high similarity / smooth transition), 
    but the function name suggests we measure "no similarity" (distance).
    """
    if len(segments) < 2:
        return 0.0
        
    # Filter out empty segments to avoid noise
    valid_segments = [s for s in segments if s.strip()]
    if len(valid_segments) < 2:
        return 0.0
        
    emb = _embed_texts_remote(valid_segments)
    if emb.shape[0] < 2:
        return 0.0
        
    # Normalize embeddings for cosine similarity
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    # Avoid division by zero
    norm[norm == 0] = 1e-10
    emb_norm = emb / norm
    
    q = emb_norm[1:]
    p = emb_norm[:-1]
    
    # Dot product of normalized vectors = cosine similarity
    sims = np.sum(q * p, axis=1) 
    
    # Distance = 1 - Similarity
    dists = 1.0 - sims
    return float(np.mean(dists))

def evaluate_single_file(segments: List[str]) -> float:
    """
    Evaluate a list of chunks.
    Returns average semantic distance (lower is better consistency).
    """
    return _no_semantic_similarity_avg(segments)
