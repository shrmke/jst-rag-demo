import numpy as np
from typing import List, Dict, Any, Generator
from utils import extract_text_from_node
try:
    from chunk_similarity_service import _embed_texts_remote
except ImportError:
    # This might happen if running from root without -m
    try:
        from evaluation_chunk.chunk_similarity_service import _embed_texts_remote
    except ImportError:
        raise


class BaseStrategy:
    def __init__(self, max_chars: int = 1000):
        self.max_chars = max_chars

    def _split_node_text(self, text: str) -> List[str]:
        """Hard split for very large nodes."""
        res = []
        for i in range(0, len(text), self.max_chars):
            res.append(text[i : i + self.max_chars])
        return res

    def chunk(self, nodes: List[Dict[str, Any]]) -> List[str]:
        raise NotImplementedError

class LengthStrategy(BaseStrategy):
    def __init__(self, max_chars: int = 800, overlap: int = 100):
        super().__init__(max_chars)
        self.overlap = overlap

    def chunk(self, nodes: List[Dict[str, Any]]) -> List[str]:
        chunks = []
        current_nodes = []
        current_len = 0
        
        for node in nodes:
            text = node.get("_content_text", "")
            if not text:
                continue
            
            # Handle oversized node
            if len(text) > self.max_chars:
                # Flush current
                if current_nodes:
                    chunks.append("\n".join([n.get("_content_text", "") for n in current_nodes]))
                    current_nodes = []
                    current_len = 0
                
                # Split large node
                parts = self._split_node_text(text)
                chunks.extend(parts)
                
                # Handle overlap for the next chunk from the last part of this large node?
                # For simplicity, we start fresh or take the last part as context if needed.
                # Strategy: just continue.
                continue

            if current_len + len(text) > self.max_chars:
                # Flush
                chunk_text = "\n".join([n.get("_content_text", "") for n in current_nodes])
                chunks.append(chunk_text)
                
                # Prepare next chunk with overlap
                # Backtrack to find nodes to keep
                overlap_nodes = []
                overlap_len = 0
                for prev in reversed(current_nodes):
                    overlap_nodes.insert(0, prev)
                    overlap_len += len(prev.get("_content_text", ""))
                    if overlap_len >= self.overlap:
                        break
                
                current_nodes = overlap_nodes + [node]
                current_len = overlap_len + len(text)
            else:
                current_nodes.append(node)
                current_len += len(text)
        
        if current_nodes:
            chunks.append("\n".join([n.get("_content_text", "") for n in current_nodes]))
            
        return chunks

class LevelStrategy(BaseStrategy):
    def __init__(self, max_chars: int = 800):
        super().__init__(max_chars)

    def chunk(self, nodes: List[Dict[str, Any]]) -> List[str]:
        chunks = []
        current_nodes = []
        current_len = 0
        
        for node in nodes:
            text = node.get("_content_text", "")
            if not text:
                continue
            
            is_level_break = node.get("text_level") is not None
            
            # Logic: If level break OR max len reached -> Split
            # Note: If level break, we typically split BEFORE this node.
            
            if len(text) > self.max_chars:
                if current_nodes:
                    chunks.append("\n".join([n.get("_content_text", "") for n in current_nodes]))
                    current_nodes = []
                    current_len = 0
                parts = self._split_node_text(text)
                chunks.extend(parts)
                continue

            force_split = is_level_break and current_nodes # Only split if we have content
            len_split = (current_len + len(text) > self.max_chars)
            
            if force_split or len_split:
                # Flush
                chunk_text = "\n".join([n.get("_content_text", "") for n in current_nodes])
                chunks.append(chunk_text)
                current_nodes = [node]
                current_len = len(text)
            else:
                current_nodes.append(node)
                current_len += len(text)

        if current_nodes:
            chunks.append("\n".join([n.get("_content_text", "") for n in current_nodes]))
            
        return chunks

class SemanticStrategy(BaseStrategy):
    def __init__(self, max_chars: int = 800, similarity_threshold: float = 0.6):
        super().__init__(max_chars)
        self.threshold = similarity_threshold

    def chunk(self, nodes: List[Dict[str, Any]]) -> List[str]:
        # 1. Pre-embed all nodes
        texts = [n.get("_content_text", "") for n in nodes]
        valid_indices = [i for i, t in enumerate(texts) if t.strip()]
        valid_texts = [texts[i] for i in valid_indices]
        
        if not valid_texts:
            return []

        # Batch embedding
        try:
            embeddings = _embed_texts_remote(valid_texts)
        except Exception as e:
            print(f"Embedding failed: {e}, falling back to LengthStrategy")
            return LengthStrategy(self.max_chars).chunk(nodes)

        # Map back to original node index
        node_embeddings = {} # index -> embedding
        for i, original_idx in enumerate(valid_indices):
            node_embeddings[original_idx] = embeddings[i]

        chunks = []
        current_nodes = []
        current_len = 0
        
        for i, node in enumerate(nodes):
            text = node.get("_content_text", "")
            if not text.strip():
                continue

            if len(text) > self.max_chars:
                if current_nodes:
                    chunks.append("\n".join([n.get("_content_text", "") for n in current_nodes]))
                    current_nodes = []
                    current_len = 0
                parts = self._split_node_text(text)
                chunks.extend(parts)
                continue

            # Decide whether to split
            split_needed = False
            
            # Length check
            if current_len + len(text) > self.max_chars:
                split_needed = True
            
            # Semantic check (if not already splitting by length and we have a previous node)
            elif current_nodes:
                prev_node = current_nodes[-1]
                # Find embedding for current and prev
                # Note: We need original indices. 
                # Since we iterate `nodes` in order, `i` is the index.
                # `prev_node` index? We didn't track it. 
                # Hack: let's assume the loop index `i` corresponds to `node`. 
                # We need index of `prev_node`.
                # Better: Store index in `current_nodes` tuples or look up by `i-1` if contiguous?
                # `current_nodes` might not be contiguous in `nodes` if we skipped empty ones.
                # Let's just rely on `i` and keep track of `last_valid_idx` in `current_nodes`.
                
                curr_emb = node_embeddings.get(i)
                
                # Find the index of the last node in current_nodes
                # This is tricky unless we stored it. 
                # Let's store (node, index) in current_nodes
                pass

            # Refined loop for Semantic:
            # We need access to "last added node embedding"
            pass

        # Re-implementing loop to handle indices correctly
        current_nodes_with_idx = [] # List of (node, embedding)
        
        for i, node in enumerate(nodes):
            text = node.get("_content_text", "")
            if not text.strip():
                continue
            
            emb = node_embeddings.get(i) # May be None if something failed, unlikely for valid text
            
            if len(text) > self.max_chars:
                if current_nodes_with_idx:
                    chunks.append("\n".join([n[0].get("_content_text", "") for n in current_nodes_with_idx]))
                    current_nodes_with_idx = []
                    current_len = 0
                parts = self._split_node_text(text)
                chunks.extend(parts)
                continue

            split_semantic = False
            if current_nodes_with_idx and emb is not None:
                last_node, last_emb = current_nodes_with_idx[-1]
                if last_emb is not None:
                    # Cosine Similarity
                    # norm already done? _embed_texts_remote returns raw embeddings usually.
                    # Helper function in service did normalization. We should do it here.
                    
                    v1 = last_emb
                    v2 = emb
                    
                    norm1 = np.linalg.norm(v1)
                    norm2 = np.linalg.norm(v2)
                    
                    if norm1 > 0 and norm2 > 0:
                        sim = np.dot(v1, v2) / (norm1 * norm2)
                        if sim < self.threshold:
                            split_semantic = True

            if (current_len + len(text) > self.max_chars) or split_semantic:
                if current_nodes_with_idx:
                    chunks.append("\n".join([n[0].get("_content_text", "") for n in current_nodes_with_idx]))
                current_nodes_with_idx = [(node, emb)]
                current_len = len(text)
            else:
                current_nodes_with_idx.append((node, emb))
                current_len += len(text)

        if current_nodes_with_idx:
             chunks.append("\n".join([n[0].get("_content_text", "") for n in current_nodes_with_idx]))
             
        return chunks
