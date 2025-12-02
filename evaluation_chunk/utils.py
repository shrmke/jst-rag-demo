import json
import os
from typing import List, Dict, Any, Generator

ACCEPT_TYPES = {"text", "table", "list", "equation"}

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_space(text: str) -> str:
    import re
    return re.sub(r"\s+", " ", (text or "").strip())

def extract_text_from_node(node: Dict[str, Any]) -> str:
    """Extract text content from a node."""
    typ = node.get("type")
    if typ == "list":
        items = node.get("items", [])
        parts = []
        for it in items:
            if isinstance(it, str):
                parts.append(normalize_space(it))
            elif isinstance(it, dict):
                content = it.get("text") or it.get("content") or ""
                if content:
                    parts.append(normalize_space(content))
        return "；".join(parts)
    
    # fallback for text/table/equation
    # Note: For tables, we might want table_text or table_body. 
    # For simplicity in this text-based evaluation, we take table_text or flattened content.
    content = node.get("text") or node.get("table_text") or node.get("content") or ""
    return normalize_space(content)

def flatten_nodes(data: Any) -> Generator[Dict[str, Any], None, None]:
    """Recursively yield content nodes."""
    if isinstance(data, dict):
        if data.get("type") in ACCEPT_TYPES:
            # Attach extracted text for easier processing
            data["_content_text"] = extract_text_from_node(data)
            yield data
        
        # Recurse into values just in case (though Mineru content_list is usually flat-ish)
        # Actually Mineru structure in content_list is typically a flat list.
        # But sometimes recursive if nested.
        # We don't recurse inside a valid Node (like inside a table)
        pass 
    elif isinstance(data, list):
        for item in data:
            yield from flatten_nodes(item)

def load_document_nodes(file_path: str) -> List[Dict[str, Any]]:
    """Load and flatten valid nodes from a content_list.json file."""
    raw_data = read_json(file_path)
    nodes = list(flatten_nodes(raw_data))
    # Filter out empty nodes
    return [n for n in nodes if n.get("_content_text")]
