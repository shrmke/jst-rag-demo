#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Search over FAISS and BM25 indexes built by build_indexes.py.

Features:
  - Variant: exp | ctrl (choose corresponding index dirs)
  - Modes: faiss-only, bm25-only, or hybrid (linear blend with alpha)
  - Filters: by doc_id/type/page range (optional)
  - For ctrl, returns raw HTML content if present; otherwise returns embed_text/content

Examples:
  python3 search.py --variant exp \
    --faiss-dir /Users/wangyaqi/Documents/cursor_project/jst-rag-demo/jst-rag-demo/indexes/faiss_exp \
    --bm25-dir  /Users/wangyaqi/Documents/cursor_project/jst-rag-demo/jst-rag-demo/indexes/bm25_exp \
    --query "募投项目 调整后金额" --topk 10

  python3 search.py --variant ctrl \
    --faiss-dir /Users/wangyaqi/Documents/cursor_project/jst-rag-demo/jst-rag-demo/indexes/faiss_ctrl \
    --bm25-dir  /Users/wangyaqi/Documents/cursor_project/jst-rag-demo/jst-rag-demo/indexes/bm25_ctrl \
    --query "可转债 转股价格" --topk 10 --alpha 0.3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np  # type: ignore
import urllib.request as _ureq
import urllib.error as _uerr
import http.client as _http
import socket as _sock
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:
    BM25Okapi = None

import re

from intent.router import analyze_intent, load_index_registry

QWEN_API_KEY = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY") or "sk-6a44d15e56dd4007945ccc41b97b499c"
QWEN_API_BASE = (
    os.environ.get("DASHSCOPE_BASE_URL")
    or os.environ.get("QWEN_API_BASE")
    or "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
QWEN_EMBED_MODEL = os.environ.get("QWEN_EMBED_MODEL", "text-embedding-v4")
QWEN_EMBED_DIM = int(os.environ.get("QWEN_EMBED_DIM", "1024"))
QWEN_RERANK_ENDPOINT = os.environ.get(
    "QWEN_RERANK_ENDPOINT",
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
)
QWEN_RERANK_MODEL = os.environ.get("QWEN_RERANK_MODEL", "qwen3-rerank")
QWEN_CHAT_MODEL = os.environ.get("QWEN_CHAT_MODEL", "qwen-plus")

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+", re.UNICODE)


def bm25_tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for m in _TOKEN_RE.finditer(text):
        s = m.group(0)
        if re.match(r"[\u4e00-\u9fff]", s):
            tokens.extend(list(s))
        else:
            tokens.append(s.lower())
    return tokens


def _embedding_batch(texts: List[str]) -> List[List[float]]:
    if not QWEN_API_KEY:
        raise RuntimeError("DASHSCOPE_API_KEY/QWEN_API_KEY 未设置，请导出有效的 API Key")
    url = f"{QWEN_API_BASE}/embeddings"
    payload: Dict[str, Any] = {
        "model": QWEN_EMBED_MODEL,
        "input": texts,
        "encoding_format": "float",
    }
    if QWEN_EMBED_DIM:
        payload["dimensions"] = QWEN_EMBED_DIM
    data_bytes = None
    retries = 3
    backoff = 1.0
    last_err: Exception | None = None
    for attempt in range(retries):
        req = _ureq.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {QWEN_API_KEY}")
        try:
            with _ureq.urlopen(req, timeout=120) as resp:
                data_bytes = resp.read()
            break
        except _uerr.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                err_body = ""
            msg = f"[EMBED][HTTP {e.code}] {e.reason} url={url}\n{err_body}"
            raise RuntimeError(msg) from e
        except (_http.IncompleteRead, _uerr.URLError, _sock.timeout) as e:
            last_err = e
            if attempt == retries - 1:
                if isinstance(e, _http.IncompleteRead) and getattr(e, "partial", None):
                    data_bytes = e.partial
                    break
                raise RuntimeError(f"[EMBED][RETRY_EXHAUSTED] {type(e).__name__}: {e}") from e
            import time as _time
            _time.sleep(backoff)
            backoff *= 2
    body = (data_bytes or b"").decode("utf-8", errors="ignore")
    try:
        obj = json.loads(body)
    except Exception as e:
        snippet = body[:1000]
        raise RuntimeError(f"[EMBED][INVALID_JSON] {type(last_err).__name__ if last_err else ''}: {e}\n<<body_snippet>>\n{snippet}\n<<end>>") from e
    data = obj.get("data") or []
    vectors: List[List[float]] = []
    for item in data:
        vec = item.get("embedding")
        if isinstance(vec, list):
            vectors.append(vec)
    return vectors


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms


def load_meta(meta_path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def min_max_norm(scores: List[float]) -> List[float]:
    if not scores:
        return scores
    mi, ma = min(scores), max(scores)
    if ma - mi < 1e-12:
        return [0.0 for _ in scores]
    return [(s - mi) / (ma - mi) for s in scores]


def list_index_dirs(root_dir: str, kind: str) -> List[str]:
    """Return list of child dirs that contain faiss.index or bm25.pkl depending on kind."""
    out: List[str] = []
    if not os.path.isdir(root_dir):
        return out
    for name in sorted(os.listdir(root_dir)):
        d = os.path.join(root_dir, name)
        if not os.path.isdir(d):
            continue
        if kind == "faiss" and os.path.exists(os.path.join(d, "faiss.index")):
            out.append(d)
        if kind == "bm25" and os.path.exists(os.path.join(d, "bm25.pkl")):
            out.append(d)
    return out


def search_faiss_multi(query_vec: np.ndarray, faiss_dirs: List[str], per_index_topk: int) -> List[Tuple[float, int, Dict[str, Any], str]]:
    """Return list of (score, idx, meta, dir) across multiple faiss sub-indexes."""
    q = query_vec.astype("float32")[None, :]
    results: List[Tuple[float, int, Dict[str, Any], str]] = []
    for d in faiss_dirs:
        try:
            index = faiss.read_index(os.path.join(d, "faiss.index"))
        except Exception:
            continue
        metas = load_meta(os.path.join(d, "meta.jsonl"))
        k = min(per_index_topk, len(metas)) if metas else per_index_topk
        if k <= 0:
            continue
        D, I = index.search(q, k)
        if len(D) == 0:
            continue
        for score, rid in zip(D[0], I[0]):
            if 0 <= rid < len(metas):
                results.append((float(score), int(rid), metas[rid], d))
    return results


def search_bm25_multi(query: str, bm25_dirs: List[str], per_index_topk: int) -> List[Tuple[float, int, Dict[str, Any], str]]:
    """Return list of (score, idx, meta, dir) across multiple bm25 sub-indexes."""
    results: List[Tuple[float, int, Dict[str, Any], str]] = []
    tokens = bm25_tokenize(query)
    import pickle
    for d in bm25_dirs:
        pkl = os.path.join(d, "bm25.pkl")
        meta_path = os.path.join(d, "meta.jsonl")
        if not (os.path.exists(pkl) and os.path.exists(meta_path)):
            continue
        try:
            with open(pkl, "rb") as f:
                bm25 = pickle.load(f)
        except Exception:
            continue
        metas = load_meta(meta_path)
        scores = list(bm25.get_scores(tokens))
        if not scores:
            continue
        # topk indices
        k = min(per_index_topk, len(scores))
        if k <= 0:
            continue
        idxs = np.argpartition(np.array(scores), -k)[-k:]
        for rid in idxs:
            i = int(rid)
            if 0 <= i < len(metas):
                results.append((float(scores[i]), i, metas[i], d))
    return results


def call_qwen_rerank(query: str, docs: List[str], top_n: int, instruct: str) -> List[Tuple[int, float]]:
    """Return list of (doc_index, score) sorted by score desc."""
    if not QWEN_API_KEY:
        raise RuntimeError("DASHSCOPE_API_KEY/QWEN_API_KEY 未设置")
    payload = {
        "model": QWEN_RERANK_MODEL,
        "input": {"query": query, "documents": docs},
        "parameters": {
            "return_documents": True,
            "top_n": top_n,
            "instruct": instruct,
        },
    }
    req = _ureq.Request(
        QWEN_RERANK_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {QWEN_API_KEY}")
    try:
        with _ureq.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
    except _uerr.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            err_body = ""
        raise RuntimeError(f"[RERANK][HTTP {e.code}] {e.reason}\n{err_body}") from e
    obj = json.loads(body)
    # API 返回格式可能变化，尝试多种路径
    out: List[Tuple[int, float]] = []
    # 常见：obj["output"]["results"] 每项有 {index, score}
    results = None
    if isinstance(obj.get("output"), dict):
        results = obj["output"].get("results") or obj["output"].get("documents")
    if results is None and isinstance(obj.get("data"), dict):
        results = obj["data"].get("results") or obj["data"].get("documents")
    if isinstance(results, list):
        for r in results:
            idx = r.get("index")
            score = r.get("score")
            if isinstance(idx, int) and isinstance(score, (int, float)):
                out.append((idx, float(score)))
    # 兜底：如果上述路径都没有 index，则按顺序截断 top_n
    if not out:
        for i in range(min(top_n, len(docs))):
            out.append((i, float(len(docs) - i)))
    # 按分数降序
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:top_n]


class ContextRestorer:
    """
    Handles context expansion and table restoration for retrieval results.
    Refactored to optimize redundancy and performance.
    """
    def __init__(self, neighbor_radius: int = 1, enable_table_restoration: bool = True):
        self.neighbor_radius = neighbor_radius
        self.enable_table_restoration = enable_table_restoration
        # index_dir -> { (doc_id, order) -> item }
        self.index_map: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]] = {}
        # index_dir -> { table_id -> [item, ...] }
        self.table_map: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self.loaded_dirs: set[str] = set()

    def _ensure_dir(self, index_dir: str):
        if index_dir in self.loaded_dirs:
            return
        if not os.path.isdir(index_dir):
            return
        meta_path = os.path.join(index_dir, "meta.jsonl")
        if not os.path.exists(meta_path):
            return
            
        metas = load_meta(meta_path)
        idx_map = {}
        tbl_map = {}
        for m in metas:
            did = str(m.get("doc_id", ""))
            odr = m.get("order")
            if isinstance(odr, int):
                idx_map[(did, odr)] = m
            
            # Table indexing
            tbl = m.get("table") or {}
            if isinstance(tbl, dict):
                tid = str(tbl.get("id", ""))
                if tid:
                    if tid not in tbl_map:
                        tbl_map[tid] = []
                    tbl_map[tid].append(m)
        
        # Sort table rows once
        for tid in tbl_map:
            # Sort by row_index then order
            tbl_map[tid].sort(key=lambda x: (
                (x.get("table", {}).get("row_index") if isinstance(x.get("table", {}).get("row_index"), int) else 999999),
                (x.get("order") if isinstance(x.get("order"), int) else 999999)
            ))

        self.index_map[index_dir] = idx_map
        self.table_map[index_dir] = tbl_map
        self.loaded_dirs.add(index_dir)

    def restore(self, item: Dict[str, Any]) -> str:
        index_dir = item.get("index_dir")
        if not index_dir:
            return str(item.get("content") or "")
        
        self._ensure_dir(index_dir)
        
        doc_id = str(item.get("doc_id", ""))
        order = item.get("order")
        if not isinstance(order, int):
             return str(item.get("content") or "")

        # Fix: Try to get table info from item root first, then meta
        table_info = item.get("table")
        if not table_info and isinstance(item.get("meta"), dict):
            table_info = item.get("meta", {}).get("table")
        
        is_table = isinstance(table_info, dict) and bool(table_info.get("id"))
        
        # 1. Base Content Restoration
        base_content = str(item.get("content") or "")
        start_order = order
        end_order = order
        
        if self.enable_table_restoration and is_table:
            tid = str(table_info.get("id"))
            # Check if table map exists for this dir
            if index_dir in self.table_map:
                rows = self.table_map[index_dir].get(tid, [])
                if rows:
                    # Reconstruct table
                    parts = [str(r.get("raw_content") or r.get("embed_text") or r.get("content") or "") for r in rows]
                    base_content = "\n".join([p for p in parts if p])
                    # Determine boundaries
                    valid_orders = [r.get("order") for r in rows if isinstance(r.get("order"), int)]
                    if valid_orders:
                        start_order = min(valid_orders)
                        end_order = max(valid_orders)
        
        if self.neighbor_radius <= 0:
            return base_content

        # 2. Window Expansion
        contents = []
        
        # Previous chunks
        for i in range(self.neighbor_radius, 0, -1):
            tgt = start_order - i
            if index_dir in self.index_map:
                neighbor = self.index_map[index_dir].get((doc_id, tgt))
                if neighbor:
                    c = str(neighbor.get("raw_content") or neighbor.get("embed_text") or neighbor.get("content") or "")
                    contents.append(c)
        
        contents.append(base_content)
        
        # Next chunks
        for i in range(1, self.neighbor_radius + 1):
            tgt = end_order + i
            if index_dir in self.index_map:
                neighbor = self.index_map[index_dir].get((doc_id, tgt))
                if neighbor:
                    c = str(neighbor.get("raw_content") or neighbor.get("embed_text") or neighbor.get("content") or "")
                    contents.append(c)

        return "\n\n".join(contents)


def search(
    query: str,
    faiss_dir: str,
    bm25_dir: str,
    # topk removed or unused
    alpha: float,
    mode: str,
    filter_doc: str,
    filter_type: str,
    page_from: int,
    page_to: int,
    pre_topk: int,
    faiss_per_index: int,
    bm25_per_index: int,
    rerank_topk: int,
    rerank_instruct: str,
    neighbor_radius: int,
    # return_table_full removed (always handled in logic)
) -> List[Dict[str, Any]]:
    # Detect whether provided dirs are single-index or roots containing sub-indexes
    faiss_dirs: List[str] = []
    bm25_dirs: List[str] = []
    if faiss and os.path.exists(os.path.join(faiss_dir, "faiss.index")):
        faiss_dirs = [faiss_dir]
    else:
        faiss_dirs = list_index_dirs(faiss_dir, "faiss")
    if BM25Okapi and os.path.exists(os.path.join(bm25_dir, "bm25.pkl")):
        bm25_dirs = [bm25_dir]
    else:
        bm25_dirs = list_index_dirs(bm25_dir, "bm25")

    # Embedding query vector
    emb_list = _embedding_batch([query])
    if not emb_list or not isinstance(emb_list[0], list) or len(emb_list[0]) == 0:
        return []
    _vec = np.array(emb_list[0], dtype="float32")
    _norm = float(np.linalg.norm(_vec) + 1e-12)
    qv = _vec / _norm

    # Per-index retrieval
    per_index_faiss_topk = max(1, int(faiss_per_index))
    per_index_bm25_topk = max(1, int(bm25_per_index))
    faiss_hits = search_faiss_multi(qv, faiss_dirs, per_index_faiss_topk) if faiss_dirs else []
    bm25_hits = search_bm25_multi(query, bm25_dirs, per_index_bm25_topk) if bm25_dirs else []

    # Apply filters and build candidate pool
    def pass_filter(m: Dict[str, Any]) -> bool:
        if filter_doc and str(m.get("doc_id", "")) != filter_doc:
            return False
        if filter_type and str(m.get("type", "")) != filter_type:
            return False
        p = m.get("page")
        if isinstance(p, int):
            if page_from is not None and p < page_from:
                return False
            if page_to is not None and p > page_to:
                return False
        return True

    candidates: List[Dict[str, Any]] = []
    # normalize scores individually then blend for pruning
    if faiss_hits:
        f_scores = [h[0] for h in faiss_hits]
        f_norm = min_max_norm(f_scores)
        for i, (score, rid, meta, d) in enumerate(faiss_hits):
            if not pass_filter(meta):
                continue
            candidates.append({
                "modal": "faiss",
                "score_raw": float(score),
                "score_norm": float(f_norm[i]),
                "rid": int(rid),
                "meta": meta,
                "dir": d,
            })
    if bm25_hits:
        b_scores = [h[0] for h in bm25_hits]
        b_norm = min_max_norm(b_scores)
        for i, (score, rid, meta, d) in enumerate(bm25_hits):
            if not pass_filter(meta):
                continue
            candidates.append({
                "modal": "bm25",
                "score_raw": float(score),
                "score_norm": float(b_norm[i]),
                "rid": int(rid),
                "meta": meta,
                "dir": d,
            })
    # blend scores for pre-topk pruning
    for c in candidates:
        c["score_blend"] = float((1 - alpha) * (c["score_norm"] if c["modal"] == "faiss" else 0.0) + alpha * (c["score_norm"] if c["modal"] == "bm25" else 0.0))
    candidates.sort(key=lambda x: x["score_blend"], reverse=True)

    # 4. Table Dedup (Logic: Keep first occurrence of any table_id)
    deduped_candidates: List[Dict[str, Any]] = []
    seen_table_ids = set()
    for c in candidates:
        m = c["meta"]
        t = m.get("table") or {}
        if isinstance(t, dict) and t.get("id"):
            tid = str(t.get("id"))
            if tid in seen_table_ids:
                continue
            seen_table_ids.add(tid)
        deduped_candidates.append(c)
    
    pre = deduped_candidates[: pre_topk]

    # Dedup by id
    seen_ids = set()
    pre_unique: List[Dict[str, Any]] = []
    for c in pre:
        mid = str(c["meta"].get("id"))
        if mid in seen_ids:
            continue
        seen_ids.add(mid)
        pre_unique.append(c)

    # Prepare docs for rerank
    docs = []
    for c in pre_unique:
        m = c["meta"]
        docs.append(m.get("raw_content") or m.get("embed_text") or "")
    # Call reranker
    rerank_pairs = call_qwen_rerank(query, docs, rerank_topk, rerank_instruct)
    final: List[Dict[str, Any]] = []
    for doc_idx, rscore in rerank_pairs:
        if not (0 <= doc_idx < len(pre_unique)):
            continue
        c = pre_unique[doc_idx]
        m = c["meta"]
        out_item = {
            "rerank_score": float(rscore),
            "modal": c["modal"],
            "id": m.get("id"),
            "doc_id": m.get("doc_id"),
            "page": m.get("page"),
            "type": m.get("type"),
            "order": m.get("order"),
            "content": m.get("raw_content") or m.get("embed_text"),
            "table": m.get("table"),
            "meta": m.get("meta"),
            "source_file": m.get("source_file"),
            "index_dir": c["dir"],
        }
        final.append(out_item)

    # 3. Context Restoration & Expansion
    # Instantiated with enable_table_restoration=True by default logic
    restorer = ContextRestorer(neighbor_radius=neighbor_radius, enable_table_restoration=True)
    for item in final:
        try:
            expanded_content = restorer.restore(item)
            item["content"] = expanded_content
            item["is_extended"] = True
        except Exception:
            pass

    return final


def _bm25_dirs_for(doc_type: str, years: List[int]) -> List[str]:
    reg = load_index_registry() or {}
    out: List[str] = []
    by_ty = (reg.get("bm25") or {}).get("by_type_year") or {}
    if not doc_type or doc_type not in by_ty:
        return out
    for y in years or []:
        ykey = str(y)
        for d in by_ty[doc_type].get(ykey, []) or []:
            if d not in out:
                out.append(d)
    if not out and "unknown" in by_ty.get(doc_type, {}):
        for d in by_ty[doc_type]["unknown"]:
            if d not in out:
                out.append(d)
    return out


def route_and_search(
    query: str,
    # topk removed/unused
    alpha: float = 0.5,
    pre_topk: int = 30,
    faiss_per_index: int = 10,
    bm25_per_index: int = 50,
    rerank_topk: int = 10,
    rerank_instruct: str = "Given a web search query, retrieve relevant passages that answer the query.",
    neighbor_radius: int = 1,
    # return_table_full removed
    prefer_year: bool = True,
    doc_type_override: Optional[str] = None,  # "report" | "notice" | "both" | None
    run_faiss: bool = True,
    run_bm25: bool = True,
) -> Dict[str, Any]:
    """
    E2E: analyze intent -> select subset of indexes -> retrieve -> rerank -> return results with intent and trace.
    """
    intent, trace = analyze_intent(query, use_llm=True)
    
    # routing
    from intent.router import _resolve_indices_by_years  # reuse resolver
    years = intent.years
    dtype = intent.doc_type
    # 覆盖文档类型：report/notice/both
    if isinstance(doc_type_override, str):
        if doc_type_override in ("report", "notice"):
            dtype = doc_type_override
        elif doc_type_override == "both":
            dtype = None  # downstream: None 表示两者都包含
    faiss_dirs: List[str] = _resolve_indices_by_years(dtype, years) if (prefer_year and years) else []
    # compute bm25 dirs for same buckets
    bm25_dirs: List[str] = []
    if dtype:
        bm25_dirs = _bm25_dirs_for(dtype, years if years else [])
    elif prefer_year and years:
        # 新增逻辑：如果类型未知但有年份，合并所有类型的 BM25 索引（对应 FAISS 的 fallback 逻辑）
        for ty in ("report", "notice"):
            for d in _bm25_dirs_for(ty, years):
                if d not in bm25_dirs:
                    bm25_dirs.append(d)
    elif doc_type_override == "both":
        # 合并两类
        # FAISS：按年份合并 report/notice 两类子库
        if prefer_year and years:
            tmp_faiss: List[str] = []
            for ty in ("report", "notice"):
                for d in _resolve_indices_by_years(ty, years):
                    if d not in tmp_faiss:
                        tmp_faiss.append(d)
            faiss_dirs = tmp_faiss
        for ty in ("report", "notice"):
            for d in _bm25_dirs_for(ty, years if years else []):
                if d not in bm25_dirs:
                    bm25_dirs.append(d)
    # If nothing resolved, fallback to global roots from registry
    if not faiss_dirs and not bm25_dirs:
        # Fallback: try global roots from registry to avoid empty retrieval
        reg = load_index_registry() or {}
        faiss_roots = (reg.get("faiss") or {}).get("global_roots") or []
        bm25_roots = (reg.get("bm25") or {}).get("global_roots") or []
        # expand roots to actual sub-index directories
        tmp_faiss: List[str] = []
        for root in faiss_roots:
            if not root:
                continue
            # if root itself is a single index dir, keep it; otherwise expand
            if os.path.exists(os.path.join(root, "faiss.index")):
                tmp_faiss.append(root)
            else:
                for d in list_index_dirs(root, "faiss"):
                    if d not in tmp_faiss:
                        tmp_faiss.append(d)
        tmp_bm25: List[str] = []
        for root in bm25_roots:
            if not root:
                continue
            if os.path.exists(os.path.join(root, "bm25.pkl")):
                tmp_bm25.append(root)
            else:
                for d in list_index_dirs(root, "bm25"):
                    if d not in tmp_bm25:
                        tmp_bm25.append(d)
        faiss_dirs = tmp_faiss
        bm25_dirs = tmp_bm25
    else:
        # If some dirs provided but they are roots, expand them to sub-index dirs
        if faiss_dirs:
            expanded: List[str] = []
            for d in faiss_dirs:
                if os.path.exists(os.path.join(d, "faiss.index")):
                    expanded.append(d)
                else:
                    for sd in list_index_dirs(d, "faiss"):
                        if sd not in expanded:
                            expanded.append(sd)
            faiss_dirs = expanded
        if bm25_dirs:
            expanded_b: List[str] = []
            for d in bm25_dirs:
                if os.path.exists(os.path.join(d, "bm25.pkl")):
                    expanded_b.append(d)
                else:
                    for sd in list_index_dirs(d, "bm25"):
                        if sd not in expanded_b:
                            expanded_b.append(sd)
            bm25_dirs = expanded_b
    # Embed query vector
    emb_list = _embedding_batch([query])
    if not emb_list or not isinstance(emb_list[0], list) or len(emb_list[0]) == 0:
        return {"intent": intent.to_dict(), "results": [], "trace": trace.to_dict()}
    import numpy as _np
    _vec = _np.array(emb_list[0], dtype="float32")
    _norm = float(_np.linalg.norm(_vec) + 1e-12)
    qv = _vec / _norm
    # Retrieve over chosen dirs
    faiss_hits = search_faiss_multi(qv, faiss_dirs, max(1, int(faiss_per_index))) if (faiss_dirs and run_faiss) else []
    bm25_hits = search_bm25_multi(query, bm25_dirs, max(1, int(bm25_per_index))) if (bm25_dirs and run_bm25) else []
    # Blend and rerank, reuse logic from search()
    def pass_filter(_m: Dict[str, Any]) -> bool:
        return True
    candidates: List[Dict[str, Any]] = []
    if faiss_hits:
        f_scores = [h[0] for h in faiss_hits]
        f_min, f_max = min(f_scores), max(f_scores)
        f_norm = [(s - f_min) / (f_max - f_min + 1e-12) for s in f_scores]
        for i, (score, rid, meta, d) in enumerate(faiss_hits):
            if not pass_filter(meta):
                continue
            candidates.append({
                "modal": "faiss",
                "score_raw": float(score),
                "score_norm": float(f_norm[i]),
                "rid": int(rid),
                "meta": meta,
                "dir": d,
            })
    if bm25_hits:
        b_scores = [h[0] for h in bm25_hits]
        b_min, b_max = min(b_scores), max(b_scores)
        b_norm = [(s - b_min) / (b_max - b_min + 1e-12) for s in b_scores]
        for i, (score, rid, meta, d) in enumerate(bm25_hits):
            if not pass_filter(meta):
                continue
            candidates.append({
                "modal": "bm25",
                "score_raw": float(score),
                "score_norm": float(b_norm[i]),
                "rid": int(rid),
                "meta": meta,
                "dir": d,
            })

    # 创建日志文件路径
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f"search_candidates.log")
    
    # 辅助函数：同时写入 stderr 和日志文件
    def log_print(msg: str) -> None:
        # print(msg, file=sys.stderr, flush=True)
        try:
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass  # 忽略文件写入错误，不影响主流程
    log_print(f"-------- [Timestamp] -------- {timestamp}")
    log_print(f"-------- [Query] -------- {query}")
    log_print(f"-------- [Intent] -------- {intent.to_dict()}")
    log_print(f"-------- [Trace] -------- {trace.to_dict()}")
    log_print(f"-------- [Faiss Dirs] -------- {faiss_dirs}")
    log_print(f"-------- [Bm25 Dirs] -------- {bm25_dirs}")
    # log_print(f"-------- [Faiss Hits] -------- {faiss_hits}")
    # log_print(f"-------- [Bm25 Hits] -------- {bm25_hits}")
    # log_print("-------- [FAISS Candidates] --------")
    # faiss_candidates = [c for c in candidates if c["modal"] == "faiss"]
    # if faiss_candidates:
    #     faiss_candidates.sort(key=lambda x: x['score_norm'], reverse=True)
    #     for i, c in enumerate(faiss_candidates):
    #         log_print(f"  [{i+1}] rid={c['rid']}, raw={c['score_raw']:.6f}, norm={c['score_norm']:.6f}, dir={c['dir']},content={c['meta'].get('raw_content')}")
    # else:
    #     log_print("  (no faiss candidates)")
    
    # log_print("-------- [BM25 Candidates] --------")
    # bm25_candidates = [c for c in candidates if c["modal"] == "bm25"]
    # if bm25_candidates:
    #     bm25_candidates.sort(key=lambda x: x['score_norm'], reverse=True)
    #     for i, c in enumerate(bm25_candidates):
    #         log_print(f"  [{i+1}] rid={c['rid']}, raw={c['score_raw']:.6f}, norm={c['score_norm']:.6f}, dir={c['dir']},content={c['meta'].get('raw_content')}")
    # else:
    #     log_print("  (no bm25 candidates)")
    # log_print("------------------------------------")

    
    # for c in candidates:
    #     c["score_blend"] = float((1 - alpha) * (c["score_norm"] if c["modal"] == "faiss" else 0.0) + alpha * (c["score_norm"] if c["modal"] == "bm25" else 0.0))
    # candidates.sort(key=lambda x: x["score_blend"], reverse=True)
    
    
    # RRF (Reciprocal Rank Fusion) 融合排序
    # RRF 公式: RRF(d) = Σ 1/(k + rank_i(d))
    # k 是常数，通常取 60
    rrf_k = 60.0
    
    # 1. 分别对 faiss 和 bm25 候选按原始分数排序（降序）
    faiss_candidates_sorted = sorted(
        [c for c in candidates if c["modal"] == "faiss"],
        key=lambda x: x["score_raw"],
        reverse=True
    )
    bm25_candidates_sorted = sorted(
        [c for c in candidates if c["modal"] == "bm25"],
        key=lambda x: x["score_raw"],
        reverse=True
    )
    
    # 2. 为每个文档创建唯一标识并记录排名
    # 使用 (rid, dir) 作为唯一标识
    def get_doc_key(c: Dict[str, Any]) -> Tuple[int, str]:
        return (c["rid"], c["dir"])
    
    # 记录每个文档在 faiss 和 bm25 中的排名（从 1 开始）
    faiss_ranks: Dict[Tuple[int, str], int] = {}
    for rank, c in enumerate(faiss_candidates_sorted, start=1):
        doc_key = get_doc_key(c)
        faiss_ranks[doc_key] = rank
    
    bm25_ranks: Dict[Tuple[int, str], int] = {}
    for rank, c in enumerate(bm25_candidates_sorted, start=1):
        doc_key = get_doc_key(c)
        bm25_ranks[doc_key] = rank
    
    # 3. 收集所有唯一文档（合并 faiss 和 bm25）
    all_doc_keys = set(faiss_ranks.keys()) | set(bm25_ranks.keys())
    
    # 4. 计算每个文档的 RRF 分数
    rrf_scores: Dict[Tuple[int, str], float] = {}
    for doc_key in all_doc_keys:
        rrf_score = 0.0
        # 如果文档在 faiss 中，加上 faiss 的 RRF 贡献
        if doc_key in faiss_ranks:
            rrf_score += (1.0 - alpha) / (rrf_k + faiss_ranks[doc_key])
        # 如果文档在 bm25 中，加上 bm25 的 RRF 贡献
        if doc_key in bm25_ranks:
            rrf_score += alpha / (rrf_k + bm25_ranks[doc_key])
        rrf_scores[doc_key] = rrf_score
    
    # 5. 为每个候选文档添加 RRF 分数，并去重（同一文档可能同时出现在 faiss 和 bm25 中）
    # 优先保留分数更高的那个（或者可以合并信息）
    merged_candidates: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for c in candidates:
        doc_key = get_doc_key(c)
        if doc_key not in merged_candidates:
            merged_candidates[doc_key] = c.copy()
            merged_candidates[doc_key]["score_rrf"] = rrf_scores.get(doc_key, 0.0)
            merged_candidates[doc_key]["faiss_rank"] = faiss_ranks.get(doc_key)
            merged_candidates[doc_key]["bm25_rank"] = bm25_ranks.get(doc_key)
        else:
            # 如果已存在，可以选择保留分数更高的，或者合并信息
            # 这里简单保留第一个遇到的
            pass
    # 6. 按 RRF 分数排序（降序）
    candidates = list(merged_candidates.values())
    candidates.sort(key=lambda x: x["score_rrf"], reverse=True)
    
    # 为了兼容性，保留 score_blend 字段（使用 RRF 分数）
    for c in candidates:
        c["score_blend"] = c["score_rrf"]
    
    #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    log_print("------------------------------ [final Candidates] ------------------------------")
    if candidates:
        for i, c in enumerate(candidates):
            if i<30:
                log_print(f"  [{i+1}] rid={c['rid']}, score_blend={c['score_blend']:.6f},norm={c['score_norm']:.6f}, modal={c['modal']},dir={c['dir']},content={c['meta'].get('raw_content')}")
            else:
                break

    # 4. Table Dedup (Logic: Keep first occurrence of any table_id)
    deduped_candidates: List[Dict[str, Any]] = []
    seen_table_ids = set()
    for c in candidates:
        m = c["meta"]
        t = m.get("table") or {}
        if isinstance(t, dict) and t.get("id"):
            tid = str(t.get("id"))
            if tid in seen_table_ids:
                continue
            seen_table_ids.add(tid)
        deduped_candidates.append(c)

    pre = deduped_candidates[: pre_topk]
    
    # Dedup and assemble docs
    seen_ids = set()
    pre_unique: List[Dict[str, Any]] = []
    for c in pre:
        mid = str(c["meta"].get("id"))
        if mid in seen_ids:
            continue
        seen_ids.add(mid)
        pre_unique.append(c)
    docs = []
    for c in pre_unique:
        m = c["meta"]
        docs.append(m.get("raw_content") or m.get("embed_text") or "")
    # If no documents to rerank, short-circuit to avoid API 400
    if not docs:
        return {"intent": intent.to_dict(), "results": [], "trace": trace.to_dict()}
    rerank_pairs = call_qwen_rerank(query, docs, rerank_topk, rerank_instruct)
    final: List[Dict[str, Any]] = []
    for doc_idx, rscore in rerank_pairs:
        if not (0 <= doc_idx < len(pre_unique)):
            continue
        c = pre_unique[doc_idx]
        m = c["meta"]
        final.append({
            "rerank_score": float(rscore),
            "modal": c["modal"],
            "id": m.get("id"),
            "doc_id": m.get("doc_id"),
            "page": m.get("page"),
            "type": m.get("type"),
            "order": m.get("order"),
            "content": m.get("raw_content") or m.get("embed_text"),
            "table": m.get("table"),
            "meta": m.get("meta"),
            "source_file": m.get("source_file"),
            "index_dir": c["dir"],
        })

    # 3. Context Restoration & Expansion
    restorer = ContextRestorer(neighbor_radius=neighbor_radius, enable_table_restoration=True)
    for item in final:
        try:
            expanded_content = restorer.restore(item)
            item["content"] = expanded_content
            item["is_extended"] = True
        except Exception:
            pass

    return {"intent": intent.to_dict(), "results": final, "trace": trace.to_dict()}


def route_and_search_multiquery_hyde(
    query: str,
    # 每路检索 topK，可前端调
    per_query_topk: int = 5,
    # Multi-Query/HyDE 开关
    use_multiquery: bool = True,
    use_hyde: bool = True,
    # 是否并行执行多路检索
    parallel: bool = True,
    # 其余检索参数（与 route_and_search 对齐，采用合理默认）
    alpha: float = 0.5,
    pre_topk: int = 30,
    faiss_per_index: int = 10,
    bm25_per_index: int = 50,
    rerank_topk: int = 10,
    rerank_instruct: str = "Given a web search query, retrieve relevant passages that answer the query.",
    neighbor_radius: int = 1,
    # return_table_full removed
    prefer_year: bool = True,
    doc_type_override: Optional[str] = None,  # "report" | "notice" | "both" | None
    run_faiss: bool = True,
    run_bm25: bool = True,
) -> Dict[str, Any]:
    """
    针对 原始query + Multi-Query扩展 + HyDE passage 分别独立检索（默认并行），
    每路取 per_query_topk，最后合并去重后返回。
    """
    # 1) 先拿原始意图（用于返回/观测）
    _intent_pack = route_and_search(
        query=query,
        alpha=alpha,
        pre_topk=pre_topk,
        faiss_per_index=faiss_per_index,
        bm25_per_index=bm25_per_index,
        rerank_topk=rerank_topk,
        rerank_instruct=rerank_instruct,
        neighbor_radius=neighbor_radius,
        prefer_year=prefer_year,
        doc_type_override=doc_type_override,
        run_faiss=run_faiss,
        run_bm25=run_bm25,
    )
    base_intent = _intent_pack.get("intent") or {}
    base_results = _intent_pack.get("results") or []

    # 2) Multi-Query 与 HyDE 文档（延后导入生成函数，避免循环导入问题）
    expansions: Dict[int, str] = {}
    hyde_passage: str = ""
    try:
        expansions = generate_multi_queries_zh(query) if use_multiquery else {}
    except Exception:
        expansions = {}
    try:
        hyde_passage = generate_hyde_passage_zh(query) if use_hyde else ""
    except Exception:
        hyde_passage = ""

    # 待检索列表：原始query + 扩展 + HyDE（若非空）
    tasks: List[Tuple[str, str]] = [("orig", query)]
    for mid, q in expansions.items():
        if q:
            tasks.append((f"mq_{mid}", q))
    if isinstance(hyde_passage, str) and hyde_passage.strip():
        # HyDE 用虚构段落直接作为查询向量化
        tasks.append(("hyde", hyde_passage.strip()))

    # 3) 多路检索（并行/串行）
    per_key_results: Dict[str, List[Dict[str, Any]]] = {}

    def _do_search(qkey: str, qtext: str) -> List[Dict[str, Any]]:
        try:
            pack = route_and_search(
                query=qtext,
                alpha=alpha,
                pre_topk=pre_topk,
                faiss_per_index=faiss_per_index,
                bm25_per_index=bm25_per_index,
                rerank_topk=rerank_topk,
                rerank_instruct=rerank_instruct,
                neighbor_radius=neighbor_radius,
                prefer_year=prefer_year,
                doc_type_override=doc_type_override,
                run_faiss=run_faiss,
                run_bm25=run_bm25,
            )
            return list(pack.get("results") or [])
        except Exception:
            return []

    if parallel and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=min(8, len(tasks))) as ex:
            futs = {ex.submit(_do_search, key, text): key for key, text in tasks}
            for fut in as_completed(futs):
                key = futs[fut]
                try:
                    per_key_results[key] = fut.result()
                except Exception:
                    per_key_results[key] = []
    else:
        for key, text in tasks:
            per_key_results[key] = _do_search(key, text)

    # 4) 合并去重（按 id 去重，保持各路顺序，优先原始query）
    merged: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    order_keys = ["orig"] + [k for k, _ in tasks if k.startswith("mq_")] + (["hyde"] if "hyde" in per_key_results else [])
    for k in order_keys:
        for item in per_key_results.get(k, []):
            mid = str(item.get("id"))
            if mid in seen_ids:
                continue
            seen_ids.add(mid)
            merged.append(item)

    return {
        "intent": base_intent,
        "expansions": expansions,  # {方法编号: 扩展后的单条查询}
        "hyde_passage": hyde_passage,
        "per_query_results": per_key_results,
        "merged_results": merged,
        "base_results": base_results,
    }

def _truncate_text(s: str, limit: int) -> str:
    if not isinstance(s, str):
        return ""
    if limit is None or limit <= 0:
        return s
    if len(s) <= limit:
        return s
    return s[: limit] + "…"


def build_answer_messages(
    query: str,
    contexts: List[Dict[str, Any]],
    system_prompt: str,
    per_chunk_limit: int,
    include_full_table: bool = True, # Default True per user request (logic enabled)
    max_total_length: int = 120000,  # API限制129024，留余量
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    # 拼装中文回答指令，包含引用片段
    lines: List[str] = []
    # 记录处理后的上下文信息，用于返回源资料
    processed_contexts: List[Dict[str, Any]] = []

    for i, c in enumerate(contexts, start=1):
        doc_id = str(c.get("doc_id") or "")
        page = c.get("page")
        typ = str(c.get("type") or "")
        
        # Content is already processed by ContextRestorer in search phase
        content = str(c.get("content") or "")
        
        is_table = isinstance(c.get("table"), dict) and bool(c.get("table").get("id"))
        
        # Determine if we should truncate
        # If it is a table and include_full_table is True, we assume content is the full table (restored) 
        # and we prefer not to truncate it heavily unless necessary.
        if is_table and include_full_table:
            # Full table content usually shouldn't be truncated by chunk limit
            pass
        else:
            content = _truncate_text(content, per_chunk_limit)

        tag = f"[{i}] doc={doc_id} page={page} type={typ}".strip()
        lines.append(f"{tag}\n{content}")
        
        # 记录处理后的上下文信息
        processed_contexts.append({
            "ref": i,
            "id": c.get("id"),
            "modal": c.get("modal", "unknown"),
            "doc_id": doc_id,
            "page": page,
            "type": "table_full" if (is_table and include_full_table) else typ,
            "order": c.get("order"),
            "content": content,
            "index_dir": c.get("index_dir"),
            "source_file": c.get("source_file"),
        })

    ctx_block = "\n\n".join(lines)
    
    # 构建user_prompt的固定部分（用于估算长度）
    prompt_prefix = "请基于下列检索到的材料回答问题，确保用中文作答，并标明参考的来源内容的[编号]。\n"
    prompt_suffix = "\n\n要求：\n- 优先使用材料中的事实；若无法确定，请明确说明无法从材料中确定。\n- 简洁作答，必要时给出要点列表。\n"
    prompt_query_part = f"问题：{query}\n\n材料：\n"
    
    # 计算可用长度
    fixed_length = len(system_prompt) + len(prompt_prefix) + len(prompt_suffix) + len(prompt_query_part)
    available_length = max_total_length - fixed_length - 1000  # 再留1000字符余量
    
    # 如果材料部分超过可用长度，进行截断
    if len(ctx_block) > available_length:
        ctx_block = ctx_block[:available_length - 100] + "\n\n[内容已截断，部分材料因长度限制未包含...]"
    
    user_prompt = prompt_prefix + prompt_query_part + ctx_block + prompt_suffix
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages, processed_contexts


def call_chat_completion(messages: List[Dict[str, Any]], model: str, temperature: float, max_tokens: int) -> str:
    if not QWEN_API_KEY:
        raise RuntimeError("DASHSCOPE_API_KEY/QWEN_API_KEY 未设置")
    url = f"{QWEN_API_BASE}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    req = _ureq.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {QWEN_API_KEY}")
    try:
        with _ureq.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
    except _uerr.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            err_body = ""
        raise RuntimeError(f"[LLM][HTTP {e.code}] {e.reason}\n{err_body}") from e
    obj = json.loads(body)
    # OpenAI 兼容返回结构
    try:
        content = obj["choices"][0]["message"]["content"]
        return str(content)
    except Exception:
        return json.dumps(obj, ensure_ascii=False)


# ---------------- Multi-Query 与 HyDE 生成（中文优化版） ----------------
_MQ_INSTRUCTIONS: Dict[int, str] = {
    1: (
        "在年度报告与财务报表的语境下，用同义词或相关术语替换关键术语，保持原意一致。"
        "输出1条改写查询，必须仅输出并用尖括号<>包裹，不要添加任何解释或多余字符。"
    ),
    2: (
        "在年度报告与财务报表的语境下，引入更宽泛或更具体的相关词（上位词/下位词），保持查询意图不变。"
        "输出1条改写查询，必须仅输出并用尖括号<>包裹，不要添加任何解释或多余字符。"
    ),
    3: (
        "在年度报告与财务报表的语境下，将该问题改写为一条保持相同意图的释义。"
        "输出1条改写查询，必须仅输出并用尖括号<>包裹，不要添加任何解释或多余字符。"
    ),
}


def _extract_angle_brackets(text: str) -> Optional[str]:
    """解析形如 <...> 的单条查询；若无则返回 None。"""
    if not isinstance(text, str):
        return None
    m = re.search(r"<\s*(.+?)\s*>", text, flags=re.DOTALL)
    if not m:
        return None
    q = m.group(1).strip()
    # 清理潜在的首尾引号或多余空格
    q = re.sub(r'^[\"“”\'`]+|[\"“”\'`]+$', "", q).strip()
    return q or None


def generate_multi_queries_zh(
    query: str,
    methods: List[int] = [1, 2, 3],
    model: str = QWEN_CHAT_MODEL,
) -> Dict[int, str]:
    """生成多路中文扩展查询：每种方法返回1条。严格要求仅输出 `<...>`。"""
    out: Dict[int, str] = {}
    for mid in methods:
        instr = _MQ_INSTRUCTIONS.get(mid)
        if not instr:
            continue
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一名中文金融检索提示词优化器。"
                    "请严格遵守输出格式：只输出一条查询，并使用尖括号<>包裹，不得添加任何其他说明、编号或标点。"
                    "语境限定在上市公司年度报告、定期报告、财务报表与相关公告。"
                ),
            },
            {
                "role": "user",
                "content": f"{instr}\n原问题：{query}",
            },
        ]
        try:
            resp = call_chat_completion(messages, model=model, temperature=0.2, max_tokens=128)
        except Exception:
            resp = ""
        parsed = _extract_angle_brackets(resp)
        if parsed:
            out[mid] = parsed
    return out


_HYDE_SYSTEM_ZH = (
    "你是一名富有创造力的报告写作者。"
    "当被问到问题时，你的任务不是检索实时或真实的财务数据，"
    "而是以“编造、整合或模拟”的方式，撰写一段有助于回答问题的段落、分析或新闻风格报道。"
    "即使问题要求真实数值或不可用的信息，也应生成连贯且语境相关的叙事内容，"
    "而不是以无数据为由进行拒绝或免责声明。"
    "目标：产出一篇文风成熟的作品（如报道、分析或评论），读起来仿佛出自报刊、杂志或研究评论。"
)


def generate_hyde_passage_zh(
    query: str,
    model: str = QWEN_CHAT_MODEL,
) -> str:
    """生成 HyDE 虚构段落（中文）。"""
    messages = [
        {"role": "system", "content": _HYDE_SYSTEM_ZH},
        {
            "role": "user",
            "content": f"请以资讯报道/分析评论/研究解读的叙事方式，写一段完整且信息量足的文字来回应此问题：{query}",
        },
    ]
    try:
        resp = call_chat_completion(messages, model=model, temperature=0.7, max_tokens=640)
        return str(resp).strip()
    except Exception:
        return ""


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search over FAISS and BM25 indexes")
    p.add_argument("--variant", choices=["exp", "ctrl"], required=False, default="exp")
    # Single-index dirs (backward compatible) OR roots containing multiple sub-indexes
    p.add_argument("--faiss-dir", required=False, default="")
    p.add_argument("--bm25-dir", required=False, default="")
    # Knowledge base shortcut
    p.add_argument("--kb", choices=["finance", "财报"], default=None, help="选择知识库：finance/财报 将映射到金盘财报_indexes/")
    # Query and params
    p.add_argument("--query", required=True)
    # topk argument removed
    p.add_argument("--alpha", type=float, default=0.5, help="混合打分时 BM25 的权重")
    p.add_argument("--pre-topk", type=int, default=30, help="BM25+Embedding 预选 topK")
    p.add_argument("--faiss-per-index", type=int, default=50, help="每个子库向量检索 topK")
    p.add_argument("--bm25-per-index", type=int, default=50, help="每个子库 BM25 检索 topK")
    p.add_argument("--rerank-topk", type=int, default=30, help="Rerank 返回 topK")
    p.add_argument("--rerank-instruct", default="Given a web search query, retrieve relevant passages that answer the query.")
    p.add_argument("--mode", choices=["hybrid", "faiss", "bm25"], default="hybrid")
    p.add_argument("--filter-doc", default="")
    p.add_argument("--filter-type", default="")
    p.add_argument("--page-from", type=int, default=None)
    p.add_argument("--page-to", type=int, default=None)
    p.add_argument("--neighbor-radius", type=int, default=1, help="返回上下相邻 chunk 的半径（0 关闭）")
    # return-table-full removed (logic implicit)
    p.add_argument("--output-cap", type=int, default=200, help="最多输出条数上限（含邻近/整表扩展）")
    # Answer generation
    p.add_argument("--gen-answer", action="store_true", help="基于TopK检索结果调用大模型生成回答")
    # answer-topk removed
    p.add_argument("--answer-model", default=None, help="覆盖默认聊天模型 QWEN_CHAT_MODEL")
    p.add_argument("--answer-max-tokens", type=int, default=512)
    p.add_argument("--answer-temperature", type=float, default=0.1)
    p.add_argument("--answer-system", default="你是严谨的中文金融助理。基于给定检索片段回答，若无法确定，请明确说明。引用用 [编号] 标注。")
    p.add_argument("--answer-max-chars", type=int, default=1200, help="每条上下文最大字符数")
    # Output control
    p.add_argument("--quiet-results", action="store_true", help="仅输出最终结果：有回答则只打印回答JSON，否则打印合并检索结果JSON")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv[1:])
    # Map KB to root dirs if provided
    kb_faiss = args.faiss_dir
    kb_bm25 = args.bm25_dir
    if args.kb in ("finance", "财报"):
        kb_faiss = kb_faiss or "/Users/wangyaqi/Documents/cursor_project/jst-rag-demo/jst-rag-demo/金盘财报_indexes/faiss_exp"
        kb_bm25 = kb_bm25 or "/Users/wangyaqi/Documents/cursor_project/jst-rag-demo/jst-rag-demo/金盘财报_indexes/bm25_exp"
    if not kb_faiss or not kb_bm25:
        print("缺少 --faiss-dir/--bm25-dir 或未选择有效的 --kb", file=sys.stderr)
        return 2
    res = search(
        query=args.query,
        faiss_dir=os.path.abspath(kb_faiss),
        bm25_dir=os.path.abspath(kb_bm25),
        # topk removed
        alpha=args.alpha,
        mode=args.mode,
        filter_doc=args.filter_doc,
        filter_type=args.filter_type,
        page_from=args.page_from,
        page_to=args.page_to,
        pre_topk=args.pre_topk,
        faiss_per_index=args.faiss_per_index,
        bm25_per_index=args.bm25_per_index,
        rerank_topk=args.rerank_topk,
        rerank_instruct=args.rerank_instruct,
        neighbor_radius=args.neighbor_radius,
        # return_table_full removed
    )
    if args.quiet_results:
        if args.gen_answer:
            used = res # use all rerank results
            messages, processed_contexts = build_answer_messages(
                query=args.query,
                contexts=used,
                system_prompt=args.answer_system,
                per_chunk_limit=args.answer_max_chars,
                include_full_table=True,
            )
            chat_model = args.answer_model or QWEN_CHAT_MODEL
            answer = call_chat_completion(
                messages=messages,
                model=chat_model,
                temperature=args.answer_temperature,
                max_tokens=args.answer_max_tokens,
            )
            # 使用处理后的上下文信息构建 sources，确保与提供给大模型的内容一致
            sources = processed_contexts
            print(json.dumps({"answer": answer, "sources": sources}, ensure_ascii=False))
        else:
            print(json.dumps({"results": res}, ensure_ascii=False))
    else:
        for r in res:
            print(json.dumps(r, ensure_ascii=False))
        if args.gen_answer:
            used = res # use all rerank results
            messages, processed_contexts = build_answer_messages(
                query=args.query,
                contexts=used,
                system_prompt=args.answer_system,
                per_chunk_limit=args.answer_max_chars,
                include_full_table=True,
            )
            chat_model = args.answer_model or QWEN_CHAT_MODEL
            answer = call_chat_completion(
                messages=messages,
                model=chat_model,
                temperature=args.answer_temperature,
                max_tokens=args.answer_max_tokens,
            )
            # 使用处理后的上下文信息构建 sources，确保与提供给大模型的内容一致
            sources = processed_contexts
            print(json.dumps({"answer": answer, "sources": sources}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
