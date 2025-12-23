#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build FAISS and BM25 indexes from chunk JSONL files.

- Input: directory containing *.jsonl chunks (one JSON per line), produced by chunk_content.py
- Embeddings: Qwen Embedding v4 via OpenAI-compatible API (env: QWEN_API_KEY, QWEN_API_BASE)
- Vector index: FAISS IndexFlatIP with L2-normalized vectors
- BM25: rank_bm25 corpus; tokens persisted via pickle; uses a simple mixed tokenizer

Usage examples:
  # experiment group (no HTML stripping)
  python3 build_indexes.py \
    --chunks-dir /Users/wangyaqi/Documents/cursor_project/jst-rag-demo/jst-rag-demo/chunks_exp \
    --out-faiss /Users/wangyaqi/Documents/cursor_project/jst-rag-demo/jst-rag-demo/indexes/faiss_exp \
    --out-bm25  /Users/wangyaqi/Documents/cursor_project/jst-rag-demo/jst-rag-demo/indexes/bm25_exp

  # control group (strip HTML for embedding, keep raw HTML in metadata for display)
  python3 build_indexes.py \
    --chunks-dir /Users/wangyaqi/Documents/cursor_project/jst-rag-demo/jst-rag-demo/chunks_ctrl \
    --out-faiss /Users/wangyaqi/Documents/cursor_project/jst-rag-demo/jst-rag-demo/indexes/faiss_ctrl \
    --out-bm25  /Users/wangyaqi/Documents/cursor_project/jst-rag-demo/jst-rag-demo/indexes/bm25_ctrl \
    --strip-html

This script prints per-file elapsed time and accumulates total elapsed; if API returns token usage, totals are printed as well.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import html as _html
import pickle
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Optional dependencies
try:
    import faiss  # type: ignore
except Exception as _e:
    faiss = None  # will error out later if used

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception as _e:
    BM25Okapi = None  # type: ignore

import math
import numpy as np  # type: ignore
import urllib.request as _ureq
import urllib.error as _uerr

QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "sk-6a44d15e56dd4007945ccc41b97b499c")
QWEN_API_BASE = os.environ.get("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_EMBED_MODEL = os.environ.get("QWEN_EMBED_MODEL", "text-embedding-v4")


def list_jsonl_files(chunks_dir: str) -> List[str]:
    out: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(chunks_dir):
        for fn in filenames:
            if fn.endswith(".jsonl"):
                out.append(os.path.join(dirpath, fn))
    out.sort()
    return out


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


# utils for progress
def count_nonempty_lines(path: str) -> int:
    try:
        n = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n
    except Exception:
        return -1


# very light HTML stripper for embeddings
_TAG_RE = re.compile(r"<[^>]+>")


def strip_html(text: str) -> str:
    if not text:
        return ""
    s = _TAG_RE.sub(" ", text)
    s = _html.unescape(s)
    s = normalize_space(s)
    return s


def load_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _embedding_batch(texts: List[str]) -> Tuple[List[List[float]], Dict[str, int]]:
    if not QWEN_API_KEY:
        raise RuntimeError("QWEN_API_KEY is not set")
    url = f"{QWEN_API_BASE}/embeddings"
    payload = {
        "model": QWEN_EMBED_MODEL,
        "input": texts,
        "dimensions": 1024,
        "encoding_format": "float"
    }
    req = _ureq.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {QWEN_API_KEY}")
    try:
        with _ureq.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
    except _uerr.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            err_body = ""
        # 打印服务端返回，帮助快速判断是维度、输入格式还是鉴权问题
        msg = f"[EMBED][HTTP {e.code}] {e.reason} url={url}\n{err_body}"
        raise RuntimeError(msg) from e
    obj = json.loads(body)
    data = obj.get("data") or []
    vectors: List[List[float]] = []
    for item in data:
        vec = item.get("embedding")
        if isinstance(vec, list):
            vectors.append(vec)
    usage = obj.get("usage") or {}
    usage_out = {
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
    }
    return vectors, usage_out


def embed_in_batches(texts: List[str], batch_size: int = 64) -> Tuple[np.ndarray, Dict[str, int]]:
    all_vecs: List[List[float]] = []
    total_usage = {"total_tokens": 0, "prompt_tokens": 0}
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs, usage = _embedding_batch(batch)
        all_vecs.extend(vecs)
        total_usage["total_tokens"] += usage.get("total_tokens", 0)
        total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
    arr = np.array(all_vecs, dtype="float32")
    return arr, total_usage


def embed_with_progress(texts: List[str], batch_size: int, label: str) -> Tuple[np.ndarray, Dict[str, int]]:
    total = len(texts)
    all_vecs: List[List[float]] = []
    total_usage = {"total_tokens": 0, "prompt_tokens": 0}
    last_percent = -1
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        vecs, usage = _embedding_batch(batch)
        all_vecs.extend(vecs)
        total_usage["total_tokens"] += usage.get("total_tokens", 0)
        total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        done = len(all_vecs)
        percent = int(done * 100 / total) if total else 100
        if percent >= last_percent + 2 or done == total:
            sys.stdout.write(f"\r{label} {percent}% ({done}/{total})")
            sys.stdout.flush()
            last_percent = percent
    sys.stdout.write("\n")
    arr = np.array(all_vecs, dtype="float32")
    return arr, total_usage


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms


# tokenization for BM25: mixed strategy for CJK/Latin
_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+", re.UNICODE)


def bm25_tokenize(text: str) -> List[str]:
    if not text:
        return []
    tokens: List[str] = []
    for m in _TOKEN_RE.finditer(text):
        s = m.group(0)
        if re.match(r"[\u4e00-\u9fff]", s):
            tokens.extend(list(s))  # CJK: per-character tokens
        else:
            tokens.append(s.lower())
    return tokens


def build_indexes(chunks_dir: str, out_faiss: str, out_bm25: str, batch_size: int, do_strip_html: bool, force_rebuild: bool) -> None:
    if faiss is None:
        raise RuntimeError("faiss is not installed. Please install faiss-cpu")
    if BM25Okapi is None:
        raise RuntimeError("rank_bm25 is not installed. Please pip install rank-bm25")

    os.makedirs(out_faiss, exist_ok=True)
    os.makedirs(out_bm25, exist_ok=True)

    files = list_jsonl_files(chunks_dir)
    if not files:
        print("未找到 *.jsonl")
        return
    print(f"将处理 {len(files)} 个文件，输入目录: {chunks_dir}")

    total_start = time.time()
    total_tokens = 0

    # per-file indexing counters
    faiss_files_count = 0
    bm25_files_count = 0

    total_files = len(files)
    for file_idx, fp in enumerate(files, start=1):
        fname = os.path.basename(fp)
        safe = os.path.splitext(fname)[0]
        faiss_dir = os.path.join(out_faiss, safe)
        bm25_dir = os.path.join(out_bm25, safe)
        faiss_done = os.path.exists(os.path.join(faiss_dir, "faiss.index"))
        bm25_done = os.path.exists(os.path.join(bm25_dir, "bm25.pkl"))
        if not force_rebuild and faiss_done and bm25_done:
            print(f"[SKIP {file_idx}/{total_files}] {fname} 已存在索引 -> {faiss_dir} / {bm25_dir}")
            continue
        t0 = time.time()
        count = 0
        total_lines = count_nonempty_lines(fp)
        last_percent = -1
        file_texts: List[str] = []
        file_metas: List[Dict[str, Any]] = []
        for row in load_jsonl(fp):
            text = row.get("content") or ""
            # If ctrl with HTML tables, you likely want embeddings over stripped text, but keep original in metadata
            embed_text = strip_html(text) if do_strip_html else text
            file_texts.append(embed_text)
            meta = {
                "id": row.get("id"),
                "doc_id": row.get("doc_id"),
                "page": row.get("page"),
                "type": row.get("type"),
                "order": row.get("order"),
                # keep both embed_text and original content for display
                "embed_text": embed_text,
                "raw_content": text,
                "table": row.get("table"),
                "meta": row.get("meta"),
                "source_file": fp,
            }
            file_metas.append(meta)
            count += 1
            if total_lines and total_lines > 0:
                percent = int(count * 100 / total_lines)
                if percent >= last_percent + 2 or count == total_lines:
                    sys.stdout.write(f"\r[READ {file_idx}/{total_files}] {fname} {percent}% ({count}/{total_lines})")
                    sys.stdout.flush()
                    last_percent = percent
            else:
                if count % 200 == 0:
                    sys.stdout.write(f"\r[READ {file_idx}/{total_files}] {fname} {count} 行")
                    sys.stdout.flush()
        t1 = time.time()
        sys.stdout.write("\n")
        print(f"[READ] {fp} rows={count} elapsed={t1-t0:.2f}s")

        # Embed for this file with progress
        if file_texts:
            print(f"[EMBED {file_idx}/{total_files}] {fname} model={QWEN_EMBED_MODEL} strip_html={do_strip_html}")
            e0 = time.time()
            vec_np, usage = embed_with_progress(file_texts, batch_size=batch_size, label=f"[EMBED {file_idx}/{total_files}] {fname}")
            total_tokens += usage.get("total_tokens", 0)
            e1 = time.time()
            print(f"[EMBED] {fname} rows={vec_np.shape[0]} dim={vec_np.shape[1] if vec_np.size else 0} elapsed={e1-e0:.2f}s tokens+={usage.get('total_tokens',0)} total_tokens={total_tokens}")
            # Per-file FAISS
            if vec_np.size > 0:
                vecs = l2_normalize(vec_np)
                d = vecs.shape[1]
                index = faiss.IndexFlatIP(d)
                index.add(vecs)
                os.makedirs(faiss_dir, exist_ok=True)
                faiss_path = os.path.join(faiss_dir, "faiss.index")
                faiss.write_index(index, faiss_path)
                with open(os.path.join(faiss_dir, "meta.jsonl"), "w", encoding="utf-8") as f:
                    for m in file_metas:
                        f.write(json.dumps(m, ensure_ascii=False) + "\n")
                print(f"[FAISS] {fname} indexed={vecs.shape[0]} dim={d} -> {faiss_path}")
                faiss_files_count += 1
            else:
                print(f"[FAISS] {fname} no vectors to index")
            # Per-file BM25
            tokenized_corpus = [bm25_tokenize(x) for x in file_texts]
            bm25 = BM25Okapi(tokenized_corpus)
            os.makedirs(bm25_dir, exist_ok=True)
            with open(os.path.join(bm25_dir, "bm25.pkl"), "wb") as f:
                pickle.dump(bm25, f)
            with open(os.path.join(bm25_dir, "meta.jsonl"), "w", encoding="utf-8") as f:
                for m in file_metas:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
            print(f"[BM25] {fname} indexed={len(tokenized_corpus)} -> {bm25_dir}")
            bm25_files_count += 1

    # Summary
    print(f"[FAISS] files indexed: {faiss_files_count} -> {out_faiss}")
    print(f"[BM25]  files indexed: {bm25_files_count} -> {out_bm25}")

    total_elapsed = time.time() - total_start
    print(f"[TOTAL] elapsed={total_elapsed:.2f}s total_tokens={total_tokens}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build FAISS and BM25 indexes from chunks")
    p.add_argument("--chunks-dir", required=True, help="Input directory containing *.jsonl chunks")
    p.add_argument("--out-faiss", required=True, help="Output directory for FAISS index")
    p.add_argument("--out-bm25", required=True, help="Output directory for BM25 index")
    p.add_argument("--batch-size", type=int, default=10, help="Embedding batch size")
    p.add_argument("--strip-html", action="store_true", help="Strip HTML tags before embedding")
    p.add_argument("--force-rebuild", action="store_true", help="Force rebuild even if per-file indexes exist")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv[1:])
    build_indexes(
        chunks_dir=os.path.abspath(args.chunks_dir),
        out_faiss=os.path.abspath(args.out_faiss),
        out_bm25=os.path.abspath(args.out_bm25),
        batch_size=args.batch_size,
        do_strip_html=args.strip_html,
        force_rebuild=args.force_rebuild,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
