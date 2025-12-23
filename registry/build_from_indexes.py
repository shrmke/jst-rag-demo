#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from intent.intent_rules import QTR_PATTERN as _QTR_PATTERN, YEAR_PATTERN as _YEAR_PATTERN, RANGE_PATTERN as _RANGE_PATTERN


def list_index_subdirs(root: str, kind: str) -> List[str]:
    """
    kind: 'faiss' -> subdir contains faiss.index and meta.jsonl
          'bm25'  -> subdir contains bm25.pkl and meta.jsonl
    """
    out: List[str] = []
    if not root or not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        if kind == "faiss":
            if os.path.exists(os.path.join(d, "faiss.index")) and os.path.exists(os.path.join(d, "meta.jsonl")):
                out.append(d)
        else:
            if os.path.exists(os.path.join(d, "bm25.pkl")) and os.path.exists(os.path.join(d, "meta.jsonl")):
                out.append(d)
    return out


def read_first_meta(meta_path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    return json.loads(line)
                except Exception:
                    continue
    except Exception:
        pass
    return None


def infer_doc_type_from_root(root: str) -> Optional[str]:
    p = root or ""
    if "公告" in p:
        return "notice"
    if "财报" in p:
        return "report"
    return None


def extract_year_quarter_from_doc_id(doc_id: str) -> Dict[str, Any]:
    prefix_year: Optional[int] = None
    rest = doc_id
    m = re.match(r"(?P<date>\d{4})-(\d{2})-(\d{2})\s+", doc_id)
    if m:
        try:
            prefix_year = int(m.group("date"))
        except Exception:
            prefix_year = None
        rest = doc_id[m.end() :]
    title = rest
    years: List[int] = []
    # ranges first
    for rm in _RANGE_PATTERN.finditer(title):
        try:
            y1 = int(rm.group(1))
            y2 = int(rm.group(3))
        except Exception:
            continue
        if y1 > y2:
            y1, y2 = y2, y1
        years.extend(list(range(y1, y2 + 1)))
    # absolute years
    for ym in _YEAR_PATTERN.finditer(title):
        try:
            y = int(ym.group(0))
        except Exception:
            continue
        if 1900 <= y <= 2100:
            years.append(y)
    years_mentioned = sorted(set(years + ([prefix_year] if prefix_year else [])))
    primary_year: Optional[int] = None
    year_source = None
    if years:
        primary_year = years[-1]
        year_source = "title"
    elif prefix_year:
        primary_year = prefix_year
        year_source = "prefix"
    quarter: Optional[str] = None
    qm = _QTR_PATTERN.search(title)
    if qm:
        q = qm.group(0).upper()
        if q.startswith("Q") and q in {"Q1", "Q2", "Q3", "Q4"}:
            quarter = q
        else:
            if ("一" in q) or ("1" in q):
                quarter = "Q1"
            elif ("二" in q) or ("2" in q):
                quarter = "Q2"
            elif ("三" in q) or ("3" in q):
                quarter = "Q3"
            elif ("四" in q) or ("4" in q):
                quarter = "Q4"
    return {
        "year": primary_year,
        "quarter": quarter,
        "years_mentioned": years_mentioned,
        "year_source": year_source,
    }


def build_registry(
    faiss_roots: List[str],
    bm25_roots: List[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    doc_registry: Dict[str, Any] = {
        "docs": {},
        "by_type_year": {},
    }
    index_registry: Dict[str, Any] = {
        "faiss": {"global_root": faiss_roots[0] if faiss_roots else "", "by_doc": {}, "by_type_year": {}, "global_roots": faiss_roots},
        "bm25": {"global_root": bm25_roots[0] if bm25_roots else "", "by_doc": {}, "by_type_year": {}, "global_roots": bm25_roots},
    }

    def _update(doc_id: str, meta_doc: Dict[str, Any], faiss_dir: Optional[str], bm25_dir: Optional[str], dtype: Optional[str]) -> None:
        # doc meta
        doc_registry["docs"][doc_id] = {
            "doc_type": dtype,
            "year": meta_doc.get("year"),
            "quarter": meta_doc.get("quarter"),
            "source_date": meta_doc.get("source_date"),
        }
        if dtype:
            doc_registry["by_type_year"].setdefault(dtype, {})
            ykey = str(meta_doc.get("year")) if meta_doc.get("year") is not None else "unknown"
            doc_registry["by_type_year"][dtype].setdefault(ykey, [])
            if doc_id not in doc_registry["by_type_year"][dtype][ykey]:
                doc_registry["by_type_year"][dtype][ykey].append(doc_id)
        # index paths
        if faiss_dir:
            index_registry["faiss"]["by_doc"][doc_id] = faiss_dir
            if dtype:
                index_registry["faiss"]["by_type_year"].setdefault(dtype, {})
                ykey = str(meta_doc.get("year")) if meta_doc.get("year") is not None else "unknown"
                index_registry["faiss"]["by_type_year"][dtype].setdefault(ykey, [])
                if faiss_dir not in index_registry["faiss"]["by_type_year"][dtype][ykey]:
                    index_registry["faiss"]["by_type_year"][dtype][ykey].append(faiss_dir)
        if bm25_dir:
            index_registry["bm25"]["by_doc"][doc_id] = bm25_dir
            if dtype:
                index_registry["bm25"]["by_type_year"].setdefault(dtype, {})
                ykey = str(meta_doc.get("year")) if meta_doc.get("year") is not None else "unknown"
                index_registry["bm25"]["by_type_year"][dtype].setdefault(ykey, [])
                if bm25_dir not in index_registry["bm25"]["by_type_year"][dtype][ykey]:
                    index_registry["bm25"]["by_type_year"][dtype][ykey].append(bm25_dir)

    # First pass FAISS
    for root in faiss_roots:
        dtype_hint = infer_doc_type_from_root(root)
        for d in list_index_subdirs(root, "faiss"):
            meta_path = os.path.join(d, "meta.jsonl")
            first = read_first_meta(meta_path)
            if not first:
                continue
            doc_id = first.get("doc_id") or os.path.basename(d)
            meta = first.get("meta") or {}
            year = meta.get("year")
            quarter = meta.get("quarter")
            source_date = meta.get("source_date")
            if year is None or quarter is None:
                # infer from doc_id if missing
                inf = extract_year_quarter_from_doc_id(str(doc_id))
                year = inf.get("year") if year is None else year
                quarter = inf.get("quarter") if quarter is None else quarter
            dtype = meta.get("doc_type") or dtype_hint
            _update(
                doc_id=str(doc_id),
                meta_doc={"year": year, "quarter": quarter, "source_date": source_date},
                faiss_dir=d,
                bm25_dir=None,
                dtype=dtype,
            )

    # Second pass BM25
    for root in bm25_roots:
        dtype_hint = infer_doc_type_from_root(root)
        for d in list_index_subdirs(root, "bm25"):
            meta_path = os.path.join(d, "meta.jsonl")
            first = read_first_meta(meta_path)
            if not first:
                continue
            doc_id = first.get("doc_id") or os.path.basename(d)
            meta = first.get("meta") or {}
            # reuse existing if in registry
            existed = doc_registry["docs"].get(doc_id) or {}
            year = existed.get("year", meta.get("year"))
            quarter = existed.get("quarter", meta.get("quarter"))
            source_date = existed.get("source_date", meta.get("source_date"))
            if year is None or quarter is None:
                inf = extract_year_quarter_from_doc_id(str(doc_id))
                year = inf.get("year") if year is None else year
                quarter = inf.get("quarter") if quarter is None else quarter
            dtype = existed.get("doc_type") or meta.get("doc_type") or dtype_hint
            _update(
                doc_id=str(doc_id),
                meta_doc={"year": year, "quarter": quarter, "source_date": source_date},
                faiss_dir=None,
                bm25_dir=d,
                dtype=dtype,
            )

    return doc_registry, index_registry


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build registry from existing FAISS/BM25 indexes without rebuilding")
    p.add_argument("--faiss-root", action="append", default=[], help="FAISS root directory (may repeat). Example: /Users/wangyaqi/Documents/cursor_project/jst-rag-demo/jst-rag-demo/金盘财报_indexes/faiss_exp")
    p.add_argument("--bm25-root", action="append", default=[], help="BM25 root directory (may repeat). Example: /Users/wangyaqi/Documents/cursor_project/jst-rag-demo/jst-rag-demo/金盘财报_indexes/bm25_exp")
    p.add_argument("--out-dir", default="/Users/wangyaqi/Documents/cursor_project/jst-rag-demo/jst-rag-demo/registry", help="Output dir for doc_registry.json and index_registry.json")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv[1:])
    faiss_roots = [os.path.abspath(x) for x in (args.faiss_root or []) if x]
    bm25_roots = [os.path.abspath(x) for x in (args.bm25_root or []) if x]
    if not (faiss_roots or bm25_roots):
        # Provide sensible defaults if none supplied
        defaults_f = [
            "/Users/wangyaqi/Documents/cursor_project/jst-rag-demo/jst-rag-demo/金盘财报_indexes/faiss_exp",
            "/Users/wangyaqi/Documents/cursor_project/jst-rag-demo/jst-rag-demo/金盘上市公告_indexes/faiss_exp",
        ]
        defaults_b = [
            "/Users/wangyaqi/Documents/cursor_project/jst-rag-demo/jst-rag-demo/金盘财报_indexes/bm25_exp",
            "/Users/wangyaqi/Documents/cursor_project/jst-rag-demo/jst-rag-demo/金盘上市公告_indexes/bm25_exp",
        ]
        faiss_roots = [x for x in defaults_f if os.path.isdir(x)]
        bm25_roots = [x for x in defaults_b if os.path.isdir(x)]
    os.makedirs(args.out_dir, exist_ok=True)
    doc_reg, idx_reg = build_registry(faiss_roots, bm25_roots)
    with open(os.path.join(args.out_dir, "doc_registry.json"), "w", encoding="utf-8") as f:
        json.dump(doc_reg, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "index_registry.json"), "w", encoding="utf-8") as f:
        json.dump(idx_reg, f, ensure_ascii=False, indent=2)
    print(f"[REGISTRY] doc_registry.json / index_registry.json -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv))


