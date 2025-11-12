#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chunk generator for experiment/control groups.

Experiment (exp):
  - Input: textified JSON (from 金盘上市公告_table2text)
  - Keep only types in {text, list, equation, table}
  - For table: each line in table_text becomes a standalone chunk (ignore max_chars)
  - For non-table: pack units up to max_chars (default 500) without splitting a unit; if a unit itself > max_chars, it forms an oversized single chunk

Control (ctrl):
  - Input: original JSON (from 金盘上市公告_mineru解析), using raw HTML tables (table_body)
  - Same rules for non-table
  - For table (only table_body): if length ≤ 4000, emit one chunk; if > 4000, split into two chunks by the middle; do NOT split per row; do NOT pack with 500-limit buckets

Outputs per input JSON (written into --out-dir):
  - <basename>_chunks_exp.jsonl or <basename>_chunks_ctrl.jsonl
  - <basename>_tables.json (table index shared)

CLI:
  python3 chunk_content.py <input_path> --mode exp|ctrl [--max-chars 500] [--out-dir DIR]
  - input_path can be a single file or a directory; directory mode recursively matches *_content_list.json

Pure standard library implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import hashlib
from typing import Any, Dict, Iterable, List, Optional, Tuple

from intent.intent_rules import (
    classify_doc_type as _classify_doc_type_rules,
    QTR_PATTERN as _QTR_PATTERN,
    YEAR_PATTERN as _YEAR_PATTERN,
    RANGE_PATTERN as _RANGE_PATTERN,
)

ACCEPT_TYPES = {"text", "table", "list", "equation"}
CONTENT_LIST_SUFFIX = "_content_list.json"


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def sha1_digest(s: str) -> str:
    h = hashlib.sha1()
    h.update(s.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def iter_input_files(root_or_file: str) -> Iterable[str]:
    if os.path.isfile(root_or_file):
        yield os.path.abspath(root_or_file)
        return
    for dirpath, _dirnames, filenames in os.walk(root_or_file):
        for fn in filenames:
            if fn.endswith(CONTENT_LIST_SUFFIX):
                yield os.path.join(dirpath, fn)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def get_doc_id(file_path: str) -> str:
    base = os.path.basename(file_path)
    if base.endswith(CONTENT_LIST_SUFFIX):
        return base[: -len(CONTENT_LIST_SUFFIX)]
    name, _ = os.path.splitext(base)
    return name


def _detect_doc_type_by_path_title(path: str, title: str) -> Optional[str]:
    p = path or ""
    t = title or ""
    if "公告" in p or "上市公告" in p or "公告" in t:
        return "notice"
    dtype, conf = _classify_doc_type_rules(t)
    if dtype != "unknown" and conf >= 0.4:
        return dtype
    if ("财报" in p) or any(k in t for k in ["年报", "季报", "季度报告", "年度报告", "述职报告", "工作报告"]):
        return "report"
    return None


def _extract_year_quarter_from_title(doc_id: str) -> Dict[str, Any]:
    prefix_year: Optional[int] = None
    m = re.match(r"(?P<date>\d{4})-(\d{2})-(\d{2})\s+", doc_id)
    rest = doc_id
    if m:
        prefix_year = int(m.group("date"))
        rest = doc_id[m.end():]
    title = rest
    years: List[int] = []
    for rm in _RANGE_PATTERN.finditer(title):
        y1 = int(rm.group(1))
        y2 = int(rm.group(3))
        if y1 > y2:
            y1, y2 = y2, y1
        years.extend(list(range(y1, y2 + 1)))
    for ym in _YEAR_PATTERN.finditer(title):
        y = int(ym.group(0))
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
    is_multi_year = len(set(years)) > 1
    return {
        "year": primary_year,
        "quarter": quarter,
        "years_mentioned": years_mentioned,
        "year_source": year_source,
        "is_multi_year": is_multi_year,
        "title": title,
        "prefix_year": prefix_year,
    }


def _build_doc_meta(source_path: str, doc_id: str) -> Dict[str, Any]:
    year_info = _extract_year_quarter_from_title(doc_id)
    doc_type = _detect_doc_type_by_path_title(source_path, year_info.get("title", "") or "")
    source_date = None
    m = re.match(r"(\d{4}-\d{2}-\d{2})\s+", doc_id)
    if m:
        source_date = m.group(1)
    meta = {
        "source_path": source_path,
        "doc_type": doc_type,
        "year": year_info.get("year"),
        "quarter": year_info.get("quarter"),
        "source_date": source_date,
        "year_source": year_info.get("year_source"),
        "years_mentioned": year_info.get("years_mentioned"),
        "is_multi_year": year_info.get("is_multi_year"),
    }
    return meta


def get_page(obj: Dict[str, Any]) -> Optional[int]:
    for key in ("page_idx", "page"):
        v = obj.get(key)
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.isdigit():
            try:
                return int(v)
            except ValueError:
                pass
    return None


def extract_text_for_unit(obj: Dict[str, Any], typ: str) -> str:
    """Best-effort extraction for text/list/equation unit text from parsed JSON."""
    if typ == "list":
        items = obj.get("items")
        if isinstance(items, list):
            parts: List[str] = []
            for it in items:
                if isinstance(it, str):
                    parts.append(normalize_space(it))
                elif isinstance(it, dict):
                    for k in ("text", "content", "value", "raw"):
                        if isinstance(it.get(k), str):
                            parts.append(normalize_space(it[k]))
                            break
            if parts:
                return "；".join([p for p in parts if p])
    # fallbacks for text/equation/list
    for k in ("text", "content", "value", "raw"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return normalize_space(v)
    return ""


def walk_collect_units(x: Any, accept_types: set) -> Iterable[Tuple[str, Optional[int], Dict[str, Any]]]:
    """Yield (type, page, obj) for accepted types."""
    if isinstance(x, dict):
        typ = x.get("type")
        if isinstance(typ, str) and typ in accept_types:
            yield (typ, get_page(x), x)
        for v in x.values():
            yield from walk_collect_units(v, accept_types)
    elif isinstance(x, list):
        for v in x:
            yield from walk_collect_units(v, accept_types)


def chunk_non_table_units(doc_id: str, page_and_texts: List[Tuple[Optional[int], str]], max_chars: int, source_path: str,
                          start_idx: int = 1) -> Tuple[List[Dict[str, Any]], int]:
    chunks: List[Dict[str, Any]] = []
    current: List[Tuple[Optional[int], str]] = []
    current_len = 0
    idx = start_idx
    base_meta = _build_doc_meta(source_path, doc_id)
    for page, text in page_and_texts:
        if not text:
            continue
        l = len(text)
        if l > max_chars:
            # flush current
            if current:
                content = "\n".join([t for _p, t in current])
                pages = [p for p, _t in current if p is not None]
                chunk_page = pages[0] if pages else None
                chunks.append({
                    "id": f"{doc_id}#p{chunk_page}#c{idx}" if chunk_page is not None else f"{doc_id}#c{idx}",
                    "doc_id": doc_id,
                    "type": "text",
                    "page": chunk_page,
                    "content": content,
                    "meta": dict(base_meta),
                })
                idx += 1
                current = []
                current_len = 0
            # oversized unit by itself
            chunks.append({
                "id": f"{doc_id}#p{page}#c{idx}" if page is not None else f"{doc_id}#c{idx}",
                "doc_id": doc_id,
                "type": "text",
                "page": page,
                "content": text,
                "meta": dict(base_meta),
            })
            idx += 1
            continue
        # normal unit
        if current_len and current_len + 1 + l > max_chars:
            # flush
            content = "\n".join([t for _p, t in current])
            pages = [p for p, _t in current if p is not None]
            chunk_page = pages[0] if pages else None
            chunks.append({
                "id": f"{doc_id}#p{chunk_page}#c{idx}" if chunk_page is not None else f"{doc_id}#c{idx}",
                "doc_id": doc_id,
                "type": "text",
                "page": chunk_page,
                "content": content,
                "meta": dict(base_meta),
            })
            idx += 1
            current = []
            current_len = 0
        if not current:
            current.append((page, text))
            current_len = l
        else:
            current.append((page, text))
            current_len += 1 + l  # add newline separator cost
    if current:
        content = "\n".join([t for _p, t in current])
        pages = [p for p, _t in current if p is not None]
        chunk_page = pages[0] if pages else None
        chunks.append({
            "id": f"{doc_id}#p{chunk_page}#c{idx}" if chunk_page is not None else f"{doc_id}#c{idx}",
            "doc_id": doc_id,
            "type": "text",
            "page": chunk_page,
            "content": content,
            "meta": dict(base_meta),
        })
        idx += 1
    return chunks, idx


def process_file(file_path: str, mode: str, max_chars: int, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    data = read_json(file_path)
    doc_id = get_doc_id(file_path)
    source_path = os.path.abspath(file_path)

    tables_index: Dict[str, Dict[str, Any]] = {}
    chunks: List[Dict[str, Any]] = []
    chunk_idx = 1
    order_idx = 1  # global sequential order per document
    table_seq_by_page: Dict[Optional[int], int] = {}
    global_table_seq = 0

    # collect non-table units as (page, text) – staged sequentially, flushed at table boundaries
    staged_non_table: List[Tuple[Optional[int], str]] = []

    def flush_non_table() -> None:
        nonlocal chunk_idx, chunks, staged_non_table, order_idx
        if not staged_non_table:
            return
        packed, next_idx = chunk_non_table_units(
            doc_id, staged_non_table, max_chars, source_path, start_idx=chunk_idx
        )
        # assign sequential order to packed chunks in original sequence
        for ch in packed:
            ch["order"] = order_idx
            order_idx += 1
        chunks.extend(packed)
        chunk_idx = next_idx
        staged_non_table = []

    for typ, page, obj in walk_collect_units(data, ACCEPT_TYPES):
        if typ == "table":
            # before handling table, flush current staged non-table to prevent non-contiguous packing
            flush_non_table()
            # allocate table seq (global within doc)
            global_table_seq += 1
            t_seq = global_table_seq
            table_id = f"{doc_id}::p{page}::t{t_seq}" if page is not None else f"{doc_id}::t{t_seq}"
            table_text = obj.get("table_text") if isinstance(obj.get("table_text"), str) else None
            table_body = obj.get("table_body") if isinstance(obj.get("table_body"), str) else None

            # index record
            digest_src = None
            if mode == "exp":
                digest_src = normalize_space(table_text or "")
            else:  # ctrl uses original html tables
                digest_src = normalize_space(table_body or "")
            tables_index[table_id] = {
                "id": table_id,
                "doc_id": doc_id,
                "page": page,
                "seq": t_seq,
                "table_text": table_text,
                "table_body": table_body,
                "digest": sha1_digest(digest_src or ""),
            }

            if mode == "exp":
                # each line becomes a chunk (ignore max_chars)
                lines = []
                if isinstance(table_text, str):
                    lines = [normalize_space(x) for x in table_text.split("\n")]
                base_meta = _build_doc_meta(source_path, doc_id)
                row_index = 0
                for line in lines:
                    if not line:
                        continue
                    row_index += 1
                    chunks.append({
                        "id": f"{doc_id}#p{page}#t{t_seq}#r{row_index}" if page is not None else f"{doc_id}#t{t_seq}#r{row_index}",
                        "doc_id": doc_id,
                        "type": "table_row",
                        "page": page,
                        "content": line,
                        "table": {
                            "id": table_id,
                            "seq": t_seq,
                            "page": page,
                            "row_index": row_index,
                            "digest": tables_index[table_id]["digest"],
                        },
                        "meta": dict(base_meta),
                    })
                    chunks[-1]["order"] = order_idx
                    order_idx += 1
            else:
                # ctrl: use original HTML table; split rule: <=4000 as single chunk; >4000 split into 2 parts
                content = table_body or ""
                if not content:
                    continue
                base_meta = _build_doc_meta(source_path, doc_id)
                l = len(content)
                if l <= 4000:
                    chunks.append({
                        "id": f"{doc_id}#p{page}#c{chunk_idx}" if page is not None else f"{doc_id}#c{chunk_idx}",
                        "doc_id": doc_id,
                        "type": "table",
                        "page": page,
                        "content": content,
                        "table": {
                            "id": table_id,
                            "seq": t_seq,
                            "page": page,
                            "digest": tables_index[table_id]["digest"],
                            "part": 1,
                            "parts": 1,
                        },
                        "meta": dict(base_meta),
                    })
                    chunks[-1]["order"] = order_idx
                    order_idx += 1
                    chunk_idx += 1
                else:
                    mid = l // 2
                    part1 = content[:mid]
                    part2 = content[mid:]
                    chunks.append({
                        "id": f"{doc_id}#p{page}#c{chunk_idx}" if page is not None else f"{doc_id}#c{chunk_idx}",
                        "doc_id": doc_id,
                        "type": "table",
                        "page": page,
                        "content": part1,
                        "table": {
                            "id": table_id,
                            "seq": t_seq,
                            "page": page,
                            "digest": tables_index[table_id]["digest"],
                            "part": 1,
                            "parts": 2,
                        },
                        "meta": dict(base_meta),
                    })
                    chunks[-1]["order"] = order_idx
                    order_idx += 1
                    chunk_idx += 1
                    chunks.append({
                        "id": f"{doc_id}#p{page}#c{chunk_idx}" if page is not None else f"{doc_id}#c{chunk_idx}",
                        "doc_id": doc_id,
                        "type": "table",
                        "page": page,
                        "content": part2,
                        "table": {
                            "id": table_id,
                            "seq": t_seq,
                            "page": page,
                            "digest": tables_index[table_id]["digest"],
                            "part": 2,
                            "parts": 2,
                        },
                        "meta": dict(base_meta),
                    })
                    chunks[-1]["order"] = order_idx
                    order_idx += 1
                    chunk_idx += 1
        else:
            text = extract_text_for_unit(obj, typ)
            if not text:
                continue
            staged_non_table.append((page, text))

    # Flush any remaining non-table units at end
    flush_non_table()

    # emit files
    base = os.path.basename(file_path)
    out_chunks = os.path.join(out_dir, f"{base}_chunks_{mode}.jsonl")
    out_tables = os.path.join(out_dir, f"{base}_tables.json")
    write_jsonl(out_chunks, chunks)
    write_json(out_tables, tables_index)


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate chunks for experiment/control groups")
    p.add_argument("input_path", help="Input file or directory. Directory mode recursively matches *_content_list.json")
    p.add_argument("--mode", choices=["exp", "ctrl"], required=True, help="Experiment (exp) or Control (ctrl)")
    p.add_argument("--max-chars", type=int, default=500, help="Max characters per chunk for non-table units")
    p.add_argument("--out-dir", default=None, help="Output directory. Default: <input>_chunks_{mode} for directory; file's dir for single file")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv[1:])
    input_path = os.path.abspath(args.input_path)
    if not os.path.exists(input_path):
        print(f"找不到输入路径: {input_path}")
        return 2
    # decide out_dir
    if args.out_dir:
        out_dir = os.path.abspath(args.out_dir)
    else:
        if os.path.isfile(input_path):
            out_dir = os.path.dirname(input_path)
        else:
            out_dir = f"{input_path}_chunks_{args.mode}"
    os.makedirs(out_dir, exist_ok=True)

    files = list(iter_input_files(input_path))
    if not files:
        print("未找到 *_content_list.json 文件")
        return 0
    print(f"将处理 {len(files)} 个文件，模式: {args.mode}，输出目录: {out_dir}")
    for idx, fp in enumerate(files, start=1):
        try:
            print(f"[{idx}/{len(files)}] -> {os.path.relpath(fp, os.path.dirname(input_path))}")
            process_file(fp, args.mode, args.max_chars, out_dir)
        except Exception as e:
            print(f"[{idx}/{len(files)}] 失败: {fp} | {e}")
    print("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


