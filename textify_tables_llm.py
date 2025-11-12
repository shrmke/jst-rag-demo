#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
表格轻量文本化（LLM版本，基于来源文件名补全时间）：
  - 对所有 type=="table" 的对象一律走大模型；失败时回退到规则法。
  - 传入来源文件名（如“2021年第一季度报告”），当表头为相对时间（如“本报告期末/期末/本期/上年同期”等），
    要求模型补全为具体期间（如“2021年第一季度末”），使输出每行的列名天然带有时间语义。
  - 单文件输入：输出到同目录同名+.textified.json，并写入同目录 <输入>.textified.md 日志
  - 目录输入：递归处理 *_content_list.json，输出到 [输出目录] 或 <输入>_table2text，并写入 textify_report.md

说明：仅使用标准库与 Qwen OpenAI-Compatible 接口。
"""

from __future__ import annotations

import json
import os
import re
import sys
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Iterable
from urllib import request as _ureq
from urllib.error import URLError as _URLError, HTTPError as _HTTPError


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


class HTMLTableGridParser(HTMLParser):
    """将单个 <table> 的 HTML 解析为二维网格，展开 colspan/rowspan。仅作 LLM 失败时的兜底。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.inside_table: bool = False
        self.inside_tr: bool = False
        self.inside_cell: bool = False

        self.current_row: List[str] = []
        self.current_col: int = 0
        self.grid: List[List[str]] = []

        # col_index -> {text: str, rows_left: int}
        self.rowspans: Dict[int, Dict[str, Any]] = {}

        self._cell_text_parts: List[str] = []
        self._cell_colspan: int = 1
        self._cell_rowspan: int = 1

        self._table_parsed: bool = False

    def _prefill_rowspans_until_gap(self) -> None:
        while self.current_col in self.rowspans and self.rowspans[self.current_col]["rows_left"] > 0:
            self.current_row.append(self.rowspans[self.current_col]["text"])
            self.rowspans[self.current_col]["rows_left"] -= 1
            if self.rowspans[self.current_col]["rows_left"] == 0:
                self.rowspans.pop(self.current_col)
            self.current_col += 1

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        if self._table_parsed:
            return
        tag = tag.lower()
        if tag == "table":
            if not self.inside_table:
                self.inside_table = True
            return
        if not self.inside_table:
            return
        if tag == "tr":
            self.inside_tr = True
            self.current_row = []
            self.current_col = 0
            self._prefill_rowspans_until_gap()
            return
        if tag in ("td", "th") and self.inside_tr:
            self._prefill_rowspans_until_gap()
            self.inside_cell = True
            self._cell_text_parts = []
            attr_map = {k.lower(): (v if v is not None else "") for k, v in attrs}
            try:
                self._cell_colspan = int(attr_map.get("colspan", "1") or "1")
            except ValueError:
                self._cell_colspan = 1
            try:
                self._cell_rowspan = int(attr_map.get("rowspan", "1") or "1")
            except ValueError:
                self._cell_rowspan = 1
            return
        if tag == "br" and self.inside_cell:
            self._cell_text_parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if self._table_parsed:
            return
        tag = tag.lower()
        if tag == "table" and self.inside_table:
            self.inside_table = False
            self._table_parsed = True
            return
        if not self.inside_table:
            return
        if tag in ("td", "th") and self.inside_cell:
            text = normalize_space("".join(self._cell_text_parts))
            csp = max(1, self._cell_colspan)
            rsp = max(1, self._cell_rowspan)
            for k in range(csp):
                self.current_row.append(text)
                if rsp > 1:
                    self.rowspans[self.current_col + k] = {"text": text, "rows_left": rsp - 1}
            self.current_col += csp
            self.inside_cell = False
            self._cell_text_parts = []
            self._cell_colspan = 1
            self._cell_rowspan = 1
            return
        if tag == "tr" and self.inside_tr:
            self._prefill_rowspans_until_gap()
            self.grid.append(self.current_row)
            self.inside_tr = False
            self.current_row = []
            self.current_col = 0

    def handle_data(self, data: str) -> None:
        if self.inside_cell and self.inside_table and data:
            self._cell_text_parts.append(data)

    def get_grid(self) -> List[List[str]]:
        width = max((len(r) for r in self.grid), default=0)
        if width == 0:
            return []
        normalized: List[List[str]] = []
        for r in self.grid:
            rr = list(r)
            if len(rr) < width:
                rr += [""] * (width - len(rr))
            normalized.append([normalize_space(c) for c in rr])
        return normalized


def grid_to_text(grid: List[List[str]]) -> str:
    if not grid:
        return ""
    header_idx: Optional[int] = None
    for idx, row in enumerate(grid):
        non_empty_cells = [c for c in row if normalize_space(c)]
        if len(non_empty_cells) >= 2:
            header_idx = idx
            break
    lines: List[str] = []
    if header_idx is None:
        width = max((len(r) for r in grid), default=0)
        headers = [f"列{j+1}" for j in range(width)]
        data_rows = grid
    else:
        headers = [normalize_space(c) or f"列{j+1}" for j, c in enumerate(grid[header_idx])]
        width = len(headers)
        data_rows = grid[header_idx + 1 :]
    for r in data_rows:
        parts: List[str] = []
        r_expanded = list(r) + [""] * (width - len(r))
        for j in range(width):
            v = normalize_space(r_expanded[j])
            if not v:
                continue
            col_name = normalize_space(headers[j]) or f"列{j+1}"
            s = f"{col_name}: {v}"
            parts.append(s)
        if parts:
            line = "，".join(parts)
            if not line.endswith(("。", ".")):
                line += "。"
            lines.append(line)
    return "\n".join(lines)


def table_html_to_text(html: str) -> str:
    parser = HTMLTableGridParser()
    parser.feed(html or "")
    grid = parser.get_grid()
    return grid_to_text(grid)


# ---------------- Qwen-Plus 配置 ----------------
QWEN_API_KEY = os.environ.get("QWEN_API_KEY") or "sk-6a44d15e56dd4007945ccc41b97b499c"
QWEN_API_BASE = os.environ.get("QWEN_API_BASE") or "https://dashscope.aliyuncs.com/compatible-mode/v1"


# _LLM_PROMPT = '''
# 你是表格重排助手。输入会提供【来源文件名】与一段 HTML 表格，可能包含合并单元格、多级表头、标题/分组/单位/口径等说明行，以及行标题与数据区。请理解表格内容并将其转换为纯文本，每行记录一条相互独立的信息，遵循：

# 1) 行格式：用“中文逗号”分隔的“列名: 值”对，行尾必须加“。”；禁止输出 HTML/Markdown。
# 2) 多级表头：合并为“父-子”；无法确定列名时使用“列1/列2/…”，一般不得回退为“列N”。
# 3) 合并单元格/行列标题：将其语义展开至对应数据行；若为标题/分组/单位/口径等说明行（仅标题文本或仅一格非空），独立成行保留。
# 4) 关键规则：当表头或列名中出现相对时间用语（如“本报告期/本报告期末/期初/期末/本期/上期/上年同期/上年年末”等），请结合【来源文件名】推断具体期间并补全到列名文本中：
#    - 例：来源文件名为“2021年第一季度报告…”，列名“本报告期末”补为“2021年第一季度末”。
# 5) 仅输出纯文本结果，一行一条记录或说明行，不要添加多余解释或前后缀。
# '''

_LLM_PROMPT = '''
你是一名财务数据整理专家，任务是将财务报表中的表格转换为可理解的自然语言描述，用于后续的语义检索系统（RAG）中使用。

输入是一张来自公司财报的 HTML 格式表格，。请你：
1. 忽略表格的 HTML 结构，将其转化为语义清晰的自然语言；
2. 对于“本期”、“上期”、“本期末”、“上期末”等时间相对的列，结合给定来源文件名，推断具体期间并补全到列名文本中，使用具体的时间标签（如“2024年第一季度末”、“2023年第四季度末”）进行替换；
3. 行格式：用“中文逗号”分隔的“列名: 值”对，行尾必须加“。”；禁止输出 HTML/Markdown。
4. 多级表头：合并为“父-子”；无法确定列名时使用“列1/列2/…”，一般不得回退为“列N”。
5. 合并单元格/行列标题：将其语义展开至对应数据行；若为标题/分组/单位/口径等说明行（仅标题文本或仅一格非空），独立成行保留。
6. 输出格式为纯文本，不使用表格符号、HTML标签或Markdown格式。

---

【输入示例】
<table><tr><th>项目</th<th>本期</th><th>上期</th></tr><tr><td>营业收入（万元）</td><td>120000</td><td>100000</td></tr><tr><td>净利润（万元）</td><td>25000</td><td>20000</td></tr></table>

---

【输出示例】
项目：营业收入，2024年第一季度：12万元，2023年第四季度：10万元。/n
项目：净利润，2024年第一季度：2.5万元，2023年第四季度：2万元。

'''


def _call_qwen_plus(html: str, file_hint: str) -> Optional[str]:
    api_key = QWEN_API_KEY
    if not api_key:
        return None
    url = f"{QWEN_API_BASE}/chat/completions"
    user_payload = (
        f"来源文件名: {file_hint}\n\n"
        f"HTML表格:\n{html or ''}"
    )
    payload = {
        "model": "qwen-plus",
        "messages": [
            {"role": "system", "content": _LLM_PROMPT},
            {"role": "user", "content": user_payload},
        ],
        "temperature": 0.2,
    }
    data = json.dumps(payload).encode("utf-8")
    req = _ureq.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with _ureq.urlopen(req, timeout=60) as resp:
            body = resp.read()
            obj = json.loads(body.decode("utf-8", errors="ignore"))
            choices = obj.get("choices") or []
            if choices and choices[0].get("message", {}).get("content"):
                return choices[0]["message"]["content"].strip()
    except (_HTTPError, _URLError, TimeoutError):
        return None
    return None


# ---------------- 统一日志 ----------------
_LOG_COLLECTOR: Optional[List[str]] = None


def _log(line: str) -> None:
    print(line, flush=True)
    if _LOG_COLLECTOR is not None:
        _LOG_COLLECTOR.append(line)


def _write_md_log(md_path: str) -> None:
    global _LOG_COLLECTOR
    if not _LOG_COLLECTOR:
        return
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(_LOG_COLLECTOR))
    except OSError:
        pass


def process_json(obj: Any, stats: Dict[str, int], file_hint: str, existing_out: Any = None) -> Any:
    # 递归处理；若 existing_out 中已存在 table_text，则复用
    if isinstance(obj, dict):
        new_obj: Dict[str, Any] = {}
        is_table = obj.get("type") == "table" and isinstance(obj.get("table_body"), str)
        existing_dict = existing_out if isinstance(existing_out, dict) else None
        # 先递归处理子项
        for k, v in obj.items():
            child_existing = existing_dict.get(k) if existing_dict is not None else None
            new_obj[k] = process_json(v, stats, file_hint, child_existing)
        # 再处理表格字段复用/生成
        if is_table:
            html = obj.get("table_body") or ""
            if html.strip():
                stats["seen_tables"] = stats.get("seen_tables", 0) + 1
                total = stats.get("total_tables", 0)
                page_idx = obj.get("page_idx")
                # 若已有输出且包含非空 table_text，直接复用
                reused_text: Optional[str] = None
                if existing_dict and isinstance(existing_dict.get("table_text"), str):
                    t = (existing_dict.get("table_text") or "").strip()
                    if t:
                        reused_text = t
                if reused_text is not None:
                    if page_idx is not None:
                        _log(f"[表格 {stats['seen_tables']}/{total}] 页面:{page_idx} 已存在，复用")
                    else:
                        _log(f"[表格 {stats['seen_tables']}/{total}] 已存在，复用")
                    new_obj["table_text"] = reused_text
                    stats["tables"] = stats.get("tables", 0) + 1
                    stats["reused"] = stats.get("reused", 0) + 1
                else:
                    if page_idx is not None:
                        _log(f"[表格 {stats['seen_tables']}/{total}] 页面:{page_idx} 方式: 大模型")
                    else:
                        _log(f"[表格 {stats['seen_tables']}/{total}] 方式: 大模型")
                    text = _call_qwen_plus(html, file_hint) or ""
                    if not text:
                        try:
                            text = table_html_to_text(html)
                        except Exception as e:
                            text = f"(表格文本化失败: {e})"
                    new_obj["table_text"] = text
                    stats["tables"] = stats.get("tables", 0) + 1
                    stats["generated"] = stats.get("generated", 0) + 1
        return new_obj
    if isinstance(obj, list):
        existing_list = existing_out if isinstance(existing_out, list) else None
        result_list: List[Any] = []
        for i, x in enumerate(obj):
            child_existing = existing_list[i] if (existing_list is not None and i < len(existing_list)) else None
            result_list.append(process_json(x, stats, file_hint, child_existing))
        return result_list
    return obj


def derive_output_path(input_path: str) -> str:
    base, _ext = os.path.splitext(input_path)
    return f"{base}.textified.json"


def _iter_content_list_files(root_dir: str) -> Iterable[str]:
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.endswith("_content_list.json"):
                yield os.path.join(dirpath, fn)


def _orig_output_name(input_path: str) -> str:
    return os.path.basename(input_path)


def main(argv: List[str]) -> int:
    global _LOG_COLLECTOR
    DEFAULT_OUTPUT_DIR = os.path.abspath("/home/wangyaqi/jst/金盘财报_table2text")
    if len(argv) < 2:
        _log("用法: python textify_tables_llm.py <输入文件或目录> [输出目录]")
        _log("  - 输入为文件: 生成 <输入>.textified.json 于同目录，同时写入 <输入>.textified.md")
        _log("  - 输入为目录: 递归处理 *_content_list.json，输出到 [输出目录] 或 <输入>_table2text，同时写入 textify_report.md")
        return 2

    input_path = os.path.abspath(argv[1])
    if not os.path.exists(input_path):
        _log(f"找不到输入路径: {input_path}")
        return 2

    # 单文件模式
    if os.path.isfile(input_path):
        _LOG_COLLECTOR = []
        # 解析输出目录（默认固定为 金盘财报_table2text）
        output_dir = os.path.abspath(argv[2]) if len(argv) >= 3 else DEFAULT_OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        # 既有输出（用于断点续跑）
        out_name_existing = _orig_output_name(input_path)
        out_path_existing = os.path.join(output_dir, out_name_existing)
        existing_out_obj: Any = None
        if os.path.exists(out_path_existing):
            try:
                with open(out_path_existing, "r", encoding="utf-8") as f:
                    existing_out_obj = json.load(f)
            except Exception:
                existing_out_obj = None
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            _log(f"JSON 解析失败: {e}")
            return 2

        def _count_tables(x: Any) -> int:
            if isinstance(x, dict):
                c = 1 if x.get("type") == "table" and isinstance(x.get("table_body"), str) else 0
                for vv in x.values():
                    c += _count_tables(vv)
                return c
            if isinstance(x, list):
                return sum(_count_tables(item) for item in x)
            return 0

        total_tables = _count_tables(data)
        _log(f"检测到表格总数: {total_tables}")

        stats: Dict[str, int] = {"total_tables": total_tables, "seen_tables": 0}
        file_hint = os.path.basename(input_path)
        out_data = process_json(data, stats, file_hint, existing_out_obj)

        # 使用原始文件名保存到固定目录
        out_name = _orig_output_name(input_path)
        output_path = os.path.join(output_dir, out_name)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False)

        _log(f"已生成: {output_path}")
        _log(f"处理表格数量: {stats.get('tables', 0)} | 复用: {stats.get('reused', 0)} | 新增: {stats.get('generated', 0)}")

        # 写入 Markdown 日志到固定目录
        base_name, _ = os.path.splitext(out_name)
        md_path = os.path.join(output_dir, f"{base_name}.textified.md")
        _write_md_log(md_path)
        _log(f"已写入日志: {md_path}")
        return 0

    # 目录模式
    root_dir = input_path
    output_dir = os.path.abspath(argv[2]) if len(argv) >= 3 else DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    files = list(_iter_content_list_files(root_dir))
    if not files:
        _log("未在目录中找到任何 *_content_list.json 文件。")
        return 0

    _LOG_COLLECTOR = []
    _log(f"将在目录中处理 {len(files)} 个 *_content_list.json，输出目录: {output_dir}")

    def _count_tables(x: Any) -> int:
        if isinstance(x, dict):
            c = 1 if x.get("type") == "table" and isinstance(x.get("table_body"), str) else 0
            for vv in x.values():
                c += _count_tables(vv)
            return c
        if isinstance(x, list):
            return sum(_count_tables(item) for item in x)
        return 0

    total_files = len(files)
    for idx, fp in enumerate(files, start=1):
        rel = os.path.relpath(fp, root_dir)
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            _log(f"[{idx}/{total_files}] 跳过 {rel} ：JSON 解析失败: {e}")
            continue
        except OSError as e:
            _log(f"[{idx}/{total_files}] 跳过 {rel} ：读取失败: {e}")
            continue

        total_tables = _count_tables(data)
        _log(f"[{idx}/{total_files}] 开始处理: {rel} | 检测到表格总数: {total_tables}")

        # 若已有输出，尝试判断是否已全部完成；若仅部分完成则断点续跑
        out_name = _orig_output_name(fp)
        out_path = os.path.join(output_dir, out_name)
        existing_out_obj: Any = None
        input_tables = total_tables
        existing_text_count = -1
        if os.path.exists(out_path):
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    existing_out_obj = json.load(f)
            except Exception:
                existing_out_obj = None
            # 统计已存在输出中的 table_text 数量
            def _count_table_texts(x: Any) -> int:
                if isinstance(x, dict):
                    c = 1 if isinstance(x.get("table_text"), str) and (x.get("table_text") or "").strip() else 0
                    for vv in x.values():
                        c += _count_table_texts(vv)
                    return c
                if isinstance(x, list):
                    return sum(_count_table_texts(item) for item in x)
                return 0
            if existing_out_obj is not None:
                existing_text_count = _count_table_texts(existing_out_obj)
        if existing_out_obj is not None and input_tables > 0 and existing_text_count >= input_tables:
            _log(f"[{idx}/{total_files}] 跳过 {rel} ：已完成 ({existing_text_count}/{input_tables})")
            continue
        if existing_out_obj is not None and existing_text_count >= 0:
            _log(f"[{idx}/{total_files}] 断点续跑 {rel} ：已有 ({existing_text_count}/{input_tables})，继续补齐...")

        stats: Dict[str, int] = {"total_tables": total_tables, "seen_tables": 0}
        file_hint = os.path.basename(fp)
        out_data = process_json(data, stats, file_hint, existing_out_obj)

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out_data, f, ensure_ascii=False)
            _log(f"[{idx}/{total_files}] 已生成: {out_path} | 处理表格数量: {stats.get('tables', 0)} | 复用: {stats.get('reused', 0)} | 新增: {stats.get('generated', 0)}")
        except OSError as e:
            _log(f"[{idx}/{total_files}] 写入失败 {out_path}: {e}")

    md_path = os.path.join(output_dir, "textify_report.md")
    _write_md_log(md_path)
    _log(f"已写入日志: {md_path}")
    _log("全部完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


