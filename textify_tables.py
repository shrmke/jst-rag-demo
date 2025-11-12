#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 JSON 中的 HTML 表格 (table_body) 轻量文本化：
  - 为所有 type=="table" 的对象新增字段 table_text，不覆盖原 table_body。
  - 文本化规则：
      * 识别并展开 colspan/rowspan，生成二维网格
      * 合并多行表头为列名（父-子），空列名回退为“列1/列2/...”
      * 数据行输出为 “列名: 值” ，列间以“，”分隔；每行以句号收尾（若无则补“。”）
      * 单元/说明行（仅一格非空，如“单位：元”或标题行）直接作为一行输出，保留行级换行/句号
  - 单文件输入：输出到同目录同名+.textified.json
  - 目录输入：递归查找所有以“_content_list.json”结尾的文件，输出到“输出目录”中，使用原始文件名保存
  - 日志：将运行期间的所有打印同时保存为 markdown 文件（单文件模式写入同目录 <输入>.textified.md；目录模式写入输出目录 textify_report.md）

纯标准库实现，无第三方依赖。
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
    """将单个 <table> 的 HTML 解析为二维网格，展开 colspan/rowspan。

    仅处理第一个表格，忽略嵌套非表格内容。
    """

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

        # 当前单元
        self._cell_text_parts: List[str] = []
        self._cell_colspan: int = 1
        self._cell_rowspan: int = 1

        self._table_parsed: bool = False

    def _prefill_rowspans_until_gap(self) -> None:
        # 在当前列位填充连续的 rowspan 占位
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
            # 在新单元开始前，先填补可能的 rowspan 空位
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
            # 单元内换行转为空格，避免破坏行级语义
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
            # 行结束后，右侧仍可能存在未填的连续 rowspan 空位
            self._prefill_rowspans_until_gap()
            self.grid.append(self.current_row)
            self.inside_tr = False
            self.current_row = []
            self.current_col = 0

    def handle_data(self, data: str) -> None:
        if self.inside_cell and self.inside_table and data:
            self._cell_text_parts.append(data)

    def get_grid(self) -> List[List[str]]:
        # 统一列宽
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


 


def finalize_line(s: str) -> str:
    s = normalize_space(s)
    if not s:
        return s
    return s if s.endswith(("。", ".")) else s + "。"


def grid_to_text(grid: List[List[str]]) -> str:
    if not grid:
        return ""
    # 寻找首个“非空单元格数≥2”的表头行
    header_idx: Optional[int] = None
    for idx, row in enumerate(grid):
        non_empty_cells = [c for c in row if normalize_space(c)]
        if len(non_empty_cells) >= 2:
            header_idx = idx
            break
    lines: List[str] = []
    if header_idx is None:
        # 未找到表头：按最大列宽生成列名，从第一行开始全部作为数据
        width = max((len(r) for r in grid), default=0)
        headers = [f"列{j+1}" for j in range(width)]
        data_rows = grid
    else:
        headers = [normalize_space(c) or f"列{j+1}" for j, c in enumerate(grid[header_idx])]
        width = len(headers)
        data_rows = grid[header_idx + 1 :]
    for r in data_rows:
        # 生成“列名: 值”
        parts: List[str] = []
        r_expanded = list(r) + [""] * (width - len(r))
        for j in range(width):
            v = normalize_space(r_expanded[j])
            if not v:
                continue
            col_name = normalize_space(headers[j]) or f"列{j+1}"
            parts.append(f"{col_name}: {v}")
        if parts:
            s = "，".join(parts)
            lines.append(s if s.endswith(("。", ".")) else s + "。")
    return "\n".join(lines)


def table_html_to_text(html: str) -> str:
    parser = HTMLTableGridParser()
    parser.feed(html or "")
    grid = parser.get_grid()
    return grid_to_text(grid)

 

# ---------------- 结构复杂性检测（含合并单元格） ----------------
_MERGED_ATTR_RE = re.compile(r"\b(?:colspan|rowspan)\s*=\s*[\"']?\s*(\d+)", re.IGNORECASE)

def _html_has_merged_cells(html: str) -> bool:
    if not html:
        return False
    for m in _MERGED_ATTR_RE.finditer(html):
        try:
            if int(m.group(1)) > 1:
                return True
        except ValueError:
            pass
    return False

# ---------------- Qwen-Plus 配置（可直接在此写入你的密钥与 BASE）----------------
# 优先使用环境变量，其次使用下方硬编码。请将 "" 改为你的真实密钥字符串。
QWEN_API_KEY = os.environ.get("QWEN_API_KEY") or "sk-6a44d15e56dd4007945ccc41b97b499c"
QWEN_API_BASE = os.environ.get("QWEN_API_BASE") or "https://dashscope.aliyuncs.com/compatible-mode/v1"

def _call_qwen_plus(prompt: str, html: str, file_hint: str) -> Optional[str]:
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
            {"role": "system", "content": prompt},
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
            # OpenAI 兼容返回结构
            choices = obj.get("choices") or []
            if choices and choices[0].get("message", {}).get("content"):
                return choices[0]["message"]["content"].strip()
    except (_HTTPError, _URLError, TimeoutError):
        return None
    return None

_LLM_PROMPT = '''
你是一名财务数据整理专家，任务是将财务/公告表格转换为可用于语义检索的自然语言描述。

输入提供【来源文件名】与一段 HTML 表格。请你：
1) 忽略 HTML 结构，将其转化为语义清晰的纯文本；每行一条记录。
2) 对“本期/上期/本期末/上期末/本报告期/本报告期末/上年同期/期初/期末”等相对时间列名，结合【来源文件名】推断具体期间并补全到列名文本中（如“2021年第一季度末”）；若无法准确解析则保持原列名。
3) 行格式：用“中文逗号”分隔“列名: 值”，行尾必须加“。”；禁止输出 HTML/Markdown。
4) 多级表头：合并为“父-子”；无法确定列名时使用“列1/列2/…”，一般不得回退为“列N”。
5) 合并单元格/行列标题：展开其语义至对应数据行；若为标题/分组/单位/口径等说明行（仅标题文本或仅一格非空），独立成行保留。
6) 保留原数值/比例/金额/日期/区间格式，不增删或翻译单位/专有名词；单元内换行合并为空格。
7) 汇总/合计行（合计/小计/总计等）用对应标签作为行前缀，不映射为列名，其他单元格按列名输出“列名: 值”。
8) 仅输出纯文本结果，不添加多余解释。
'''
# _LLM_PROMPT = (
#     "你是表格重排助手。给你一个HTML表格（可能含colspan/rowspan、多级表头、说明行）。"
#     "请输出轻量文本，每行代表一条记录，格式：‘列名: 值’用中文逗号分隔列，行尾加‘。’；"
#     "若存在说明/单位/分组行（仅标题文本），独立成行保留；"
#     "表头多级时合并为‘父-子’；无法判定列名时用‘列1/列2…’；勿丢失显著信息。"
# )

# 在原有流程基础上尝试兜底（在 process_json 中调用）

# ---------------- 统一日志收集与打印 ----------------
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


def process_json(obj: Any, stats: Dict[str, int], file_hint: str) -> Any:
    # 递归处理（不做逐表复用；按文件级断点即可）
    if isinstance(obj, dict):
        new_obj: Dict[str, Any] = {}
        is_table = obj.get("type") == "table" and isinstance(obj.get("table_body"), str)
        for k, v in obj.items():
            new_obj[k] = process_json(v, stats, file_hint)
        if is_table:
            html = obj.get("table_body") or ""
            if html.strip():
                stats["seen_tables"] = stats.get("seen_tables", 0) + 1
                total = stats.get("total_tables", 0)
                page_idx = obj.get("page_idx")
                use_llm = _html_has_merged_cells(html)
                mode_str = "大模型" if use_llm else "脚本"
                if page_idx is not None:
                    _log(f"[表格 {stats['seen_tables']}/{total}] 页面:{page_idx} 方式: {mode_str}")
                else:
                    _log(f"[表格 {stats['seen_tables']}/{total}] 方式: {mode_str}")
                if use_llm:
                    llm_text = _call_qwen_plus(_LLM_PROMPT, html, file_hint) or ""
                    if llm_text:
                        text = llm_text
                    else:
                        try:
                            text = table_html_to_text(html)
                        except Exception as e:
                            text = f"(表格文本化失败: {e})"
                else:
                    try:
                        text = table_html_to_text(html)
                    except Exception as e:
                        text = f"(表格文本化失败: {e})"
                new_obj["table_text"] = text
                stats["tables"] = stats.get("tables", 0) + 1
        return new_obj
    if isinstance(obj, list):
        return [process_json(x, stats, file_hint) for x in obj]
    return obj


def derive_output_path(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    return f"{base}.textified.json"


def _iter_content_list_files(root_dir: str) -> Iterable[str]:
    """递归枚举 root_dir 下所有以 _content_list.json 结尾的文件。"""
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.endswith("_content_list.json"):
                yield os.path.join(dirpath, fn)


def _orig_output_name(input_path: str) -> str:
    """返回输入文件的原始文件名（保持不变）。"""
    return os.path.basename(input_path)


def main(argv: List[str]) -> int:
    global _LOG_COLLECTOR
    if len(argv) < 2:
        _log("用法: python textify_tables.py <输入文件或目录> [输出目录]")
        _log("  - 输入为文件: 生成结果至固定输出目录，同时写入同名 .textified.md 日志")
        _log("  - 输入为目录: 递归处理 *_content_list.json，输出到 [输出目录] 或固定目录，同时写入 textify_report.md")
        return 2

    input_path = os.path.abspath(argv[1])
    if not os.path.exists(input_path):
        _log(f"找不到输入路径: {input_path}")
        return 2

    DEFAULT_OUTPUT_DIR = os.path.abspath("/home/wangyaqi/jst/金盘上市公告_table2text")

    # 单文件模式：输出到固定目录，文件名保持不变；仅按文件是否已存在来跳过
    if os.path.isfile(input_path):
        _LOG_COLLECTOR = []
        output_dir = os.path.abspath(argv[2]) if len(argv) >= 3 else DEFAULT_OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        out_name = _orig_output_name(input_path)
        out_path = os.path.join(output_dir, out_name)
        if os.path.exists(out_path):
            _log(f"已存在结果文件，跳过: {out_path}")
            return 0
        with open(input_path, "r", encoding="utf-8") as f:
            try:
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
        out_data = process_json(data, stats, file_hint)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False)

        _log(f"已生成: {out_path}")
        _log(f"处理表格数量: {stats.get('tables', 0)}")

        # 写入 markdown 日志到固定目录
        base_name, _ = os.path.splitext(out_name)
        md_path = os.path.join(output_dir, f"{base_name}.textified.md")
        _write_md_log(md_path)
        _log(f"已写入日志: {md_path}")
        return 0

    # 目录模式：递归处理所有 *_content_list.json
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

        # 若已有输出文件则跳过（文件级断点）
        out_name = _orig_output_name(fp)
        out_path = os.path.join(output_dir, out_name)
        if os.path.exists(out_path):
            _log(f"[{idx}/{total_files}] 跳过 {rel} ：已存在结果文件")
            continue

        stats: Dict[str, int] = {"total_tables": total_tables, "seen_tables": 0}
        file_hint = os.path.basename(fp)
        out_data = process_json(data, stats, file_hint)

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out_data, f, ensure_ascii=False)
            _log(f"[{idx}/{total_files}] 已生成: {out_path} | 处理表格数量: {stats.get('tables', 0)}")
        except OSError as e:
            _log(f"[{idx}/{total_files}] 写入失败 {out_path}: {e}")

    # 写入 markdown 日志
    md_path = os.path.join(output_dir, "textify_report.md")
    _write_md_log(md_path)
    _log(f"已写入日志: {md_path}")

    _log("全部完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))