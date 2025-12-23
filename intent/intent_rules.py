from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional, Tuple
from datetime import datetime


# Basic keyword dictionaries for fast path detection
REPORT_KEYWORDS = [
    "年报", "年度报告", "半年度", "半年报", "一季报", "第一季度", "二季报", "第二季度",
    "三季报", "第三季度","前三季度" "四季度", "财务", "净利润", "营收", "营业收入", "经营现金流",
    "现金流量", "资产负债", "资产负债表", "利润表", "毛利率", "ROE", "ROA", "研发费用",
    "费用率", "报表", "科目", "分部", "审计报告",
]
NOTICE_KEYWORDS = [
    "公告", "关于", "签订合同", "中标", "回购", "股份回购", "质押", "减持", "增持",
    "停牌", "复牌", "对外担保", "担保", "项目", "定增", "配股", "董事会", "股东大会",
    "回复问询", "更正", "变更", "提示性公告", "提示公告", "进展", "终止", "中止",
]

# Regex patterns
YEAR_PATTERN = re.compile(r"(19|20)\d{2}")
YEAR_CN_SUFFIX_PATTERN = re.compile(r"((19|20)\d{2})\s*年(度)?")
QTR_PATTERN = re.compile(
    r"(Q[1-4]|[一二三四]季(?:度|报)?|第[一二三四]季度|[1-4]季报)",
    re.IGNORECASE,
)
RANGE_PATTERN = re.compile(
    r"((19|20)\d{2})\s*[-~至到—－]\s*((19|20)\d{2})"
)
RECENT_ARABIC_PATTERN = re.compile(r"近\s*([1-9]|1[0-9])\s*年")
RELATIVE_PATTERN = re.compile(r"(去年|前年|今年|去年同期)")

CN_NUM_MAP = {
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}
RECENT_CN_PATTERN = re.compile(r"近([ 一二两三四五六七八九十]+)年")


def _parse_recent_cn(text: str) -> Optional[int]:
    m = RECENT_CN_PATTERN.search(text)
    if not m:
        return None
    s = m.group(1).strip().replace(" ", "")
    # Simple Chinese numerals parsing up to 19
    if s == "十":
        return 10
    if "十" in s:
        parts = s.split("十")
        if parts[0] == "":  # 十三
            base = 10
        else:
            base = CN_NUM_MAP.get(parts[0], 0) * 10
        tail = CN_NUM_MAP.get(parts[1], 0) if len(parts) > 1 and parts[1] != "" else 0
        n = base + tail
        return n if 1 <= n <= 19 else None
    return CN_NUM_MAP.get(s)


def classify_doc_type(text: str) -> Tuple[Literal["report", "notice", "unknown"], float]:
    """
    Returns doc_type and a heuristic confidence score in [0,1].
    """
    t = text or ""
    report_hits = sum(1 for k in REPORT_KEYWORDS if k in t)
    notice_hits = sum(1 for k in NOTICE_KEYWORDS if k in t)
    if report_hits == 0 and notice_hits == 0:
        return "unknown", 0.0
    if report_hits > notice_hits:
        conf = min(1.0, 0.5 + 0.1 * (report_hits - notice_hits))
        return "report", conf
    if notice_hits > report_hits:
        conf = min(1.0, 0.5 + 0.1 * (notice_hits - report_hits))
        return "notice", conf
    # tie
    return "unknown", 0.5


def _expand_range(y1: int, y2: int, cap: int = 10) -> List[int]:
    if y1 > y2:
        y1, y2 = y2, y1
    rng = list(range(y1, y2 + 1))
    if len(rng) > cap:
        return rng[-cap:]
    return rng


def extract_years_and_quarters(text: str, current_year: Optional[int] = None) -> Dict[str, Any]:
    """
    Extract absolute years, quarters, and interpret relative phrases like '近三年', '去年'.
    Returns:
      {
        "years": List[int],
        "quarters": List[str],  # e.g., ["Q1","Q2"]
        "is_range": bool,
        "n_recent": Optional[int],
        "has_relative": bool
      }
    """
    if current_year is None:
        current_year = datetime.now().year
    years: List[int] = []
    quarters: List[str] = []
    is_range = False
    n_recent: Optional[int] = None
    has_relative = False

    # 1) Absolute years
    for m in YEAR_PATTERN.finditer(text):
        y = int(m.group(0))
        if 1900 <= y <= 2100:
            years.append(y)

    # 2) Ranges like 2019-2021
    for m in RANGE_PATTERN.finditer(text):
        y1 = int(m.group(1))
        y2 = int(m.group(3))
        expanded = _expand_range(y1, y2, cap=10)
        years.extend(expanded)
        is_range = True

    # 3) Recent N years (arabic)
    m = RECENT_ARABIC_PATTERN.search(text)
    if m:
        n_recent = int(m.group(1))
        years.extend([current_year - i for i in range(n_recent)])

    # 4) Recent N years (Chinese)
    cn = _parse_recent_cn(text)
    if cn:
        n_recent = cn
        years.extend([current_year - i for i in range(cn)])

    # 5) Relative words
    if RELATIVE_PATTERN.search(text):
        has_relative = True
        if "去年" in text:
            years.append(current_year - 1)
        if "前年" in text:
            years.append(current_year - 2)
        if "今年" in text:
            years.append(current_year)
        # 去年同期：如果识别到季度，则保持同季度；此处仅标记年份
        if "去年同期" in text and (current_year - 1) not in years:
            years.append(current_year - 1)

    # 6) Quarters
    for m in QTR_PATTERN.finditer(text):
        q = m.group(0).upper()
        if q.startswith("Q") and q in {"Q1", "Q2", "Q3", "Q4"}:
            quarters.append(q)
        else:
            # map Chinese to Qx
            if "一" in q or "1" in q:
                quarters.append("Q1")
            elif "二" in q or "2" in q:
                quarters.append("Q2")
            elif "三" in q or "3" in q:
                quarters.append("Q3")
            elif "四" in q or "4" in q:
                quarters.append("Q4")

    # Normalize uniques and sort
    years = sorted(set(y for y in years if 1900 <= y <= 2100))
    quarters = list(dict.fromkeys(quarters))  # keep order, unique

    return {
        "years": years,
        "quarters": quarters,
        "is_range": is_range,
        "n_recent": n_recent,
        "has_relative": has_relative,
    }
