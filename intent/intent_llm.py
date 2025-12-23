from __future__ import annotations
import sys
import json
import time
import urllib.request as _ureq
from typing import Any, Dict, List, Optional, Tuple
from .intent_schema import TokenStats
from datetime import datetime

# Hardcoded Configuration
QWEN_API_KEY = "sk-6a44d15e56dd4007945ccc41b97b499c"
QWEN_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_CHAT_MODEL = "qwen-plus"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _estimate_tokens(text: str) -> int:
    # Very rough heuristic fallback if usage is missing
    if not text:
        return 0
    return max(1, len(text) // 2)


def _extract_usage(resp: Dict[str, Any], prompt: str, completion: str) -> TokenStats:
    usage = resp.get("usage") or {}
    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens")
    tt = usage.get("total_tokens")
    if pt is None or ct is None or tt is None:
        pt = _estimate_tokens(prompt)
        ct = _estimate_tokens(completion)
        tt = pt + ct
    return TokenStats(prompt_tokens=int(pt), completion_tokens=int(ct), total_tokens=int(tt))


def _call_llm(
    messages: List[Dict[str, str]],
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[str, TokenStats, int]:
    """
    Directly call Qwen API with hardcoded credentials.
    Returns (content, token_stats, duration_ms)
    """
    if extra is None:
        extra = {}
    
    start = _now_ms()
    
    url = f"{QWEN_API_BASE}/chat/completions"
    model = extra.get("model", QWEN_CHAT_MODEL)
    temperature = extra.get("temperature", 0.1)
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 2000
    }
    
    # Optional params
    if "response_format" in extra:
        payload["response_format"] = {"type": "json_object"}

    req = _ureq.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {QWEN_API_KEY}")

    content = ""
    resp_json = {}
    
    try:
        with _ureq.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            resp_json = json.loads(body)
            print(f"-------- [LLM resp_json] --------\n{resp_json}\n----------------------------------", file=sys.stderr, flush=True)
            if "choices" in resp_json:
                choices = resp_json["choices"] or []
                if choices:
                    msg = choices[0].get("message") or {}
                    content = msg.get("content", "") or ""
    except Exception as e:
        # Simple error handling: return empty content on failure
        # print(f"LLM Call Error: {e}")
        print(f"[LLM Call Error]: {e}", file=sys.stderr, flush=True)
        pass

    duration_ms = max(1, _now_ms() - start)
    token_stats = _extract_usage(resp_json, json.dumps(messages, ensure_ascii=False), content)
    
    return content, token_stats, duration_ms


def llm_extract_metadata(
    text: str,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """
    Ask LLM to extract years and document type from the query.
    Returns:
      {
        "years": [int],
        "doc_type": "report|notice|unknown",
        "tokens": { ... },
        "ms": int,
        "error": bool  # True if JSON parsing failed
      }
    """
    current_year = datetime.now().year
    sys_prompt = f"""你是一个专业的RAG查询路由助手。你的任务是分析用户的查询，提取元数据过滤条件。

今年是 {current_year}年

请提取以下维度的过滤条件（如果用户未提及，则不要包含该字段）：
1. years (List[int]): 具体的年份列表。请将"今年"、"去年"、"前年"等相对时间转换为具体的四位数字年份。如果是范围，请列出范围内的所有年份。如果无法识别或未提及年份，请返回空列表 []。
2. category (str): 判断查询的信息来源，例如如果询问某财务数额，那么信息来源为"report",否则请返回"unknown"。

请直接返回 JSON 格式结果，不要包含任何 Markdown 格式或额外解释。
JSON 格式示例:
{{
    "search_filters": {{
        "years": [2023, 2024],
        "category": "report"
    }}
}}
"""
    user_prompt = f"用户问题：{text}\n请仅输出JSON。"
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    # No caller passed, direct call
    content, token_stats, ms = _call_llm(
        messages,
        extra={"temperature": temperature, "response_format": "json"},
    )
    print(f"-------- [LLM Raw Output] --------\n{content}\n----------------------------------", file=sys.stderr, flush=True)
    
    years = []
    doc_type = "unknown"
    error = False
    
    try:
        data = json.loads(content)
        # Handle nested search_filters
        filters = data.get("search_filters", {})
        
        # Extract years
        # Support both "years" (list) and "year" (int/list) for robustness
        raw_years = filters.get("years") or filters.get("year") or []
        if isinstance(raw_years, (int, str)):
             raw_years = [raw_years]
        
        if isinstance(raw_years, list):
            for y in raw_years:
                try:
                    years.append(int(y))
                except (ValueError, TypeError):
                    pass
        
        # Extract category -> doc_type
        # Map "announce" to "notice" to match system convention
        cat = str(filters.get("category", "unknown")).lower().strip()
        if cat == "report":
            doc_type = "report"
        else:
            doc_type = "unknown"
            
    except Exception as e:
        print(f"[Parse Error]: {e}", file=sys.stderr, flush=True)
        error = True

    result = {
        "years": sorted(list(set(years))),
        "doc_type": doc_type,
        "tokens": token_stats.to_dict(),
        "ms": ms,
        "error": error,
    }
    print(f"-------- [Extracted Metadata] --------\n{result}\n--------------------------------------", file=sys.stderr, flush=True)
    return result
