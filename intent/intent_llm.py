from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from .intent_schema import TokenStats


LLMCaller = Callable[[List[Dict[str, str]], Dict[str, Any]], Dict[str, Any]]


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
    caller: Optional[LLMCaller],
    messages: List[Dict[str, str]],
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[str, TokenStats, int]:
    """
    Returns (content, token_stats, duration_ms)
    """
    if extra is None:
        extra = {}
    start = _now_ms()
    if caller is None:
        # No-op fallback
        content = "{\"rewrite\": \"%s\", \"sub_questions\": [], \"doc_type\": \"unknown\"}" % (
            messages[-1]["content"].replace('"', '\\"')
        )
        ts = TokenStats(prompt_tokens=_estimate_tokens(messages[-1]["content"]), completion_tokens=10, total_tokens=10 + _estimate_tokens(messages[-1]["content"]))
        return content, ts, max(1, _now_ms() - start)
    resp = caller(messages, extra)
    duration_ms = max(1, _now_ms() - start)
    content = ""
    # Common shapes
    if isinstance(resp, dict):
        if "choices" in resp:
            choices = resp["choices"] or []
            if choices:
                msg = choices[0].get("message") or {}
                content = msg.get("content", "") or ""
        elif "output" in resp:
            # Some SDKs use "output" to store text
            content = resp.get("output", "") or ""
    if not content and isinstance(resp, dict):
        content = resp.get("content", "") or ""
    token_stats = _extract_usage(resp if isinstance(resp, dict) else {}, json.dumps(messages, ensure_ascii=False), content)
    return content, token_stats, duration_ms


def llm_rewrite_and_split(
    text: str,
    caller: Optional[LLMCaller] = None,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    Ask LLM (Qwen) to rewrite and split query into structured JSON.
    Returns:
      {
        "rewrite": str,
        "subqs": [{"text": str, "years": [int], "doc_type": "report|notice|unknown"}],
        "tokens": { ... },
        "ms": int
      }
    """
    sys_prompt = (
        "今年是2025年，你是一个中文信息抽取助手。请将用户问题改写为更明确、不可歧义的表达，并在需要时拆分为若干子问题。\n"
        "输出必须是JSON，字段：\n"
        "rewrite: 字符串；\n"
        "subqs: 子问题数组，每项包含 text（字符串）、可选 years（数组，整数）、可选 doc_type（report|notice|unknown）。\n"
    )
    user_prompt = f"用户问题：{text}\n请仅输出JSON，不要其他文字。"
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    content, token_stats, ms = _call_llm(
        caller,
        messages,
        extra={"temperature": temperature, "response_format": "json"},
    )
    rewrite = text
    subqs: List[Dict[str, Any]] = []
    try:
        data = json.loads(content)
        rewrite = data.get("rewrite", text) or text
        for sq in data.get("subqs", []) or []:
            if not isinstance(sq, dict):
                continue
            item = {"text": str(sq.get("text", "")).strip()}
            if "years" in sq and isinstance(sq["years"], list):
                years = []
                for y in sq["years"]:
                    try:
                        years.append(int(y))
                    except Exception:
                        pass
                if years:
                    item["years"] = years
            if "doc_type" in sq:
                item["doc_type"] = sq["doc_type"]
            subqs.append(item)
    except Exception:
        # Fallback: keep original as single subq
        subqs = [{"text": text}]
    return {
        "rewrite": rewrite,
        "subqs": subqs,
        "tokens": token_stats.to_dict(),
        "ms": ms,
    }


def llm_type_disambiguation(
    text: str,
    caller: Optional[LLMCaller] = None,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """
    Ask LLM to decide the more likely doc_type with confidence.
    Returns:
      {
        "type": "report|notice|unknown",
        "prob": float,
        "tokens": { ... },
        "ms": int
      }
    """
    sys_prompt = (
        "你是一个分类助手。根据用户问题判断其更接近哪一类：财报类（report）或公告类（notice）。\n"
        "请只输出JSON：{type:'report|notice|unknown', prob: 0..1}。"
    )
    user_prompt = f"用户问题：{text}\n请仅输出JSON。"
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    content, token_stats, ms = _call_llm(
        caller,
        messages,
        extra={"temperature": temperature, "response_format": "json"},
    )
    dtype = "unknown"
    prob = 0.0
    try:
        data = json.loads(content)
        t = str(data.get("type", "unknown")).strip().lower()
        if t in {"report", "notice", "unknown"}:
            dtype = t
        p = float(data.get("prob", 0.0))
        if 0.0 <= p <= 1.0:
            prob = p
    except Exception:
        pass
    return {
        "type": dtype,
        "prob": prob,
        "tokens": token_stats.to_dict(),
        "ms": ms,
    }


