from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple
from .intent_schema import QueryIntent, RoutingDecision, Trace, TokenStats
from .intent_rules import (
    classify_doc_type,
    extract_years_and_quarters,
)
from .intent_llm import llm_extract_metadata


def _project_root() -> str:
    # intent/ -> project root assumed one level up
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _registry_dir() -> str:
    return os.path.join(_project_root(), "registry")


def _load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_index_registry() -> Dict[str, Any]:
    return _load_json(os.path.join(_registry_dir(), "index_registry.json"))


def load_doc_registry() -> Dict[str, Any]:
    return _load_json(os.path.join(_registry_dir(), "doc_registry.json"))


def analyze_intent(
    text: str,
    use_llm: bool = True,
) -> Tuple[QueryIntent, Trace]:
    trace = Trace()
    
    years: List[int] = []
    doc_type: Optional[str] = None
    confidence: float = 0.0

    llm_success = False

    # 1. Try LLM extraction first if enabled
    if use_llm:
        t0 = trace.begin_stage("llm.extract_metadata")
        # No caller needed anymore
        out = llm_extract_metadata(text)
        llm_error = out.get("error", False)
        
        trace.end_stage(
            "llm.extract_metadata",
            t0,
            {"years": out.get("years"), "doc_type": out.get("doc_type"), "error": llm_error},
            tokens=TokenStats.from_dict(out.get("tokens") or {}),
        )

        if not llm_error:
            # LLM success: use its results
            llm_years = out.get("years", [])
            llm_type = out.get("doc_type", "unknown")
            
            years = llm_years
            if llm_type in {"report", "notice"}:
                doc_type = llm_type
                confidence = 0.9 # High confidence for successful LLM parse
            else:
                doc_type = None # Unknown
            
            llm_success = True

    # 2. If LLM failed (or disabled), fallback to Rules
    if not llm_success:
        # Rules: Year extraction
        t0 = trace.begin_stage("rules.extract_years")
        years_ext = extract_years_and_quarters(text)
        years = years_ext.get("years", [])
        trace.end_stage("rules.extract_years", t0, {"years": years})

        # Rules: Doc type
        t0 = trace.begin_stage("rules.classify_doc_type")
        r_type, r_conf = classify_doc_type(text)
        trace.end_stage("rules.classify_doc_type", t0, {"doc_type": r_type, "conf": r_conf})

        doc_type = r_type if r_type != "unknown" else None
        confidence = float(r_conf or 0.0)

    intent = QueryIntent(
        doc_type=doc_type,  # type: ignore
        years=[int(y) for y in years if isinstance(y, (int, str))],
        confidence=confidence,
        routing=None,
    )
    
    t1 = trace.begin_stage("intent.finalize")
    trace.end_stage(
        "intent.finalize",
        t1,
        {
            "doc_type": doc_type,
            "years": years,
            "source": "llm" if llm_success else "rules"
        },
    )
    return intent, trace


def _resolve_indices_by_years(
    doc_type: Optional[str],
    years: List[int],
) -> List[str]:
    reg = load_index_registry()
    paths: List[str] = []
    if not reg:
        return paths
    by_ty = (reg.get("faiss") or {}).get("by_type_year") or {}
    # 辅助：添加路径并去重
    def _add(p_list):
        for p in p_list or []:
            if p not in paths:
                paths.append(p)

    if doc_type and doc_type in by_ty:
        if years:
            # 指定了年份：精确匹配
            for y in years:
                ykey = str(y)
                _add(by_ty[doc_type].get(ykey))
            # 兜底：如果指定年份没找到，尝试 unknown 桶
            if not paths:
                _add(by_ty[doc_type].get("unknown"))
        else:
            # 【优化】未指定年份：包含该类型下所有年份的索引
            for ykey in by_ty[doc_type]:
                _add(by_ty[doc_type][ykey])
    else:
        # 类型未知：现有逻辑保持不变（或也可改为全选）
        for ty in by_ty.keys():
            for y in years or []:
                ykey = str(y)
                _add(by_ty[ty].get(ykey))
                
    return paths


def route_indices(
    intent: QueryIntent,
    prefer_year: bool = True,
) -> RoutingDecision:
    years = intent.years
    dtype = intent.doc_type
    indices: List[str] = []
    strategy: str = "global"
    fallback: str = "global"

    if prefer_year and years:
        indices = _resolve_indices_by_years(dtype, years)
        if indices:
            strategy = "year-buckets"
        else:
            strategy = "global"

    # If still empty, global fallback (will be used by search layer)
    return RoutingDecision(
        strategy=strategy,  # type: ignore
        year_buckets=years or [],
        fallback=fallback,  # type: ignore
        indices=indices,
    )


def route_and_search(
    text: str,
    k: int = 8,
    prefer_year: bool = True,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """
    End-to-end: analyze -> route -> (search TBD, to be integrated in search.py) -> return intent + placeholder answers.
    """
    # Removed llm_caller
    intent, trace = analyze_intent(text, use_llm=use_llm)
    t0 = trace.begin_stage("router.select_indices")
    routing = route_indices(intent, prefer_year=prefer_year)
    intent.routing = routing
    trace.end_stage("router.select_indices", t0, {"strategy": routing.strategy, "indices": routing.indices, "years": routing.year_buckets})
    # Placeholder answers: integration will be done in search.py
    return {
        "intent": intent.to_dict(),
        "answers": [],
        "trace": trace.to_dict(),
    }
