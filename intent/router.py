from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple
from .intent_schema import QueryIntent, SubQuestion, RoutingDecision, Trace, TokenStats
from .intent_rules import (
    detect_complexity_and_split,
    classify_doc_type,
    extract_years_and_quarters,
)
from .intent_llm import llm_rewrite_and_split, llm_type_disambiguation, LLMCaller


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
    llm_caller: Optional[LLMCaller] = None,
) -> Tuple[QueryIntent, Trace]:
    trace = Trace()
    # rules: complexity/split
    t0 = trace.begin_stage("rules.detect_complexity")
    rules_ext = detect_complexity_and_split(text)
    trace.end_stage(
        "rules.detect_complexity",
        t0,
        {
            "is_complex_rules": bool(rules_ext.get("is_complex")),
            "rules": rules_ext,
        },
    )

    # rules: doc_type
    t0 = trace.begin_stage("rules.classify_doc_type")
    r_type, r_conf = classify_doc_type(text)
    trace.end_stage("rules.classify_doc_type", t0, {"doc_type": r_type, "conf": r_conf})

    # init from rules (years and quarters already extracted in detect_complexity_and_split)
    is_complex = bool(rules_ext.get("is_complex"))
    rewrite = str(rules_ext.get("rewrite") or text)
    subqs_rules = rules_ext.get("sub_questions") or []
    years = rules_ext.get("years") or []
    quarters = rules_ext.get("quarters") or []
    doc_type = r_type if r_type != "unknown" else None
    confidence = float(r_conf or 0.0)

    # LLM enhancement - temporarily disabled
    # if use_llm:
    #     t0 = trace.begin_stage("llm.rewrite_and_split")
    #     out = llm_rewrite_and_split(text, caller=llm_caller)
    #     rewrite = out.get("rewrite", rewrite) or rewrite
    #     llm_subqs = out.get("subqs", []) or []
    #     trace.end_stage(
    #         "llm.rewrite_and_split",
    #         t0,
    #         {"rewrite": rewrite, "subqs": llm_subqs},
    #         tokens=TokenStats.from_dict(out.get("tokens") or {}),
    #     )
    #     # merge sub-questions: prefer LLM if produced
    #     if llm_subqs:
    #         subqs_rules = llm_subqs

    #     # LLM type disambiguation if rules unclear
    #     need_type_llm = (doc_type is None) or (confidence < 0.6)
    #     if need_type_llm:
    #         t0 = trace.begin_stage("llm.type_disambiguation")
    #         tout = llm_type_disambiguation(text, caller=llm_caller)
    #         t_type = tout.get("type", "unknown")
    #         t_prob = float(tout.get("prob", 0.0))
    #         trace.end_stage(
    #             "llm.type_disambiguation",
    #             t0,
    #             {"type": t_type, "prob": t_prob},
    #             tokens=TokenStats.from_dict(tout.get("tokens") or {}),
    #         )
    #         if t_type in {"report", "notice"} and t_prob >= confidence:
    #             doc_type = t_type
    #             confidence = t_prob

    # finalize sub_questions objects
    sub_questions: List[SubQuestion] = []
    if subqs_rules:
        for sq in subqs_rules:
            if not isinstance(sq, dict):
                continue
            ys = sq.get("years") or []
            sub_questions.append(
                SubQuestion(
                    text=str(sq.get("text", "")),
                    doc_type=sq.get("doc_type") if sq.get("doc_type") in {"report", "notice"} else None,
                    years=[int(y) for y in ys if isinstance(y, (int, str))],
                )
            )

    final_is_complex = is_complex or bool(sub_questions)
    intent = QueryIntent(
        is_complex=final_is_complex,
        rewrite=rewrite,
        sub_questions=sub_questions,
        doc_type=doc_type,  # type: ignore
        years=[int(y) for y in years if isinstance(y, (int, str))],
        confidence=confidence,
        routing=None,
    )
    # record finalized intent complexity to avoid confusion with rules stage output
    t1 = trace.begin_stage("intent.finalize")
    trace.end_stage(
        "intent.finalize",
        t1,
        {
            "is_complex": final_is_complex,
            "sources": {
                "rules_is_complex": bool(rules_ext.get("is_complex")),
                "rules_sub_questions": len(sub_questions),
            },
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
    if doc_type and doc_type in by_ty:
        # exact years
        for y in years or []:
            ykey = str(y)
            for p in by_ty[doc_type].get(ykey, []) or []:
                if p not in paths:
                    paths.append(p)
        # fallback to unknown year if no hit
        if not paths and "unknown" in by_ty[doc_type]:
            for p in by_ty[doc_type]["unknown"]:
                if p not in paths:
                    paths.append(p)
    else:
        # unknown doc_type: include both types for years
        for ty in by_ty.keys():
            for y in years or []:
                ykey = str(y)
                for p in by_ty[ty].get(ykey, []) or []:
                    if p not in paths:
                        paths.append(p)
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
    llm_caller: Optional[LLMCaller] = None,
) -> Dict[str, Any]:
    """
    End-to-end: analyze -> route -> (search TBD, to be integrated in search.py) -> return intent + placeholder answers.
    """
    intent, trace = analyze_intent(text, use_llm=use_llm, llm_caller=llm_caller)
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


