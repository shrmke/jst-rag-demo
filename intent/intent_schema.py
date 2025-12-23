from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional
import uuid
import time


def _prune_none(value: Any) -> Any:
    """
    Recursively drop keys where value is None, and filter None items in lists.
    """
    if isinstance(value, dict):
        return {k: _prune_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_prune_none(v) for v in value if v is not None]
    return value


@dataclass
class TokenStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return _prune_none(asdict(self))

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TokenStats":
        return TokenStats(
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            total_tokens=int(data.get("total_tokens", 0)),
        )


@dataclass
class StageEvent:
    name: str
    start_ms: Optional[int] = None
    duration_ms: Optional[int] = None
    output: Optional[Dict[str, Any]] = None
    tokens: Optional[TokenStats] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.tokens is not None:
            data["tokens"] = self.tokens.to_dict()
        return _prune_none(data)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "StageEvent":
        tokens = data.get("tokens")
        return StageEvent(
            name=str(data.get("name")),
            start_ms=data.get("start_ms"),
            duration_ms=data.get("duration_ms"),
            output=data.get("output"),
            tokens=TokenStats.from_dict(tokens) if isinstance(tokens, dict) else None,
        )


@dataclass
class Trace:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    stages: List[StageEvent] = field(default_factory=list)
    progress: Dict[str, Any] = field(default_factory=dict)
    totals: Dict[str, Any] = field(default_factory=dict)

    def add_stage(
        self,
        name: str,
        output: Optional[Dict[str, Any]] = None,
        start_ms: Optional[int] = None,
        duration_ms: Optional[int] = None,
        tokens: Optional[TokenStats] = None,
    ) -> None:
        self.stages.append(
            StageEvent(
                name=name,
                start_ms=start_ms,
                duration_ms=duration_ms,
                output=output,
                tokens=tokens,
            )
        )

    def begin_stage(self, name: str) -> int:
        # returns monotonic ms
        return int(time.time() * 1000)

    def end_stage(
        self,
        name: str,
        start_ms: int,
        output: Optional[Dict[str, Any]] = None,
        tokens: Optional[TokenStats] = None,
    ) -> None:
        now_ms = int(time.time() * 1000)
        self.add_stage(
            name=name,
            output=output,
            start_ms=start_ms,
            duration_ms=max(0, now_ms - start_ms),
            tokens=tokens,
        )

    def to_dict(self) -> Dict[str, Any]:
        return _prune_none(
            {
                "trace_id": self.trace_id,
                "stages": [s.to_dict() for s in self.stages],
                "progress": self.progress,
                "totals": self.totals,
            }
        )

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Trace":
        t = Trace(trace_id=str(data.get("trace_id", str(uuid.uuid4()))))
        t.stages = [
            StageEvent.from_dict(s) for s in data.get("stages", []) if isinstance(s, dict)
        ]
        t.progress = data.get("progress", {}) or {}
        t.totals = data.get("totals", {}) or {}
        return t


@dataclass
class RoutingDecision:
    strategy: Literal["year-buckets", "global", "hybrid"] = "year-buckets"
    year_buckets: List[int] = field(default_factory=list)
    fallback: Literal["global", "none"] = "global"
    indices: List[str] = field(default_factory=list)  # resolved index paths (optional)

    def to_dict(self) -> Dict[str, Any]:
        return _prune_none(asdict(self))

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "RoutingDecision":
        return RoutingDecision(
            strategy=data.get("strategy", "year-buckets"),
            year_buckets=[int(y) for y in data.get("year_buckets", [])],
            fallback=data.get("fallback", "global"),
            indices=[str(p) for p in data.get("indices", [])],
        )


@dataclass
class QueryIntent:
    # Removed: is_complex, rewrite, sub_questions
    doc_type: Optional[Literal["report", "notice"]] = None
    years: List[int] = field(default_factory=list)
    confidence: float = 0.0
    routing: Optional[RoutingDecision] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.routing is not None:
            data["routing"] = self.routing.to_dict()
        return _prune_none(data)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "QueryIntent":
        return QueryIntent(
            doc_type=data.get("doc_type"),
            years=[int(y) for y in data.get("years", []) if isinstance(y, (int, str))],
            confidence=float(data.get("confidence", 0.0)),
            routing=RoutingDecision.from_dict(data["routing"]) if isinstance(data.get("routing"), dict) else None,
        )


@dataclass
class AnswerChunk:
    text: str
    source: Optional[str] = None
    score: Optional[float] = None
    year: Optional[int] = None
    doc_type: Optional[Literal["report", "notice"]] = None

    def to_dict(self) -> Dict[str, Any]:
        return _prune_none(asdict(self))
