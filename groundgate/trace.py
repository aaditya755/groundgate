"""Structured execution trace and audit logging for groundgate pipeline decisions.

Why this module exists:
Users and judges need complete visibility into every pipeline decision: language detection,
BM25 score, pre-check gate, Tier-1 gate verification, escalation trigger, and final outcome.
This trace guarantees 100% explainability without hidden side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AttemptRecord:
    """Record of a single model call attempt and its deterministic gate results."""

    tier: str  # "tier1" or "tier2"
    provider: str
    model: str
    latency: float
    finish_reason: str
    raw_output: str
    gate_passed: bool
    gate_outcome: str
    gate_checks: dict[str, bool]
    gate_failures: list[str] = field(default_factory=list)
    gate_details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "provider": self.provider,
            "model": self.model,
            "latency": self.latency,
            "finish_reason": self.finish_reason,
            "raw_output": self.raw_output,
            "gate_passed": self.gate_passed,
            "gate_outcome": self.gate_outcome,
            "gate_checks": self.gate_checks,
            "gate_failures": self.gate_failures,
            "gate_details": self.gate_details,
        }


@dataclass
class ExecutionTrace:
    """Comprehensive trace of a query through retrieval, pre-checks, model calls, and gates."""

    question: str
    detected_language: str
    retrieval_passages_count: int = 0
    top_bm25_score: float = 0.0
    precheck_passed: bool = False
    retrieved_passage_ids: list[str] = field(default_factory=list)
    attempts: list[AttemptRecord] = field(default_factory=list)
    escalated: bool = False
    total_model_calls: int = 0
    final_outcome: str = "failed"  # "passed", "refused", or "failed"
    final_answer: str = ""
    final_sources: list[str] = field(default_factory=list)
    provider_error: str | None = None

    def add_attempt(self, attempt: AttemptRecord) -> None:
        """Add an attempt record and increment call counter."""
        self.attempts.append(attempt)
        self.total_model_calls = len(self.attempts)

    def to_dict(self) -> dict[str, Any]:
        """Convert trace to JSON-serializable dictionary."""
        return {
            "question": self.question,
            "detected_language": self.detected_language,
            "retrieval": {
                "passages_count": self.retrieval_passages_count,
                "top_bm25_score": self.top_bm25_score,
                "precheck_passed": self.precheck_passed,
                "retrieved_passage_ids": self.retrieved_passage_ids,
            },
            "attempts": [att.to_dict() for att in self.attempts],
            "escalated": self.escalated,
            "total_model_calls": self.total_model_calls,
            "final_outcome": self.final_outcome,
            "final_answer": self.final_answer,
            "final_sources": self.final_sources,
            "provider_error": self.provider_error,
        }
