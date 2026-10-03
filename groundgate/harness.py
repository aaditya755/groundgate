"""Grounding harness pipeline with pre-check, single escalation cascade, and language-aware refusals.

Why this module exists:
This is the core orchestrator of groundgate:
1. Detects question language (en/hi/mr) and filters retrieval.
2. BM25 Pre-check: If relevance is below threshold, safely refuses BEFORE making model calls (saves quota).
3. Invokes Tier-1 fast model with structured JSON contract.
4. Deterministic Gate: Runs pure-code verification checks.
5. Single Escalation: If Tier-1 fails verification, escalates ONCE to Tier-2 with itemized failure reasons.
6. Safe Refusals: Whenever a refusal occurs (pre-check or model refuse=true), discards generated text and
   returns a verified, safe refusal in the user's language.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

from groundgate.contract import build_escalation_messages, build_messages
from groundgate.gate import GateResult, verify_grounding
from groundgate.normalize import detect_language
from groundgate.providers import ProviderManager
from groundgate.retrieve import retrieve_passages
from groundgate.trace import AttemptRecord, ExecutionTrace

# Standard safe refusal strings in each supported language
# Why: LLMs often append apologetic or conversational text during refusals.
# Replacing model text with verified strings ensures 100% deterministic, clean refusals.
SAFE_REFUSALS: dict[str, str] = {
    "en": "I cannot answer this question based on the provided sources.",
    "hi": "प्रदान किए गए स्रोतों के आधार पर मैं इस प्रश्न का उत्तर नहीं दे सकता।",
    "mr": "उपलब्ध माहितीच्या आधारे मी या प्रश्नाचे उत्तर देऊ शकत नाही.",
}


def get_safe_refusal(lang: str) -> str:
    """Return canonical safe refusal message for the requested language."""
    return SAFE_REFUSALS.get(lang.lower(), SAFE_REFUSALS["en"])


class GroundingHarness:
    """Pipeline orchestrator for grounded question answering."""

    def __init__(
        self,
        provider_manager: ProviderManager | None = None,
        min_bm25_score: float | None = None,
        overlap_threshold: float | None = None,
        tier1_model: str | None = None,
        tier2_model: str | None = None,
    ) -> None:
        self.provider_manager = provider_manager or ProviderManager()
        self.min_bm25_score = (
            min_bm25_score
            if min_bm25_score is not None
            else float(os.environ.get("MIN_BM25_SCORE", "1.5"))
        )
        self.overlap_threshold = (
            overlap_threshold
            if overlap_threshold is not None
            else float(os.environ.get("OVERLAP_THRESHOLD", "0.60"))
        )
        self.tier1_model = (
            tier1_model or os.environ.get("TIER1_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
        ).strip()
        self.tier2_model = (
            tier2_model or os.environ.get("TIER2_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
        ).strip()

    def ask(
        self,
        question: str,
        knowledge_pack: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Execute the full grounded QA pipeline for a user question.

        Returns:
            dict containing:
            - answer: str (final answer or safe refusal)
            - sources: list[str] (verified passage IDs cited)
            - outcome: str ("passed", "refused", or "failed")
            - escalated: bool (whether Tier-2 model was invoked)
            - model_calls: int (total model executions)
            - trace: dict (detailed step-by-step audit record)
        """
        detected_lang = detect_language(question)
        trace = ExecutionTrace(
            question=question,
            detected_language=detected_lang,
        )

        # Step 1: Retrieval filtered by detected language
        retrieved, top_score, precheck_passed = retrieve_passages(
            query=question,
            passages=knowledge_pack,
            top_k=3,
            min_score=self.min_bm25_score,
            filter_language=True,
        )

        trace.retrieval_passages_count = len(retrieved)
        trace.top_bm25_score = top_score
        trace.precheck_passed = precheck_passed
        trace.retrieved_passage_ids = [str(p.get("id", "")) for p in retrieved]

        # Step 2: Pre-check refusal (saves API quota when query is unsupported)
        if not precheck_passed or not retrieved:
            safe_text = get_safe_refusal(detected_lang)
            trace.final_outcome = "refused"
            trace.final_answer = safe_text
            trace.final_sources = []
            return {
                "answer": safe_text,
                "sources": [],
                "outcome": "refused",
                "escalated": False,
                "model_calls": 0,
                "trace": trace.to_dict(),
            }

        # Step 3: Tier-1 initial model call
        tier1_messages = build_messages(question, retrieved)
        try:
            comp_tier1 = self.provider_manager.call_model(
                model_id=self.tier1_model,
                messages=tier1_messages,
            )
        except Exception as exc:
            # If all providers fail, refuse safely
            safe_text = get_safe_refusal(detected_lang)
            trace.final_outcome = "failed"
            trace.final_answer = safe_text
            trace.final_sources = []
            return {
                "answer": safe_text,
                "sources": [],
                "outcome": "failed",
                "escalated": False,
                "model_calls": 0,
                "trace": trace.to_dict(),
            }

        gate_res_1: GateResult = verify_grounding(
            model_output=comp_tier1.text,
            retrieved_passages=retrieved,
            question=question,
            overlap_threshold=self.overlap_threshold,
        )

        record_1 = AttemptRecord(
            tier="tier1",
            provider=comp_tier1.provider,
            model=comp_tier1.model,
            latency=comp_tier1.latency,
            finish_reason=comp_tier1.finish_reason,
            raw_output=comp_tier1.text,
            gate_passed=gate_res_1.passed,
            gate_outcome=gate_res_1.outcome,
            gate_checks=gate_res_1.checks,
            gate_failures=gate_res_1.failures,
            gate_details=gate_res_1.details,
        )
        trace.add_attempt(record_1)

        # Check for model refusal on Tier-1
        # RULE: When model returns refuse=True, DISCARD model's text and return safe refusal
        if gate_res_1.outcome == "refused":
            safe_text = get_safe_refusal(detected_lang)
            trace.final_outcome = "refused"
            trace.final_answer = safe_text
            trace.final_sources = []
            return {
                "answer": safe_text,
                "sources": [],
                "outcome": "refused",
                "escalated": False,
                "model_calls": trace.total_model_calls,
                "trace": trace.to_dict(),
            }

        # If Tier-1 passes cleanly, return immediately
        if gate_res_1.passed:
            final_ans = str(gate_res_1.details.get("answer", ""))
            final_src = [str(s) for s in gate_res_1.details.get("sources", [])]
            trace.final_outcome = "passed"
            trace.final_answer = final_ans
            trace.final_sources = final_src
            return {
                "answer": final_ans,
                "sources": final_src,
                "outcome": "passed",
                "escalated": False,
                "model_calls": trace.total_model_calls,
                "trace": trace.to_dict(),
            }

        # Step 4: Tier-1 failed verification -> ONE Escalation to Tier-2
        trace.escalated = True
        tier2_messages = build_escalation_messages(
            question=question,
            passages=retrieved,
            previous_output=comp_tier1.text,
            failure_reasons=gate_res_1.failures,
        )

        try:
            comp_tier2 = self.provider_manager.call_model(
                model_id=self.tier2_model,
                messages=tier2_messages,
            )
        except Exception:
            safe_text = get_safe_refusal(detected_lang)
            trace.final_outcome = "failed"
            trace.final_answer = safe_text
            trace.final_sources = []
            return {
                "answer": safe_text,
                "sources": [],
                "outcome": "failed",
                "escalated": True,
                "model_calls": trace.total_model_calls,
                "trace": trace.to_dict(),
            }

        gate_res_2: GateResult = verify_grounding(
            model_output=comp_tier2.text,
            retrieved_passages=retrieved,
            question=question,
            overlap_threshold=self.overlap_threshold,
        )

        record_2 = AttemptRecord(
            tier="tier2",
            provider=comp_tier2.provider,
            model=comp_tier2.model,
            latency=comp_tier2.latency,
            finish_reason=comp_tier2.finish_reason,
            raw_output=comp_tier2.text,
            gate_passed=gate_res_2.passed,
            gate_outcome=gate_res_2.outcome,
            gate_checks=gate_res_2.checks,
            gate_failures=gate_res_2.failures,
            gate_details=gate_res_2.details,
        )
        trace.add_attempt(record_2)

        # Handle Tier-2 refusal
        if gate_res_2.outcome == "refused":
            safe_text = get_safe_refusal(detected_lang)
            trace.final_outcome = "refused"
            trace.final_answer = safe_text
            trace.final_sources = []
            return {
                "answer": safe_text,
                "sources": [],
                "outcome": "refused",
                "escalated": True,
                "model_calls": trace.total_model_calls,
                "trace": trace.to_dict(),
            }

        # If Tier-2 passes
        if gate_res_2.passed:
            final_ans = str(gate_res_2.details.get("answer", ""))
            final_src = [str(s) for s in gate_res_2.details.get("sources", [])]
            trace.final_outcome = "passed"
            trace.final_answer = final_ans
            trace.final_sources = final_src
            return {
                "answer": final_ans,
                "sources": final_src,
                "outcome": "passed",
                "escalated": True,
                "model_calls": trace.total_model_calls,
                "trace": trace.to_dict(),
            }

        # Both Tier-1 and Tier-2 failed verification -> Safely refuse
        safe_text = get_safe_refusal(detected_lang)
        trace.final_outcome = "failed"
        trace.final_answer = safe_text
        trace.final_sources = []
        return {
            "answer": safe_text,
            "sources": [],
            "outcome": "failed",
            "escalated": True,
            "model_calls": trace.total_model_calls,
            "trace": trace.to_dict(),
        }
