"""Deterministic grounding verification gate for multilingual QA.

Why this module exists:
Traditional RAG evaluation uses an LLM-as-a-judge, which is non-deterministic, slow,
adds significant API costs, and often struggles with Devanagari numerals and low-resource
languages. This gate runs 100% deterministic Python checks to verify:
1. JSON contract compliance and strict type enforcement
2. Valid source citations (must be a subset of retrieved passage IDs)
3. Number grounding (every figure/percentage must exist in the cited passages)
4. Token overlap (substantive content tokens must meet threshold)
5. Script and language consistency between question and answer
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from groundgate.normalize import (
    detect_script,
    extract_numbers,
    tokenize,
)

DEFAULT_OVERLAP_THRESHOLD: float = 0.60
SCRIPT_MISMATCH_THRESHOLD: float = 0.30


@dataclass(frozen=True)
class GateResult:
    """Result of deterministic grounding verification.

    Why dataclass(frozen=True):
    Immutability guarantees gate results cannot be altered downstream after verification.
    """
    passed: bool
    outcome: str  # "passed", "refused", or "failed"
    checks: dict[str, bool]
    failures: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to serializable dictionary for logging and API responses."""
        return {
            "passed": self.passed,
            "outcome": self.outcome,
            "checks": self.checks,
            "failures": self.failures,
            "details": self.details,
        }


def parse_model_json(raw_output: str | dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Parse JSON model response, stripping markdown code block fences if present.

    Why: Many open-weight instruction-tuned models wrap JSON in markdown blocks
    (e.g., ```json ... ```) even when instructed to return raw JSON.
    """
    if isinstance(raw_output, dict):
        return raw_output, None

    if not isinstance(raw_output, str):
        return None, f"Expected string or dict, received {type(raw_output).__name__}"

    cleaned = raw_output.strip()

    # Strip markdown fences if present
    fence_pattern = re.compile(r"^```(?:json)?\s*([\s\S]*?)\s*```$", re.MULTILINE)
    match = fence_pattern.match(cleaned)
    if match:
        cleaned = match.group(1).strip()

    # Also look for first '{' and last '}' if model included preamble text
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        cleaned = cleaned[start_idx : end_idx + 1]

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data, None
        return None, "Parsed JSON is not an object/dict"
    except json.JSONDecodeError as exc:
        return None, f"JSON parse error: {exc}"


def check_sources_real(
    cited_sources: Sequence[str],
    retrieved_ids: set[str],
) -> tuple[bool, str | None]:
    """Verify that cited sources are non-empty and exist in the retrieved passages.

    Why: Hallucinated citations are a common failure mode. The model must only cite
    passages that were actually provided in its prompt context.
    """
    if not cited_sources:
        return False, "Answer cites no sources (sources list is empty)."

    cited_set = {str(s).strip() for s in cited_sources if str(s).strip()}
    if not cited_set:
        return False, "Answer contains only empty or whitespace source citations."

    invalid_ids = cited_set - retrieved_ids
    if invalid_ids:
        sorted_invalid = sorted(list(invalid_ids))
        return False, f"Cited source ID(s) not in retrieved passages: {sorted_invalid}."

    return True, None


def check_numbers_grounded(
    answer_text: str,
    cited_passages_text: str,
) -> tuple[bool, list[str]]:
    """Verify that every factual number in the answer appears in the cited passages.

    Numbers are compared as sets with '%' stripped from both sides, without substring
    fallback. For example:
    - Answer '2' vs passage '2,000' fails ('2' is not in {'2000'}).
    - Answer '50' vs passage '50%' passes ('50' is in {'50'}).

    Why: Citizens and farmers rely on specific figures (e.g. subsidy amounts,
    eligibility hectares, dates). A substring fallback caused '2' to match '2,000',
    which allowed severe factual misrepresentations.
    """
    answer_numbers = extract_numbers(answer_text)
    if not answer_numbers:
        # No numbers to verify in the answer
        return True, []

    cited_numbers = extract_numbers(cited_passages_text)
    # Strip '%' from both sides for exact set comparison
    cited_set = {n.strip("%") for n in cited_numbers if n.strip("%")}

    ungrounded: list[str] = []
    for num in answer_numbers:
        bare_num = num.strip("%")
        if bare_num not in cited_set:
            ungrounded.append(num)

    if ungrounded:
        return False, ungrounded
    return True, []


def check_token_overlap(
    answer_text: str,
    cited_passages_text: str,
    threshold: float = DEFAULT_OVERLAP_THRESHOLD,
    lang: str | None = None,
) -> tuple[bool, float]:
    """Calculate substantive content token overlap between answer and cited passages.

    Why: A hallucinated answer might cite a real passage ID but discuss completely
    unrelated topics. Overlap measures how much of the answer's vocabulary is
    grounded in the cited passages.
    """
    answer_tokens = tokenize(answer_text, remove_stopwords=True, lang=lang)
    if not answer_tokens:
        # If answer has no content tokens after stopword removal, treat as insufficient
        return False, 0.0

    cited_tokens = set(tokenize(cited_passages_text, remove_stopwords=True, lang=lang))
    if not cited_tokens:
        return False, 0.0

    # Fraction of unique content tokens in the answer found in the cited text
    unique_answer_tokens = set(answer_tokens)
    overlap_count = sum(1 for tok in unique_answer_tokens if tok in cited_tokens)
    overlap_ratio = overlap_count / len(unique_answer_tokens)

    return overlap_ratio >= threshold, overlap_ratio


def check_language_match(
    question: str,
    answer_text: str,
) -> tuple[bool, str | None, dict[str, float]]:
    """Check that the answer's script matches the question's script.

    Why: If a user asks a question in Marathi (Devanagari), the model must not
    switch to English, and vice versa. Script consistency is a core requirement.
    """
    q_ratio = detect_script(question)
    a_ratio = detect_script(answer_text)

    stats = {
        "question_devanagari_ratio": round(q_ratio, 3),
        "answer_devanagari_ratio": round(a_ratio, 3),
    }

    # If question is Devanagari (Hindi/Marathi)
    if q_ratio >= SCRIPT_MISMATCH_THRESHOLD:
        if a_ratio < 0.20:
            return (
                False,
                f"Question was in Devanagari script ({q_ratio:.1%}), but answer is in Latin script ({a_ratio:.1%}).",
                stats,
            )
    else:
        # If question is Latin (English)
        if a_ratio >= SCRIPT_MISMATCH_THRESHOLD:
            return (
                False,
                f"Question was in Latin script, but answer is in Devanagari script ({a_ratio:.1%}).",
                stats,
            )

    return True, None, stats


def verify_grounding(
    model_output: str | dict[str, Any],
    retrieved_passages: Sequence[Mapping[str, Any]],
    question: str,
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
) -> GateResult:
    """Run all deterministic gate checks against a model response.

    Returns a GateResult containing passed status, outcome, and itemized failure reasons.

    Checks:
    1. json_valid: Conforms to JSON contract with answer (str), sources (list of str), refuse (real bool)
    2. refusal_ok: If refuse=True, accepted as a legitimate refusal (warnings noted without failing)
    3. sources_real: Sources must be a non-empty subset of retrieved IDs
    4. numbers_grounded: All numerical values must exist in cited passages
    5. token_overlap: Substantive token overlap must meet threshold
    6. language_match: Script ratio must match question script
    """
    checks: dict[str, bool] = {
        "json_valid": False,
        "refusal_ok": False,
        "sources_real": False,
        "numbers_grounded": False,
        "token_overlap": False,
        "language_match": False,
    }
    failures: list[str] = []
    details: dict[str, Any] = {}

    # Check 1: JSON validity and contract structure
    data, parse_err = parse_model_json(model_output)
    if data is None or parse_err is not None:
        failures.append(f"Invalid JSON: {parse_err}")
        return GateResult(
            passed=False,
            outcome="failed",
            checks=checks,
            failures=failures,
            details={"error": parse_err},
        )

    # Contract requires keys: answer (str), sources (list[str]), refuse (bool)
    if "answer" not in data or "sources" not in data or "refuse" not in data:
        failures.append("JSON missing required contract fields: 'answer', 'sources', 'refuse'.")
        return GateResult(
            passed=False,
            outcome="failed",
            checks=checks,
            failures=failures,
            details={"data": data},
        )

    raw_answer = data["answer"]
    raw_sources = data["sources"]
    raw_refuse = data["refuse"]

    # Strict type validation:
    # answer must be str, sources must be list[str], refuse must be real bool (reject "false" or "true")
    if not isinstance(raw_answer, str):
        failures.append(f"Contract field 'answer' must be str, got {type(raw_answer).__name__}.")
    if not isinstance(raw_sources, list) or not all(isinstance(s, str) for s in raw_sources):
        failures.append(f"Contract field 'sources' must be a list of strings, got {type(raw_sources).__name__}.")
    # bool is a subclass of int in Python, but isinstance(raw_refuse, bool) checks for actual boolean
    if not isinstance(raw_refuse, bool):
        failures.append(
            f"Contract field 'refuse' must be a boolean (True/False), got {type(raw_refuse).__name__} (value: {raw_refuse!r})."
        )

    if failures:
        return GateResult(
            passed=False,
            outcome="failed",
            checks=checks,
            failures=failures,
            details={"data": data},
        )

    checks["json_valid"] = True
    answer_text = raw_answer.strip()
    sources = raw_sources
    refuse = raw_refuse

    details["answer"] = answer_text
    details["sources"] = sources
    details["refuse"] = refuse

    # Check 2: Refusal handling
    # If the model explicitly refused because the knowledge pack doesn't have the answer,
    # this is a successful, safe outcome.
    if refuse:
        checks["refusal_ok"] = True
        lang_ok, lang_err, lang_stats = check_language_match(question, answer_text)
        checks["language_match"] = lang_ok
        details.update(lang_stats)

        # On refusals, a language mismatch must not fail the gate.
        # Record it in details["warnings"] and keep outcome "refused".
        if not lang_ok and lang_err:
            details.setdefault("warnings", []).append(f"Refusal script mismatch: {lang_err}")

        return GateResult(
            passed=True,
            outcome="refused",
            checks=checks,
            failures=[],
            details=details,
        )

    checks["refusal_ok"] = True

    # Retrieve valid passage IDs and compile cited passages text
    retrieved_id_map: dict[str, str] = {
        str(p.get("id", "")).strip(): str(p.get("text", p.get("content", "")))
        for p in retrieved_passages
        if str(p.get("id", "")).strip()
    }

    # Check 3: Sources real and subset of retrieved IDs
    sources_ok, sources_err = check_sources_real(sources, set(retrieved_id_map.keys()))
    checks["sources_real"] = sources_ok
    if not sources_ok and sources_err:
        failures.append(sources_err)

    # Aggregate text only from valid cited passages
    cited_text_parts: list[str] = [
        retrieved_id_map[str(s).strip()]
        for s in sources
        if str(s).strip() in retrieved_id_map
    ]
    cited_passages_text = " ".join(cited_text_parts)

    # Check 4: Numbers grounded
    numbers_ok, ungrounded_numbers = check_numbers_grounded(answer_text, cited_passages_text)
    checks["numbers_grounded"] = numbers_ok
    if not numbers_ok:
        failures.append(
            f"Answer contains ungrounded number(s) not found in cited sources: {ungrounded_numbers}"
        )
    details["ungrounded_numbers"] = ungrounded_numbers

    # Check 5: Token overlap
    overlap_ok, overlap_ratio = check_token_overlap(
        answer_text,
        cited_passages_text,
        threshold=overlap_threshold,
    )
    checks["token_overlap"] = overlap_ok
    details["token_overlap_ratio"] = round(overlap_ratio, 3)
    details["overlap_threshold"] = overlap_threshold
    if not overlap_ok:
        failures.append(
            f"Token overlap ({overlap_ratio:.1%}) is below threshold ({overlap_threshold:.1%})."
        )

    # Check 6: Language / script consistency
    lang_ok, lang_err, lang_stats = check_language_match(question, answer_text)
    checks["language_match"] = lang_ok
    details.update(lang_stats)
    if not lang_ok and lang_err:
        failures.append(lang_err)

    all_passed = all(checks.values())
    outcome = "passed" if all_passed else "failed"

    return GateResult(
        passed=all_passed,
        outcome=outcome,
        checks=checks,
        failures=failures,
        details=details,
    )
