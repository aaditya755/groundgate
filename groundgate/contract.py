"""JSON contract specifications, context formatters, and prompt builders for groundgate.

Why this module exists:
Grounding harnesses depend on a predictable interface with open-weight LLMs.
This module creates prompts that constrain the model to structured JSON containing
explicit source citations, answer text, and a refusal flag. It also builds targeted
escalation prompts that feed deterministic gate failure reasons back to Tier-2.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

SYSTEM_PROMPT = """You are a grounded question-answering assistant.
Answer ONLY from the provided facts. Do not assume, extrapolate, or use outside knowledge.
You must return a single JSON object with exact keys:
{
  "answer": "concise answer in the exact same language and script as the question",
  "sources": ["list of passage IDs that directly support your answer"],
  "refuse": false
}

Rules:
1. Every factual claim and number in 'answer' must be present in the cited passages.
2. If the passages do not contain the answer, return:
{
  "answer": "I cannot answer this question based on the provided sources.",
  "sources": [],
  "refuse": true
}
3. 'refuse' must be a boolean (true or false). Do not use string "false".
4. Do not wrap output in markdown code fences. Return raw JSON only."""


def format_context(passages: Sequence[Mapping[str, Any]]) -> str:
    """Format retrieved passages into a structured reference context.

    Why explicit [ID] tags:
    Enables the model to copy exact passage IDs into its 'sources' array so the gate
    can deterministically verify citation authenticity.
    """
    if not passages:
        return "No reference passages available."

    lines: list[str] = []
    for p in passages:
        pid = p.get("id", "UNKNOWN")
        title = p.get("title", "").strip()
        text = p.get("text", p.get("content", "")).strip()
        header = f"[{pid}] {title}" if title else f"[{pid}]"
        lines.append(f"{header}\n{text}")

    return "\n\n".join(lines)


def build_messages(
    question: str,
    passages: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Build standard Tier-1 chat messages conforming to the JSON contract."""
    context_str = format_context(passages)
    user_prompt = f"Reference Context:\n{context_str}\n\nQuestion: {question}"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_escalation_messages(
    question: str,
    passages: Sequence[Mapping[str, Any]],
    previous_output: str,
    failure_reasons: Sequence[str],
) -> list[dict[str, str]]:
    """Build Tier-2 escalation messages detailing previous gate failures for self-correction.

    Why feedback matters:
    Escalation is not just re-running with a larger model; it explicitly informs the model
    which numbers were ungrounded or citations invalid, drastically improving adherence.
    """
    context_str = format_context(passages)
    failures_formatted = "\n".join(f"- {reason}" for reason in failure_reasons)

    user_prompt = (
        f"Reference Context:\n{context_str}\n\n"
        f"Question: {question}\n\n"
        f"A previous attempt generated the following output, which FAILED verification:\n"
        f"{previous_output}\n\n"
        f"Verification failures identified:\n"
        f"{failures_formatted}\n\n"
        f"Correction instructions:\n"
        f"Carefully correct the output. Ensure every number and claim is strictly verified\n"
        f"in the reference passages. Cite only valid passage IDs. If the passages do not\n"
        f"support the claim, set 'refuse': true. Return raw JSON."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
