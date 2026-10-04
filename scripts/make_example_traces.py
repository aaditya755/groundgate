#!/usr/bin/env python3
"""Record example traces from REAL harness runs for the web UI.

Why this script exists:
The UI offers pre-recorded examples so a visitor can see the gate work even when
no live model call is possible. Every value in the output file comes from a real
harness run (the disk cache makes repeat runs free). Nothing is hand-written.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

import os  # noqa: E402

from groundgate.harness import GroundingHarness  # noqa: E402
from groundgate.providers import ProviderManager, RateLimiter  # noqa: E402

# why: reuse questions that already exist in the benchmark (one per language,
# a near-miss trap and an off-topic trap) so no text is retyped here
QUESTION_IDS = ["Q_EN_01", "Q_HI_01", "Q_MR_10", "Q_EN_08"]
LANG_NAMES = {"en": "English", "hi": "Hindi", "mr": "Marathi"}
NOTICE = "Pre-recorded example from a real run, not a live answer."


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text and not text.startswith("#"):
                rows.append(json.loads(text))
    return rows


def label_for(result: dict) -> str:
    outcome = result["outcome"]
    if outcome == "passed":
        return "answered and verified"
    if outcome == "refused" and result["model_calls"] == 0:
        return "refused by the pre-check"
    if outcome == "refused":
        return "refused after the model declined"
    return "no verified answer"


def describe(result: dict) -> str:
    """Describe what REALLY happened, using only values from the run."""
    outcome = result["outcome"]
    calls = result["model_calls"]
    if outcome == "passed" and not result["escalated"]:
        return f"Answered by tier-1 and verified by the gate ({calls} model call)."
    if outcome == "passed":
        return f"Tier-1 failed the gate; escalated once to tier-2 and verified ({calls} model calls)."
    if outcome == "refused" and calls == 0:
        return "Refused by the BM25 pre-check before any model call."
    if outcome == "refused":
        return "The model declined; the canonical safe refusal was returned."
    return "No answer passed the gate; the canonical safe refusal was returned."


def main() -> None:
    questions = {q["id"]: q for q in load_jsonl(REPO_ROOT / "eval" / "questions.jsonl")}
    pack = load_jsonl(REPO_ROOT / "data" / "knowledge_pack.jsonl")

    rpm = int(os.environ.get("REQUESTS_PER_MINUTE", "20"))
    manager = ProviderManager(rate_limiter=RateLimiter(requests_per_minute=rpm))
    harness = GroundingHarness(provider_manager=manager)

    examples = []
    for qid in QUESTION_IDS:
        item = questions.get(qid)
        if item is None:
            print(f"{qid}: not found in eval/questions.jsonl, skipped")
            continue

        result = harness.ask(question=item["question"], knowledge_pack=pack)
        trace = result.get("trace", {})

        # why: a network failure is not a useful example, so never record one
        if result["outcome"] == "failed" and trace.get("provider_error"):
            print(f"{qid}: provider error, skipped (check network and keys, then rerun)")
            continue

        examples.append(
            {
                "id": f"example_{qid.lower()}",
                "title": f"{LANG_NAMES.get(item['lang'], item['lang'])} question: {label_for(result)}",
                "scenario": describe(result),
                "question": item["question"],
                "lang": item["lang"],
                "answer": result["answer"],
                "sources": result["sources"],
                "outcome": result["outcome"],
                "escalated": result["escalated"],
                "model_calls": result["model_calls"],
                "trace": trace,
                "is_cached_example": True,
                "notice": NOTICE,
                "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            }
        )
        print(f"{qid}: outcome={result['outcome']} calls={result['model_calls']} escalated={result['escalated']}")

    if not examples:
        print("No examples recorded; leaving any existing file untouched.")
        sys.exit(1)

    out_dir = REPO_ROOT / "app" / "static"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "example_traces.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False, default=str)
    print(f"Wrote {len(examples)} real example(s) to {out_path}")


if __name__ == "__main__":
    main()