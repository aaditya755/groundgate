#!/usr/bin/env python3
"""Multilingual RAG benchmark runner comparing Raw vs LLM-as-Judge vs GroundGate.

Why this runner exists:
To evaluate grounding strategies objectively, we run identical queries through:
1. (a) Raw baseline: Tier-1 model with JSON contract and retrieval, but NO gate.
2. (b) LLM-as-Judge: Raw output evaluated by a second Tier-1 model call for groundedness.
3. (c) Gate (groundgate): Full deterministic verification with single-escalation cascade.

All three setups share the exact same retrieval logic and prompt formatting, allowing
identical prompt calls to hit the disk cache and share baseline model executions.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

# Add workspace root to sys.path so groundgate package can be imported directly
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Load .env so provider keys and model IDs reach the harness
from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from groundgate.contract import build_messages, format_context
from groundgate.gate import parse_model_json, verify_grounding
from groundgate.harness import GroundingHarness, get_safe_refusal
from groundgate.normalize import candidate_languages
from groundgate.providers import ProviderManager, RateLimiter
from groundgate.retrieve import retrieve_passages

JUDGE_SYSTEM_PROMPT = """You are a strict factual grounding verifier.
Review the provided Reference Context, Question, and Candidate Answer.
Determine whether EVERY factual statement and number in Candidate Answer is strictly
supported by the Reference Context. Do not allow assumptions or outside knowledge.
Return raw JSON with a single boolean field:
{"grounded": true}
or
{"grounded": false}
Do not wrap your response in markdown fences. Return raw JSON only."""


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Load records from a UTF-8 JSONL file."""
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                records.append(json.loads(stripped))
    return records


def compute_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute benchmark metrics across a collection of evaluation records.

    Why this function is decoupled:
    Permits deterministic unit testing on synthetic result sets without invoking network.
    """
    total = len(records)
    if total == 0:
        return {
            "total_questions": 0,
            "citation_valid_rate": 0.0,
            "ungrounded_answer_rate": 0.0,
            "wrong_answer_rate_on_traps": 0.0,
            "correct_refusal_rate_on_traps": 0.0,
            "refusal_rate_on_answerable": 0.0,
            "avg_model_calls": 0.0,
            "avg_extra_model_calls": 0.0,
            "avg_latency": 0.0,
            "escalation_rate": 0.0,
        }

    traps = [r for r in records if not r.get("answerable", True)]
    answerables = [r for r in records if r.get("answerable", True)]
    answered = [r for r in records if r.get("outcome") == "passed"]

    # 1. Citation-valid rate: answered outputs whose sources are non-empty subset of retrieved IDs
    if answered:
        valid_citations = 0
        for r in answered:
            sources = set(r.get("sources", []))
            retrieved = set(r.get("retrieved_passage_ids", []))
            if sources and sources.issubset(retrieved):
                valid_citations += 1
        citation_valid_rate = round(valid_citations / len(answered), 3)
    else:
        citation_valid_rate = 0.0

    # 2. Ungrounded-answer rate: answered outputs that fail deterministic grounding checks
    if answered:
        ungrounded = 0
        for r in answered:
            # If the record already recorded gate_passed=False, or verification fails
            if not r.get("is_grounded", True):
                ungrounded += 1
        ungrounded_answer_rate = round(ungrounded / len(answered), 3)
    else:
        ungrounded_answer_rate = 0.0

    # 3. Wrong-answer rate on traps (answered when it should have refused)
    if traps:
        wrong_on_traps = sum(1 for r in traps if r.get("outcome") == "passed")
        wrong_answer_rate_on_traps = round(wrong_on_traps / len(traps), 3)
    else:
        wrong_answer_rate_on_traps = 0.0

    # 4. Correct-refusal rate on traps
    if traps:
        correct_refusals = sum(1 for r in traps if r.get("outcome") in ("refused", "failed"))
        correct_refusal_rate_on_traps = round(correct_refusals / len(traps), 3)
    else:
        correct_refusal_rate_on_traps = 0.0

    # 5. Refusal rate on answerable questions (over-abstention)
    if answerables:
        refused_on_ans = sum(1 for r in answerables if r.get("outcome") != "passed")
        refusal_rate_on_answerable = round(refused_on_ans / len(answerables), 3)
    else:
        refusal_rate_on_answerable = 0.0

    total_calls = sum(r.get("model_calls", 0) for r in records)
    avg_model_calls = round(total_calls / total, 2)
    # Extra model calls beyond the baseline 1 call
    avg_extra_model_calls = round(sum(max(0, r.get("model_calls", 0) - 1) for r in records) / total, 2)

    total_lat = sum(r.get("latency", 0.0) for r in records)
    avg_latency = round(total_lat / total, 2)

    escalated_count = sum(1 for r in records if r.get("escalated", False))
    escalation_rate = round(escalated_count / total, 3)

    return {
        "total_questions": total,
        "citation_valid_rate": citation_valid_rate,
        "ungrounded_answer_rate": ungrounded_answer_rate,
        "wrong_answer_rate_on_traps": wrong_answer_rate_on_traps,
        "correct_refusal_rate_on_traps": correct_refusal_rate_on_traps,
        "refusal_rate_on_answerable": refusal_rate_on_answerable,
        "avg_model_calls": avg_model_calls,
        "avg_extra_model_calls": avg_extra_model_calls,
        "avg_latency": avg_latency,
        "escalation_rate": escalation_rate,
    }


def eval_single_question_raw(
    item: dict[str, Any],
    knowledge_pack: list[dict[str, Any]],
    provider_manager: ProviderManager,
    min_bm25_score: float,
    no_precheck: bool,
) -> dict[str, Any]:
    """Execute raw setup: retrieve, optional pre-check, Tier-1 model call, JSON parse, NO gate."""
    question = item["question"]
    q_id = item["id"]
    lang = item["lang"]
    answerable = item["answerable"]

    candidates = candidate_languages(question)
    effective_min_score = -999.0 if no_precheck else min_bm25_score

    start_t = time.perf_counter()
    retrieved, top_score, precheck_passed = retrieve_passages(
        query=question,
        passages=knowledge_pack,
        top_k=3,
        min_score=effective_min_score,
        filter_language=True,
    )

    resolved_lang = candidates[0] if len(candidates) == 1 else (
        retrieved[0].get("lang", candidates[0]) if retrieved else candidates[0]
    )

    retrieved_ids = [str(p.get("id", "")) for p in retrieved]

    # Pre-check refusal
    if not precheck_passed or not retrieved:
        elapsed = time.perf_counter() - start_t
        safe_ans = get_safe_refusal(str(resolved_lang))
        return {
            "id": q_id,
            "setup": "raw",
            "lang": lang,
            "answerable": answerable,
            "outcome": "refused",
            "answer": safe_ans,
            "sources": [],
            "retrieved_passage_ids": retrieved_ids,
            "model_calls": 0,
            "escalated": False,
            "latency": round(elapsed, 2),
            "is_grounded": True,
        }

    messages = build_messages(question, retrieved)
    comp = provider_manager.call_model(tier="tier1", messages=messages)
    elapsed = time.perf_counter() - start_t

    parsed_json, parse_err = parse_model_json(comp.text)
    if not parsed_json or parse_err:
        return {
            "id": q_id,
            "setup": "raw",
            "lang": lang,
            "answerable": answerable,
            "outcome": "failed",
            "answer": comp.text,
            "sources": [],
            "retrieved_passage_ids": retrieved_ids,
            "model_calls": 1,
            "escalated": False,
            "latency": round(elapsed, 2),
            "is_grounded": False,
        }

    if parsed_json.get("refuse", False):
        safe_ans = get_safe_refusal(str(resolved_lang))
        return {
            "id": q_id,
            "setup": "raw",
            "lang": lang,
            "answerable": answerable,
            "outcome": "refused",
            "answer": safe_ans,
            "sources": [],
            "retrieved_passage_ids": retrieved_ids,
            "model_calls": 1,
            "escalated": False,
            "latency": round(elapsed, 2),
            "is_grounded": True,
        }

    raw_answer = str(parsed_json.get("answer", ""))
    raw_sources = [str(s) for s in parsed_json.get("sources", [])]

    # Audit ground truth using gate checks to evaluate raw baseline quality
    gate_check = verify_grounding(
        model_output=comp.text,
        retrieved_passages=retrieved,
        question=question,
    )

    return {
        "id": q_id,
        "setup": "raw",
        "lang": lang,
        "answerable": answerable,
        "outcome": "passed",
        "answer": raw_answer,
        "sources": raw_sources,
        "retrieved_passage_ids": retrieved_ids,
        "model_calls": 1,
        "escalated": False,
        "latency": round(elapsed, 2),
        "is_grounded": gate_check.passed,
    }


def eval_single_question_llm_judge(
    item: dict[str, Any],
    knowledge_pack: list[dict[str, Any]],
    provider_manager: ProviderManager,
    min_bm25_score: float,
    no_precheck: bool,
) -> dict[str, Any]:
    """Execute LLM-as-a-judge setup: Tier-1 answer evaluated by second Tier-1 judge call."""
    raw_res = eval_single_question_raw(
        item=item,
        knowledge_pack=knowledge_pack,
        provider_manager=provider_manager,
        min_bm25_score=min_bm25_score,
        no_precheck=no_precheck,
    )

    # If raw refused or failed initially, no judge call is required
    if raw_res["outcome"] != "passed":
        res = dict(raw_res)
        res["setup"] = "llm_judge"
        return res

    question = item["question"]
    lang = item["lang"]
    answer_text = raw_res["answer"]
    retrieved_ids = set(raw_res["retrieved_passage_ids"])
    cited_passages = [p for p in knowledge_pack if str(p.get("id")) in retrieved_ids]
    context_str = format_context(cited_passages)

    judge_prompt = (
        f"Reference Context:\n{context_str}\n\n"
        f"Question:\n{question}\n\n"
        f"Candidate Answer:\n{answer_text}\n\n"
        f"Is the Candidate Answer completely grounded in the Reference Context without hallucinated facts or numbers?"
    )

    judge_messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": judge_prompt},
    ]

    start_judge = time.perf_counter()
    judge_comp = provider_manager.call_model(tier="tier1", messages=judge_messages)
    judge_elapsed = time.perf_counter() - start_judge

    judge_json, _ = parse_model_json(judge_comp.text)
    is_judge_grounded = bool(judge_json.get("grounded", False)) if judge_json else False

    total_latency = round(raw_res["latency"] + judge_elapsed, 2)
    resolved_lang = candidate_languages(question)[0]

    if not is_judge_grounded:
        safe_ans = get_safe_refusal(resolved_lang)
        return {
            "id": item["id"],
            "setup": "llm_judge",
            "lang": lang,
            "answerable": item["answerable"],
            "outcome": "refused",
            "answer": safe_ans,
            "sources": [],
            "retrieved_passage_ids": raw_res["retrieved_passage_ids"],
            "model_calls": 2,
            "escalated": False,
            "latency": total_latency,
            "is_grounded": True,
        }

    # Verify with deterministic checks for comparative reporting
    gate_check = verify_grounding(
        model_output={"answer": answer_text, "sources": raw_res["sources"], "refuse": False},
        retrieved_passages=cited_passages,
        question=question,
    )

    return {
        "id": item["id"],
        "setup": "llm_judge",
        "lang": lang,
        "answerable": item["answerable"],
        "outcome": "passed",
        "answer": answer_text,
        "sources": raw_res["sources"],
        "retrieved_passage_ids": raw_res["retrieved_passage_ids"],
        "model_calls": 2,
        "escalated": False,
        "latency": total_latency,
        "is_grounded": gate_check.passed,
    }


def eval_single_question_gate(
    item: dict[str, Any],
    knowledge_pack: list[dict[str, Any]],
    harness: GroundingHarness,
) -> dict[str, Any]:
    """Execute gate setup using standard GroundingHarness pipeline."""
    start_t = time.perf_counter()
    res = harness.ask(
        question=item["question"],
        knowledge_pack=knowledge_pack,
    )
    elapsed = time.perf_counter() - start_t

    trace = res.get("trace", {})
    retrieval_meta = trace.get("retrieval", {})
    retrieved_ids = retrieval_meta.get("retrieved_passage_ids", [])

    is_grounded = (res["outcome"] == "passed")

    return {
        "id": item["id"],
        "setup": "gate",
        "lang": item["lang"],
        "answerable": item["answerable"],
        "outcome": res["outcome"],
        "answer": res["answer"],
        "sources": res["sources"],
        "retrieved_passage_ids": retrieved_ids,
        "model_calls": res["model_calls"],
        "escalated": res["escalated"],
        "latency": round(elapsed, 2),
        "is_grounded": is_grounded,
    }


def run_benchmark(
    questions: list[dict[str, Any]],
    knowledge_pack: list[dict[str, Any]],
    setups: list[str],
    provider_manager: ProviderManager,
    min_bm25_score: float = 1.5,
    overlap_threshold: float = 0.60,
    no_precheck: bool = False,
    workers: int = 3,
) -> dict[str, Any]:
    """Run benchmark across specified setups and compile metrics."""
    harness = GroundingHarness(
        provider_manager=provider_manager,
        min_bm25_score=-999.0 if no_precheck else min_bm25_score,
        overlap_threshold=overlap_threshold,
    )

    results_by_setup: dict[str, list[dict[str, Any]]] = {s: [] for s in setups}

    for setup_name in setups:
        def worker(q_item: dict[str, Any]) -> dict[str, Any]:
            if setup_name == "raw":
                return eval_single_question_raw(
                    item=q_item,
                    knowledge_pack=knowledge_pack,
                    provider_manager=provider_manager,
                    min_bm25_score=min_bm25_score,
                    no_precheck=no_precheck,
                )
            elif setup_name == "llm_judge":
                return eval_single_question_llm_judge(
                    item=q_item,
                    knowledge_pack=knowledge_pack,
                    provider_manager=provider_manager,
                    min_bm25_score=min_bm25_score,
                    no_precheck=no_precheck,
                )
            elif setup_name == "gate":
                return eval_single_question_gate(
                    item=q_item,
                    knowledge_pack=knowledge_pack,
                    harness=harness,
                )
            raise ValueError(f"Unknown setup: {setup_name}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            records = list(executor.map(worker, questions))

        results_by_setup[setup_name] = records

    # Calculate overall and per-language metrics
    summary: dict[str, Any] = {}
    for s_name, records in results_by_setup.items():
        summary[s_name] = {
            "overall": compute_metrics(records),
            "by_language": {},
        }
        for l_code in ("en", "hi", "mr"):
            sub_records = [r for r in records if r["lang"] == l_code]
            summary[s_name]["by_language"][l_code] = compute_metrics(sub_records)

    return {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "questions_count": len(questions),
            "passages_count": len(knowledge_pack),
            "setups": setups,
            "min_bm25_score": min_bm25_score,
            "overlap_threshold": overlap_threshold,
            "no_precheck": no_precheck,
        },
        "summary": summary,
        "details": results_by_setup,
    }


def write_results_markdown(
    results: dict[str, Any],
    output_path: Path | str,
    sweep_results: dict[float, dict[str, Any]] | None = None,
) -> None:
    """Generate results.md document with clear scientific caveats and metric comparisons."""
    summary = results["summary"]
    lines: list[str] = [
        "# groundgate Multilingual Evaluation Benchmark Results",
        "",
        "## Important Methodological Caveats",
        "1. **Circularity Note:** The ungrounded-answer rate for Setup (c) (`gate`) is computed using",
        "   the same deterministic Python checks that define the gate itself. Therefore, it is a circular",
        "   consistency verification and must NOT be presented as an independent external accuracy metric.",
        "2. **Sample Size:** With a 30-question evaluation dataset, observed percentage variations between",
        "   setups are illustrative and anecdotal; they do not represent statistical significance.",
        "3. **Placeholder Knowledge Pack:** The knowledge pack consists of synthetic, fictional agricultural",
        "   schemes and clearly fake amounts to prevent real-world misguidance during hackathon development.",
        "",
        "## Setup Performance Comparison",
        "",
        "| Setup | Questions | Citation Valid | Ungrounded Answers | Trap Wrong Answer | Trap Refusal | Answerable Refusal | Avg Calls | Avg Extra Calls | Avg Latency | Escalation Rate |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for setup_name, data in summary.items():
        o = data["overall"]
        lines.append(
            f"| {setup_name} | {o['total_questions']} | {o['citation_valid_rate']:.1%} | "
            f"{o['ungrounded_answer_rate']:.1%} | {o['wrong_answer_rate_on_traps']:.1%} | "
            f"{o['correct_refusal_rate_on_traps']:.1%} | {o['refusal_rate_on_answerable']:.1%} | "
            f"{o['avg_model_calls']} | {o['avg_extra_model_calls']} | {o['avg_latency']}s | "
            f"{o['escalation_rate']:.1%} |"
        )

    lines.extend([
        "",
        "## Performance by Language",
        "",
    ])

    for setup_name, data in summary.items():
        lines.append(f"### Setup: {setup_name}")
        lines.append("| Language | Questions | Citation Valid | Ungrounded Answers | Trap Refusal | Answerable Refusal | Avg Calls | Avg Latency |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for lang_code, m in data["by_language"].items():
            lines.append(
                f"| {lang_code.upper()} | {m['total_questions']} | {m['citation_valid_rate']:.1%} | "
                f"{m['ungrounded_answer_rate']:.1%} | {m['correct_refusal_rate_on_traps']:.1%} | "
                f"{m['refusal_rate_on_answerable']:.1%} | {m['avg_model_calls']} | {m['avg_latency']}s |"
            )
        lines.append("")

    if sweep_results:
        lines.extend([
            "## Gate Overlap Threshold Sensitivity Sweep",
            "",
            "| Overlap Threshold | Citation Valid | Ungrounded Rate | Trap Refusal Rate | Over-Abstention | Escalation Rate |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ])
        for thresh, res in sorted(sweep_results.items()):
            o = res["summary"]["gate"]["overall"]
            lines.append(
                f"| {thresh:.2f} | {o['citation_valid_rate']:.1%} | {o['ungrounded_answer_rate']:.1%} | "
                f"{o['correct_refusal_rate_on_traps']:.1%} | {o['refusal_rate_on_answerable']:.1%} | "
                f"{o['escalation_rate']:.1%} |"
            )
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def print_summary_table(results: dict[str, Any]) -> None:
    """Print ASCII summary table to console without Unicode box characters."""
    summary = results["summary"]
    print("\n" + "=" * 105)
    print(f"{'Setup':<12} {'Total':<6} {'CitValid':<10} {'Ungrounded':<12} {'TrapWrong':<11} {'TrapRefuse':<12} {'AnsRefuse':<11} {'Calls':<7} {'Extra':<7} {'Latency':<8}")
    print("-" * 105)
    for s_name, data in summary.items():
        o = data["overall"]
        cit_val = f"{o['citation_valid_rate']:.1%}"
        ungrounded = f"{o['ungrounded_answer_rate']:.1%}"
        trap_wrong = f"{o['wrong_answer_rate_on_traps']:.1%}"
        trap_ref = f"{o['correct_refusal_rate_on_traps']:.1%}"
        ans_ref = f"{o['refusal_rate_on_answerable']:.1%}"
        lat_str = f"{o['avg_latency']}s"
        print(
            f"{s_name:<12} {o['total_questions']:<6} {cit_val:<10} "
            f"{ungrounded:<12} {trap_wrong:<11} "
            f"{trap_ref:<12} {ans_ref:<11} "
            f"{o['avg_model_calls']:<7} {o['avg_extra_model_calls']:<7} {lat_str:<8}"
        )
    print("=" * 105)


def main() -> None:
    """Main CLI entrypoint for running evaluation benchmarks."""
    parser = argparse.ArgumentParser(description="groundgate multilingual grounding benchmark")
    parser.add_argument("--setups", default="raw,llm_judge,gate", help="Comma-separated setups to run")
    parser.add_argument("--lang", default="all", help="Language filter: 'all', 'en', 'hi', or 'mr'")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to evaluate")
    parser.add_argument("--workers", type=int, default=3, help="Concurrent worker threads (default 3)")
    parser.add_argument("--dry-run", action="store_true", help="Print call estimates and exit without network")
    parser.add_argument("--no-precheck", action="store_true", help="Disable BM25 pre-check scoring")
    parser.add_argument("--sweep-overlap", action="store_true", help="Run sensitivity sweep across overlap thresholds")
    parser.add_argument("--replay", action="store_true", help="Run strictly from disk cache (REPLAY_ONLY)")
    parser.add_argument("--knowledge-pack", default=str(REPO_ROOT / "data" / "knowledge_pack.jsonl"))
    parser.add_argument("--questions", default=str(REPO_ROOT / "eval" / "questions.jsonl"))
    parser.add_argument("--output-json", default=str(REPO_ROOT / "eval" / "results" / "results.json"))
    parser.add_argument("--output-md", default=str(REPO_ROOT / "eval" / "results" / "results.md"))
    args = parser.parse_args()

    if args.replay:
        os.environ["REPLAY_ONLY"] = "true"

    knowledge_pack = load_jsonl(args.knowledge_pack)
    questions = load_jsonl(args.questions)

    if args.lang != "all":
        questions = [q for q in questions if q.get("lang") == args.lang]

    if args.limit:
        questions = questions[: args.limit]

    chosen_setups = [s.strip().lower() for s in args.setups.split(",") if s.strip()]

    print("==========================================================================")
    print("  groundgate Multilingual Evaluation Benchmark")
    print("==========================================================================")
    print(f"Loaded Questions       : {len(questions)}")
    print(f"Loaded Passages        : {len(knowledge_pack)}")
    print(f"Target Setups          : {chosen_setups}")
    print(f"Language Filter        : {args.lang}")
    print(f"Concurrency Workers    : {args.workers}")
    print(f"REPLAY_ONLY Mode       : {os.environ.get('REPLAY_ONLY', 'false')}")

    if args.dry_run:
        print("\n--- Dry Run Call Estimates ---")
        q_count = len(questions)
        for s in chosen_setups:
            if s == "raw":
                est = f"{q_count} calls (1 Tier-1 per question)"
            elif s == "llm_judge":
                est = f"{q_count * 2} calls (1 Tier-1 + 1 Judge per question)"
            elif s == "gate":
                est = f"{q_count} - {q_count * 2} calls (1 Tier-1 + 1 Tier-2 on escalation)"
            else:
                est = "unknown"
            print(f"  Setup {s:<10}: approx {est}")
        print("Dry run completed. No network calls executed.")
        return

    # Initialize shared RateLimiter and ProviderManager
    rpm = int(os.environ.get("REQUESTS_PER_MINUTE", "20"))
    rate_limiter = RateLimiter(requests_per_minute=rpm)
    provider_manager = ProviderManager(rate_limiter=rate_limiter)

    print("\nRunning benchmark across evaluation questions...")
    results = run_benchmark(
        questions=questions,
        knowledge_pack=knowledge_pack,
        setups=chosen_setups,
        provider_manager=provider_manager,
        min_bm25_score=1.5,
        overlap_threshold=0.60,
        no_precheck=args.no_precheck,
        workers=args.workers,
    )

    sweep_results: dict[float, dict[str, Any]] | None = None
    if args.sweep_overlap and "gate" in chosen_setups:
        print("\nRunning gate threshold sweep across [0.4, 0.5, 0.6, 0.7, 0.8]...")
        sweep_results = {}
        for thresh in [0.4, 0.5, 0.6, 0.7, 0.8]:
            sw_res = run_benchmark(
                questions=questions,
                knowledge_pack=knowledge_pack,
                setups=["gate"],
                provider_manager=provider_manager,
                overlap_threshold=thresh,
                no_precheck=args.no_precheck,
                workers=args.workers,
            )
            sweep_results[thresh] = sw_res

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    write_results_markdown(
        results=results,
        output_path=args.output_md,
        sweep_results=sweep_results,
    )

    print(f"\nWrote results JSON to: {args.output_json}")
    print(f"Wrote results Markdown to: {args.output_md}")
    print_summary_table(results)


if __name__ == "__main__":
    main()
