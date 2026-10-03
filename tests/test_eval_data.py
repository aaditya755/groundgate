"""Unit tests for knowledge pack integrity, evaluation dataset consistency, and benchmark metrics.

Why these tests exist:
Evaluation datasets must have verified structure before running benchmarks:
1. Every topic in the knowledge pack must have complete parallel triplets (en, hi, mr).
2. Every expected source cited by a question must exist in the knowledge pack.
3. Dataset distributions (20 answerable, 10 traps across languages) must remain consistent.
4. Metric calculations (citation validity, ungrounded rates, trap refusal) must be mathematically correct.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from eval.run_eval import compute_metrics, load_jsonl

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestEvalData(unittest.TestCase):
    """Test suite for evaluation data schema and metric calculation accuracy."""

    def setUp(self) -> None:
        self.pack_path = REPO_ROOT / "data" / "knowledge_pack.jsonl"
        self.questions_path = REPO_ROOT / "eval" / "questions.jsonl"
        self.pack = load_jsonl(self.pack_path)
        self.questions = load_jsonl(self.questions_path)

    def test_knowledge_pack_is_valid_jsonl(self) -> None:
        """knowledge_pack.jsonl is valid JSONL with required schema fields."""
        self.assertGreater(len(self.pack), 0)
        required_fields = {"id", "topic_id", "title", "text", "lang", "source_url"}
        for row in self.pack:
            self.assertTrue(required_fields.issubset(set(row.keys())))
            self.assertTrue(row["text"].startswith("PLACEHOLDER - replace with verified text"))
            self.assertEqual(row["source_url"], "")

    def test_knowledge_pack_triplets(self) -> None:
        """Every topic_id has exactly one en, one hi, and one mr row."""
        topic_languages: dict[str, set[str]] = {}
        topic_counts: dict[str, int] = {}

        for row in self.pack:
            tid = row["topic_id"]
            lang = row["lang"]
            topic_languages.setdefault(tid, set()).add(lang)
            topic_counts[tid] = topic_counts.get(tid, 0) + 1

        self.assertEqual(len(topic_languages), 9)  # 9 topics
        self.assertEqual(len(self.pack), 27)  # 27 rows total (9 * 3)

        for tid, langs in topic_languages.items():
            self.assertEqual(langs, {"en", "hi", "mr"}, f"Topic {tid} missing required languages")
            self.assertEqual(topic_counts[tid], 3, f"Topic {tid} has incorrect row count")

    def test_questions_count_and_distribution(self) -> None:
        """questions.jsonl has exactly 30 rows with 20 answerable and 10 traps."""
        self.assertEqual(len(self.questions), 30)

        answerables = [q for q in self.questions if q["answerable"]]
        traps = [q for q in self.questions if not q["answerable"]]

        self.assertEqual(len(answerables), 20)
        self.assertEqual(len(traps), 10)

        # Verify language distribution: 10 en, 10 hi, 10 mr
        en_qs = [q for q in self.questions if q["lang"] == "en"]
        hi_qs = [q for q in self.questions if q["lang"] == "hi"]
        mr_qs = [q for q in self.questions if q["lang"] == "mr"]

        self.assertEqual(len(en_qs), 10)
        self.assertEqual(len(hi_qs), 10)
        self.assertEqual(len(mr_qs), 10)

    def test_expected_sources_exist_in_knowledge_pack(self) -> None:
        """Every expected_sources id cited in questions.jsonl exists in the knowledge pack."""
        known_ids = {row["id"] for row in self.pack}
        for q in self.questions:
            expected = q.get("expected_sources", [])
            if not q["answerable"]:
                self.assertEqual(expected, [], f"Trap question {q['id']} must have empty expected_sources")
            else:
                self.assertGreater(len(expected), 0, f"Answerable question {q['id']} must specify expected sources")
                for sid in expected:
                    self.assertIn(sid, known_ids, f"Source ID {sid} cited in {q['id']} not found in pack")

    def test_compute_metrics_accuracy(self) -> None:
        """compute_metrics calculates rates correctly on a hand-crafted test record set."""
        mock_records = [
            # 1. Answerable question, passed, valid citations, grounded
            {
                "answerable": True,
                "outcome": "passed",
                "sources": ["T01_EN"],
                "retrieved_passage_ids": ["T01_EN", "T02_EN"],
                "model_calls": 1,
                "latency": 1.0,
                "escalated": False,
                "is_grounded": True,
            },
            # 2. Answerable question, passed, hallucinated source ID, ungrounded
            {
                "answerable": True,
                "outcome": "passed",
                "sources": ["FAKE_SOURCE_99"],
                "retrieved_passage_ids": ["T01_EN"],
                "model_calls": 2,
                "latency": 2.0,
                "escalated": True,
                "is_grounded": False,
            },
            # 3. Trap question, refused (correct refusal)
            {
                "answerable": False,
                "outcome": "refused",
                "sources": [],
                "retrieved_passage_ids": [],
                "model_calls": 0,
                "latency": 0.1,
                "escalated": False,
                "is_grounded": True,
            },
            # 4. Trap question, erroneously answered (wrong answer on trap)
            {
                "answerable": False,
                "outcome": "passed",
                "sources": ["T02_EN"],
                "retrieved_passage_ids": ["T02_EN"],
                "model_calls": 1,
                "latency": 0.9,
                "escalated": False,
                "is_grounded": False,
            },
        ]

        metrics = compute_metrics(mock_records)

        # 3 answered out of 4 records
        # valid citations: records 1 and 4 have valid citations -> 2 / 3 = 0.667
        self.assertEqual(metrics["citation_valid_rate"], 0.667)

        # ungrounded: records 2 and 4 are ungrounded -> 2 / 3 = 0.667
        self.assertEqual(metrics["ungrounded_answer_rate"], 0.667)

        # 2 traps: record 3 refused, record 4 answered
        # wrong answer on traps: 1 / 2 = 0.5
        self.assertEqual(metrics["wrong_answer_rate_on_traps"], 0.5)
        # correct refusal on traps: 1 / 2 = 0.5
        self.assertEqual(metrics["correct_refusal_rate_on_traps"], 0.5)

        # 2 answerables: both passed -> refusal on answerable = 0.0
        self.assertEqual(metrics["refusal_rate_on_answerable"], 0.0)

        # average model calls: (1 + 2 + 0 + 1) / 4 = 1.0
        self.assertEqual(metrics["avg_model_calls"], 1.0)
        self.assertEqual(metrics["avg_extra_model_calls"], 0.0)

        # average latency: (1.0 + 2.0 + 0.1 + 0.9) / 4 = 1.0
        self.assertEqual(metrics["avg_latency"], 1.0)

        # escalation rate: 1 escalated out of 4 = 0.25
        self.assertEqual(metrics["escalation_rate"], 0.25)


if __name__ == "__main__":
    unittest.main()
