"""Unit tests for deterministic grounding verification gate.

Why these tests exist:
The grounding gate is the core defense mechanism of groundgate. It must deterministically
prevent hallucinations (ungrounded numbers, hallucinated citation IDs, low token overlap,
type violations, and unprompted script switches) without making non-deterministic model calls.
"""

from __future__ import annotations

import unittest

from groundgate.gate import (
    GateResult,
    check_language_match,
    check_numbers_grounded,
    check_sources_real,
    check_token_overlap,
    parse_model_json,
    verify_grounding,
)


class TestGate(unittest.TestCase):
    """Test suite for deterministic grounding checks and full gate pipeline."""

    def setUp(self) -> None:
        """Sample mock retrieved passages for verification testing."""
        self.retrieved_passages = [
            {
                "id": "DOC_EN_01",
                "title": "PM Kisan Scheme",
                "text": "The PM Kisan Scheme provides 6,000 rupees annually in three installments of 2,000 rupees to small landholding farmers.",
                "lang": "en",
            },
            {
                "id": "DOC_MR_01",
                "title": "पीएम किसान योजना",
                "text": "पीएम किसान योजनेअंतर्गत अल्पभूधारक शेतकऱ्यांना दरवर्षी ६,००० रुपये तीन हप्त्यांमध्ये (प्रत्येकी २,००० रुपये) दिले जातात.",
                "lang": "mr",
            },
        ]

    def test_parse_model_json_markdown_fences(self) -> None:
        """parse_model_json must cleanly strip markdown code block fences and whitespace."""
        raw = """```json
{
  "answer": "Eligible farmers receive 6000 rupees annually.",
  "sources": ["DOC_EN_01"],
  "refuse": false
}
```"""
        data, err = parse_model_json(raw)
        self.assertIsNone(err)
        self.assertIsNotNone(data)
        assert data is not None
        self.assertEqual(data["sources"], ["DOC_EN_01"])
        self.assertFalse(data["refuse"])

    def test_sources_real_valid_subset(self) -> None:
        """sources_real passes when all cited sources exist in retrieved passage IDs."""
        valid_ids = {"DOC_EN_01", "DOC_MR_01"}
        ok, err = check_sources_real(["DOC_EN_01"], valid_ids)
        self.assertTrue(ok)
        self.assertIsNone(err)

    def test_sources_real_hallucinated_id(self) -> None:
        """sources_real fails when model invents or cites a non-existent source ID."""
        valid_ids = {"DOC_EN_01", "DOC_MR_01"}
        ok, err = check_sources_real(["DOC_EN_01", "DOC_UNKNOWN_99"], valid_ids)
        self.assertFalse(ok)
        self.assertIsNotNone(err)
        assert err is not None
        self.assertIn("DOC_UNKNOWN_99", err)

    def test_sources_real_empty_citations(self) -> None:
        """sources_real fails when the answer provides an empty sources list."""
        valid_ids = {"DOC_EN_01"}
        ok, err = check_sources_real([], valid_ids)
        self.assertFalse(ok)
        self.assertIsNotNone(err)

    def test_numbers_grounded_answer_2_vs_passage_2000_fails(self) -> None:
        """Fix 2: Substring fallback removed. Answer '2' vs passage '2,000' must FAIL."""
        passage_text = "The installment is 2,000 rupees."
        answer_text = "The installment is 2 rupees."
        ok, ungrounded = check_numbers_grounded(answer_text, passage_text)
        self.assertFalse(ok)
        self.assertIn("2", ungrounded)

    def test_numbers_grounded_answer_50_vs_passage_50_percent_passes(self) -> None:
        """Fix 2: Compare numbers as sets with '%' stripped. Answer '50' vs passage '50%' must PASS."""
        passage_text = "The government provides a 50% subsidy on seeds."
        answer_text = "The seed subsidy is 50 percent."
        ok, ungrounded = check_numbers_grounded(answer_text, passage_text)
        self.assertTrue(ok)
        self.assertEqual(ungrounded, [])

    def test_numbers_grounded_pass(self) -> None:
        """numbers_grounded passes when all numbers in the answer appear in cited passages."""
        passage_text = "Farmers get 6,000 rupees per year with 2,000 per installment and 50% subsidy."
        answer_text = "The financial grant is 6000 rupees paid as 2000 per installment with a 50% subsidy."
        ok, ungrounded = check_numbers_grounded(answer_text, passage_text)
        self.assertTrue(ok)
        self.assertEqual(ungrounded, [])

    def test_numbers_grounded_catches_invented_number(self) -> None:
        """numbers_grounded catches hallucinated figures (e.g. 10000 instead of 6000)."""
        passage_text = "The financial grant is 6,000 rupees."
        hallucinated_answer = "Eligible farmers receive 10,000 rupees per year."
        ok, ungrounded = check_numbers_grounded(hallucinated_answer, passage_text)
        self.assertFalse(ok)
        self.assertIn("10000", ungrounded)

    def test_numbers_grounded_devanagari_numeral_matching(self) -> None:
        """numbers_grounded correctly matches Devanagari numerals (६,०००) against ASCII figures."""
        passage_text = "पीएम किसान योजनेअंतर्गत ६,००० रुपये दिले जातात."
        answer_devanagari = "योजनेअंतर्गत शेतकऱ्यांना ६००० रुपये मिळतात."
        ok, ungrounded = check_numbers_grounded(answer_devanagari, passage_text)
        self.assertTrue(ok)
        self.assertEqual(ungrounded, [])

        answer_ascii = "योजनेअंतर्गत शेतकऱ्यांना 6000 रुपये मिळतात."
        ok2, ungrounded2 = check_numbers_grounded(answer_ascii, passage_text)
        self.assertTrue(ok2)
        self.assertEqual(ungrounded2, [])

    def test_token_overlap_threshold(self) -> None:
        """check_token_overlap calculates vocabulary overlap accurately against threshold."""
        passage_text = "The PM Kisan Scheme provides 6000 rupees annually to small landholding farmers."
        grounded_answer = "PM Kisan Scheme provides 6000 rupees to small landholding farmers."
        ok, ratio = check_token_overlap(grounded_answer, passage_text, threshold=0.60)
        self.assertTrue(ok)
        self.assertGreaterEqual(ratio, 0.60)

        # Unrelated answer with low overlap
        unrelated_answer = "Solar panels are installed on rooftop buildings across the state."
        ok_bad, ratio_bad = check_token_overlap(unrelated_answer, passage_text, threshold=0.60)
        self.assertFalse(ok_bad)
        self.assertLess(ratio_bad, 0.60)

    def test_language_match(self) -> None:
        """check_language_match ensures model answers in the question's script."""
        en_q = "How much financial support is provided annually?"
        en_a = "Eligible farmers receive 6000 rupees annually."
        ok_en, err_en, _ = check_language_match(en_q, en_a)
        self.assertTrue(ok_en)
        self.assertIsNone(err_en)

        mr_q = "योजनेअंतर्गत किती आर्थिक सहाय्य दिले जाते?"
        mr_a = "शेतकऱ्यांना दरवर्षी ६,००० रुपयांचे सहाय्य दिले जाते."
        ok_mr, err_mr, _ = check_language_match(mr_q, mr_a)
        self.assertTrue(ok_mr)
        self.assertIsNone(err_mr)

        # Mismatch: Marathi question, but English answer
        ok_mismatch, err_mismatch, _ = check_language_match(mr_q, en_a)
        self.assertFalse(ok_mismatch)
        self.assertIsNotNone(err_mismatch)

    def test_verify_grounding_pass(self) -> None:
        """Full verify_grounding pipeline passes for fully compliant answer."""
        model_output = {
            "answer": "Under the PM Kisan Scheme, small landholding farmers receive 6000 rupees annually.",
            "sources": ["DOC_EN_01"],
            "refuse": False,
        }
        question = "What is the annual benefit for small farmers under PM Kisan?"
        result: GateResult = verify_grounding(
            model_output,
            self.retrieved_passages,
            question,
            overlap_threshold=0.60,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.outcome, "passed")
        self.assertEqual(result.failures, [])
        self.assertTrue(result.checks["json_valid"])
        self.assertTrue(result.checks["sources_real"])
        self.assertTrue(result.checks["numbers_grounded"])
        self.assertTrue(result.checks["token_overlap"])
        self.assertTrue(result.checks["language_match"])

    def test_verify_grounding_rejects_string_refuse_false(self) -> None:
        """Fix 3: Reject 'false' string for refuse; must be a real boolean."""
        bad_type_output = {
            "answer": "Under the PM Kisan Scheme, small landholding farmers receive 6000 rupees annually.",
            "sources": ["DOC_EN_01"],
            "refuse": "false",  # String instead of real bool
        }
        question = "What is the annual benefit for small farmers under PM Kisan?"
        result = verify_grounding(
            bad_type_output,
            self.retrieved_passages,
            question,
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.outcome, "failed")
        self.assertTrue(any("Contract field 'refuse' must be a boolean" in f for f in result.failures))

    def test_verify_grounding_rejects_non_str_answer_or_sources(self) -> None:
        """Fix 3: Reject non-str answer or non-string list in sources."""
        # Non-string answer
        bad_answer = {
            "answer": 6000,
            "sources": ["DOC_EN_01"],
            "refuse": False,
        }
        res_a = verify_grounding(bad_answer, self.retrieved_passages, "Question?")
        self.assertFalse(res_a.passed)
        self.assertTrue(any("must be str" in f for f in res_a.failures))

        # Non-string elements in sources
        bad_sources = {
            "answer": "Answer text",
            "sources": [123, "DOC_EN_01"],
            "refuse": False,
        }
        res_s = verify_grounding(bad_sources, self.retrieved_passages, "Question?")
        self.assertFalse(res_s.passed)
        self.assertTrue(any("list of strings" in f for f in res_s.failures))

    def test_verify_grounding_refusal_accepted(self) -> None:
        """A valid refusal (refuse=True) passes the gate with outcome 'refused'."""
        refusal_output = {
            "answer": "I cannot answer this question based on the provided sources.",
            "sources": [],
            "refuse": True,
        }
        question = "What is the subsidy for tractors in 2030?"
        result: GateResult = verify_grounding(
            refusal_output,
            self.retrieved_passages,
            question,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.outcome, "refused")
        self.assertEqual(result.failures, [])
        self.assertTrue(result.checks["refusal_ok"])

    def test_verify_grounding_refusal_language_mismatch_keeps_refused(self) -> None:
        """Fix 5: On refusals, language mismatch must not fail the gate. Record in warnings."""
        # Question is in Marathi, but model refused in English
        marathi_question = "ट्रॅक्टर अनुदानाबाबत नियम काय आहेत?"
        english_refusal = {
            "answer": "I cannot answer this question based on the provided sources.",
            "sources": [],
            "refuse": True,
        }
        result = verify_grounding(
            english_refusal,
            self.retrieved_passages,
            marathi_question,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.outcome, "refused")
        self.assertEqual(result.failures, [])
        self.assertIn("warnings", result.details)
        self.assertTrue(any("Refusal script mismatch" in w for w in result.details["warnings"]))

    def test_verify_grounding_invented_number_fails(self) -> None:
        """Answer containing an invented number fails verification with detailed reasons."""
        bad_output = {
            "answer": "Under the PM Kisan Scheme, farmers receive 15000 rupees annually.",
            "sources": ["DOC_EN_01"],
            "refuse": False,
        }
        question = "What is the benefit under PM Kisan?"
        result: GateResult = verify_grounding(
            bad_output,
            self.retrieved_passages,
            question,
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.outcome, "failed")
        self.assertFalse(result.checks["numbers_grounded"])
        self.assertTrue(any("15000" in f for f in result.failures))


if __name__ == "__main__":
    unittest.main()
