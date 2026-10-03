"""Unit tests for BM25 retrieval, character 3-grams, and pre-check scoring.

Why these tests exist:
Retrieval must accurately match relevant passages across English, Hindi, and Marathi.
Character 3-grams guarantee recall for Devanagari morphological variations, while
language filtering guarantees cross-language confusion does not occur.
"""

from __future__ import annotations

import unittest

from groundgate.normalize import candidate_languages
from groundgate.retrieve import (
    BM25Index,
    extract_features,
    retrieve_passages,
)


class TestRetrieve(unittest.TestCase):
    """Test suite for multilingual BM25 retrieval and n-gram feature extraction."""

    def setUp(self) -> None:
        self.sample_passages = [
            {
                "id": "DOC_EN_01",
                "title": "PM Kisan Scheme",
                "text": "Under the PM Kisan scheme, eligible farmers receive 6000 rupees annual income support.",
                "lang": "en",
            },
            {
                "id": "DOC_HI_01",
                "title": "पीएम किसान सम्मान निधि",
                "text": "पीएम किसान योजना के तहत पात्र किसानों को प्रति वर्ष ₹6,000 की वित्तीय सहायता मिलती है।",
                "lang": "hi",
            },
            {
                "id": "DOC_MR_01",
                "title": "पीएम किसान सन्मान योजना",
                "text": "पीएम किसान योजनेअंतर्गत पात्र शेतकऱ्यांना दरवर्षी ६,००० रुपयांचे आर्थिक सहाय्य दिले जाते.",
                "lang": "mr",
            },
            {
                "id": "DOC_EN_02",
                "title": "Solar Pump Subsidy",
                "text": "The solar agriculture pump subsidy covers 90% cost for small landholders.",
                "lang": "en",
            },
            {
                "id": "DOC_MR_02",
                "title": "सौर कृषी पंप अनुदान",
                "text": "सौर कृषी पंप योजनेअंतर्गत अल्पभूधारक शेतकऱ्यांना ९०% अनुदान दिले जाते.",
                "lang": "mr",
            },
        ]

    def test_extract_features(self) -> None:
        """extract_features combines content words and per-word character 3-grams."""
        phrase = "शेतकरी अनुदान"
        features = extract_features(phrase, lang="mr")
        self.assertIn("शेतकरी", features)
        self.assertIn("अनुदान", features)
        self.assertTrue(any(len(f) == 3 for f in features))

    def test_bm25_finds_correct_passage_english(self) -> None:
        """BM25 retrieves PM Kisan document for English question."""
        retrieved, top_score, precheck_passed = retrieve_passages(
            query="What is the annual financial support under PM Kisan?",
            passages=self.sample_passages,
            top_k=2,
            min_score=1.0,
        )
        self.assertTrue(precheck_passed)
        self.assertGreater(top_score, 1.0)
        self.assertEqual(retrieved[0]["id"], "DOC_EN_01")

    def test_bm25_finds_correct_passage_marathi(self) -> None:
        """BM25 retrieves Marathi PM Kisan document for Marathi question."""
        query = "पीएम किसान योजनेअंतर्गत किती आर्थिक सहाय्य दिले जाते?"
        retrieved, top_score, precheck_passed = retrieve_passages(
            query=query,
            passages=self.sample_passages,
            top_k=2,
            min_score=1.0,
        )
        self.assertTrue(precheck_passed)
        self.assertGreater(top_score, 1.0)
        self.assertEqual(retrieved[0]["id"], "DOC_MR_01")
        self.assertEqual(retrieved[0]["lang"], "mr")

    def test_bm25_finds_correct_passage_hindi(self) -> None:
        """BM25 retrieves Hindi PM Kisan document for Hindi question."""
        query = "किसान सम्मान योजना में कितनी वित्तीय सहायता दी जाती है?"
        retrieved, top_score, precheck_passed = retrieve_passages(
            query=query,
            passages=self.sample_passages,
            top_k=2,
            min_score=1.0,
        )
        self.assertTrue(precheck_passed)
        self.assertGreater(top_score, 1.0)
        self.assertEqual(retrieved[0]["id"], "DOC_HI_01")
        self.assertEqual(retrieved[0]["lang"], "hi")

    def test_ambiguous_marathi_question_without_markers_retrieves_marathi_doc(self) -> None:
        """Fix 1: An ambiguous Marathi question without marker words retrieves DOC_MR_01."""
        # Query has no distinct markers and returns ['hi', 'mr'] from candidate_languages
        ambiguous_mr_query = "पीएम किसान सन्मान योजना"
        self.assertEqual(candidate_languages(ambiguous_mr_query), ["hi", "mr"])

        retrieved, top_score, precheck_passed = retrieve_passages(
            query=ambiguous_mr_query,
            passages=self.sample_passages,
            top_k=2,
            min_score=1.0,
        )
        self.assertTrue(precheck_passed)
        self.assertEqual(retrieved[0]["id"], "DOC_MR_01")
        self.assertEqual(retrieved[0]["lang"], "mr")

    def test_precheck_refusal_unrelated_query(self) -> None:
        """Completely unrelated query yields low BM25 score, triggering precheck failure."""
        unrelated_query = "What is the speed of light in quantum astrophysics?"
        retrieved, top_score, precheck_passed = retrieve_passages(
            query=unrelated_query,
            passages=self.sample_passages,
            top_k=2,
            min_score=2.5,
        )
        self.assertFalse(precheck_passed)
        self.assertLess(top_score, 2.5)


if __name__ == "__main__":
    unittest.main()
