"""Unit tests for full GroundingHarness pipeline with simulated/fake provider transports.

Why these tests exist:
These tests verify end-to-end pipeline execution without requiring external network access
or spending model quota:
1. Pre-check refusal before making model calls (0 calls)
2. Clean Tier-1 pass (1 call, escalated=False)
3. Model refuse=True causing answer replacement with canonical safe refusals in en/hi/mr
4. Tier-1 failure triggering single escalation to Tier-2 (2 calls, escalated=True)
5. Double failure handling
6. Provider fallback on empty content and length truncation
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from typing import Any

from groundgate.cache import DiskCache
from groundgate.harness import GroundingHarness, get_safe_refusal
from groundgate.providers import ProviderManager


class TestHarness(unittest.TestCase):
    """Test suite for GroundingHarness with simulated provider responses."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.cache = DiskCache(cache_dir=self.temp_dir, replay_only=False)

        self.knowledge_pack = [
            {
                "id": "DOC_EN_01",
                "title": "PM Kisan Scheme",
                "text": "The PM Kisan Scheme provides 6000 rupees annually to small landholding farmers.",
                "lang": "en",
            },
            {
                "id": "DOC_MR_01",
                "title": "पीएम किसान योजना",
                "text": "पीएम किसान योजनेअंतर्गत अल्पभूधारक शेतकऱ्यांना दरवर्षी ६००० रुपये आर्थिक सहाय्य दिले जाते.",
                "lang": "mr",
            },
            {
                "id": "DOC_HI_01",
                "title": "पीएम किसान योजना",
                "text": "पीएम किसान योजना के तहत पात्र किसानों को प्रति वर्ष 6000 रुपये दिए जाते हैं।",
                "lang": "hi",
            },
        ]

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_precheck_refusal_zero_model_calls(self) -> None:
        """Unrelated question fails BM25 precheck; refuses with 0 model calls."""
        def failing_transport(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("Network should not be called on precheck refusal!")

        pm = ProviderManager(
            provider_order=["nvidia"],
            cache=self.cache,
            transport=failing_transport,
        )
        harness = GroundingHarness(
            provider_manager=pm,
            min_bm25_score=2.0,
        )

        response = harness.ask(
            question="What is the chemical composition of basalt rocks on Mars?",
            knowledge_pack=self.knowledge_pack,
        )

        self.assertEqual(response["outcome"], "refused")
        self.assertEqual(response["model_calls"], 0)
        self.assertFalse(response["escalated"])
        self.assertEqual(response["answer"], get_safe_refusal("en"))

    def test_clean_tier1_pass(self) -> None:
        """Well-grounded answer passes Tier-1 with 1 model call and no escalation."""
        call_count = 0

        def mock_transport(url: str, payload: dict[str, Any], api_key: str, timeout: int) -> tuple[int, Any, Any, float]:
            nonlocal call_count
            call_count += 1
            body = {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps({
                                "answer": "Under the PM Kisan Scheme, small landholding farmers receive 6000 rupees annually.",
                                "sources": ["DOC_EN_01"],
                                "refuse": False,
                            })
                        },
                    }
                ]
            }
            return 200, body, None, 0.25

        pm = ProviderManager(
            provider_order=["nvidia"],
            cache=self.cache,
            transport=mock_transport,
        )
        harness = GroundingHarness(provider_manager=pm, min_bm25_score=0.5)

        response = harness.ask(
            question="What is the annual benefit for small farmers under PM Kisan?",
            knowledge_pack=self.knowledge_pack,
        )

        self.assertEqual(response["outcome"], "passed")
        self.assertEqual(response["model_calls"], 1)
        self.assertFalse(response["escalated"])
        self.assertIn("6000", response["answer"])
        self.assertEqual(response["sources"], ["DOC_EN_01"])

    def test_model_refusal_discards_model_text_and_uses_safe_refusal(self) -> None:
        """When model outputs refuse=true, harness discards model text and uses safe refusal."""
        def mock_transport(url: str, payload: dict[str, Any], api_key: str, timeout: int) -> tuple[int, Any, Any, float]:
            body = {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            # Model returned conversational apology, which must be discarded
                            "content": json.dumps({
                                "answer": "I am so sorry, I do not have enough information to answer this.",
                                "sources": [],
                                "refuse": True,
                            })
                        },
                    }
                ]
            }
            return 200, body, None, 0.20

        pm = ProviderManager(
            provider_order=["nvidia"],
            cache=self.cache,
            transport=mock_transport,
        )
        harness = GroundingHarness(provider_manager=pm, min_bm25_score=0.5)

        marathi_q = "पीएम किसान योजनेअंतर्गत शेतकऱ्यांना किती मदत मिळते?"
        response = harness.ask(question=marathi_q, knowledge_pack=self.knowledge_pack)

        self.assertEqual(response["outcome"], "refused")
        # Model text was discarded in favor of Marathi safe refusal
        self.assertEqual(response["answer"], get_safe_refusal("mr"))
        self.assertEqual(response["sources"], [])

    def test_tier1_failure_triggers_single_escalation_to_tier2(self) -> None:
        """Tier-1 ungrounded number triggers escalation to Tier-2, which succeeds."""
        calls: list[str] = []

        def mock_transport(url: str, payload: dict[str, Any], api_key: str, timeout: int) -> tuple[int, Any, Any, float]:
            model = payload.get("model", "")
            calls.append(model)
            if len(calls) == 1:
                # Tier-1 hallucinated 15000 instead of 6000
                content = json.dumps({
                    "answer": "The PM Kisan Scheme provides 15000 rupees annually.",
                    "sources": ["DOC_EN_01"],
                    "refuse": False,
                })
            else:
                # Tier-2 corrects to 6000
                content = json.dumps({
                    "answer": "Under the PM Kisan Scheme, small landholding farmers receive 6000 rupees annually.",
                    "sources": ["DOC_EN_01"],
                    "refuse": False,
                })

            body = {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
            return 200, body, None, 0.30

        pm = ProviderManager(
            provider_order=["nvidia"],
            cache=self.cache,
            transport=mock_transport,
        )
        harness = GroundingHarness(
            provider_manager=pm,
            min_bm25_score=0.5,
            tier1_model="model-tier-1",
            tier2_model="model-tier-2",
        )

        response = harness.ask(
            question="What is the benefit under PM Kisan for small farmers?",
            knowledge_pack=self.knowledge_pack,
        )

        self.assertEqual(response["outcome"], "passed")
        self.assertEqual(response["model_calls"], 2)
        self.assertTrue(response["escalated"])
        self.assertEqual(calls, ["model-tier-1", "model-tier-2"])
        self.assertIn("6000", response["answer"])

    def test_double_failure_safely_refuses(self) -> None:
        """If Tier-1 and Tier-2 both fail verification, harness safely refuses."""
        def mock_transport(url: str, payload: dict[str, Any], api_key: str, timeout: int) -> tuple[int, Any, Any, float]:
            # Both attempts hallucinate non-existent citation ID
            content = json.dumps({
                "answer": "Scheme provides 6000 rupees.",
                "sources": ["DOC_NON_EXISTENT_99"],
                "refuse": False,
            })
            body = {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
            return 200, body, None, 0.20

        pm = ProviderManager(
            provider_order=["nvidia"],
            cache=self.cache,
            transport=mock_transport,
        )
        harness = GroundingHarness(provider_manager=pm, min_bm25_score=0.5)

        # Query matches PM Kisan passage to pass precheck
        response = harness.ask(
            question="What is the PM Kisan annual grant amount for small farmers?",
            knowledge_pack=self.knowledge_pack,
        )

        self.assertEqual(response["outcome"], "failed")
        self.assertEqual(response["model_calls"], 2)
        self.assertTrue(response["escalated"])
        self.assertEqual(response["answer"], get_safe_refusal("en"))

    def test_provider_fallback_on_empty_content_and_length(self) -> None:
        """Provider falls through to next provider on empty content or finish_reason='length'."""
        attempted_urls: list[str] = []

        def mock_transport(url: str, payload: dict[str, Any], api_key: str, timeout: int) -> tuple[int, Any, Any, float]:
            attempted_urls.append(url)
            if "integrate.api.nvidia.com" in url:
                # Nvidia returns finish_reason 'length' or empty content
                body = {"choices": [{"finish_reason": "length", "message": {"content": ""}}]}
                return 200, body, None, 0.10
            # OpenRouter succeeds
            body = {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps({
                                "answer": "Small farmers receive 6000 rupees.",
                                "sources": ["DOC_EN_01"],
                                "refuse": False,
                            })
                        },
                    }
                ]
            }
            return 200, body, None, 0.15

        pm = ProviderManager(
            provider_order=["nvidia", "openrouter"],
            cache=self.cache,
            transport=mock_transport,
        )

        res = pm.call_model(
            model_id="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )

        self.assertEqual(res.provider, "openrouter")
        self.assertEqual(len(attempted_urls), 2)


if __name__ == "__main__":
    unittest.main()
