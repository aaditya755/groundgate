"""Unit tests for full GroundingHarness pipeline with simulated/fake provider transports.

Why these tests exist:
These tests verify end-to-end pipeline execution without requiring external network access
or spending model quota:
1. Pre-check refusal before making model calls (0 calls)
2. Clean Tier-1 pass (1 call, escalated=False)
3. Model refuse=True causing answer replacement with canonical safe refusals in en/hi/mr
4. Ambiguous Devanagari queries using top-retrieved document language for refusals
5. Tier-1 failure triggering single escalation to Tier-2 (2 calls, escalated=True)
6. Double failure handling and provider error recording in trace
7. Provider fallback on empty content, length truncation, and timeouts
8. 429 retry with exponential backoff
9. REPLAY_ONLY propagation without swallowing CacheMissError
10. Reasoning effort 400 retry fallback
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from typing import Any
from unittest.mock import patch

from groundgate.cache import CacheMissError, DiskCache
from groundgate.harness import GroundingHarness, get_safe_refusal
from groundgate.normalize import candidate_languages
from groundgate.providers import ProviderError, ProviderManager, RateLimiter, get_model_id


class TestHarness(unittest.TestCase):
    """Test suite for GroundingHarness with simulated provider responses."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.cache = DiskCache(cache_dir=self.temp_dir, replay_only=False)
        # Use no-wait limiter (rpm=0) so tests run instantly without throttling sleeps
        self.no_wait_limiter = RateLimiter(requests_per_minute=0)

        # Set default test environment model IDs
        os.environ["TIER1_MODEL"] = "model-tier-1"
        os.environ["TIER2_MODEL"] = "model-tier-2"

        self.knowledge_pack = [
            {
                "id": "DOC_EN_01",
                "title": "PM Kisan Scheme",
                "text": "The PM Kisan Scheme provides 6000 rupees annually to small landholding farmers.",
                "lang": "en",
            },
            {
                "id": "DOC_MR_01",
                "title": "पीएम किसान सन्मान योजना",
                "text": "पीएम किसान सन्मान योजनेअंतर्गत अल्पभूधारक शेतकऱ्यांना दरवर्षी ६००० रुपये आर्थिक सहाय्य दिले जाते.",
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
            rate_limiter=self.no_wait_limiter,
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

    def test_ambiguous_question_retrieves_doc_and_gets_marathi_refusal(self) -> None:
        """Fix 1: An ambiguous Marathi question without marker words retrieves DOC_MR_01 and gets Marathi refusal."""
        pm = ProviderManager(
            provider_order=["nvidia"],
            cache=self.cache,
            transport=lambda *args: (200, {}, None, 0.0),
            rate_limiter=self.no_wait_limiter,
        )
        # Use high min_bm25_score so it triggers a precheck refusal AFTER ranking
        harness = GroundingHarness(provider_manager=pm, min_bm25_score=100.0)

        # Ambiguous query containing Marathi-matching content words but no distinct marker words
        ambiguous_q = "पीएम किसान सन्मान योजना"
        # Confirm that candidate_languages actually returns ['hi', 'mr']
        self.assertEqual(candidate_languages(ambiguous_q), ["hi", "mr"])

        response = harness.ask(question=ambiguous_q, knowledge_pack=self.knowledge_pack)

        self.assertEqual(response["outcome"], "refused")
        # Language must resolve to Marathi based on top retrieved DOC_MR_01
        self.assertEqual(response["trace"]["detected_language"], "mr")
        self.assertEqual(response["answer"], get_safe_refusal("mr"))

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
            rate_limiter=self.no_wait_limiter,
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
            rate_limiter=self.no_wait_limiter,
        )
        harness = GroundingHarness(provider_manager=pm, min_bm25_score=0.5)

        marathi_q = "पीएम किसान योजनेअंतर्गत शेतकऱ्यांना किती मदत मिळते?"
        response = harness.ask(question=marathi_q, knowledge_pack=self.knowledge_pack)

        self.assertEqual(response["outcome"], "refused")
        self.assertEqual(response["answer"], get_safe_refusal("mr"))
        self.assertEqual(response["sources"], [])

    def test_tier1_failure_triggers_single_escalation_to_tier2(self) -> None:
        """Tier-1 ungrounded number triggers escalation to Tier-2, which succeeds."""
        calls: list[str] = []

        def mock_transport(url: str, payload: dict[str, Any], api_key: str, timeout: int) -> tuple[int, Any, Any, float]:
            model = payload.get("model", "")
            calls.append(model)
            if len(calls) == 1:
                content = json.dumps({
                    "answer": "The PM Kisan Scheme provides 15000 rupees annually.",
                    "sources": ["DOC_EN_01"],
                    "refuse": False,
                })
            else:
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
            rate_limiter=self.no_wait_limiter,
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

    def test_double_failure_safely_refuses_and_records_trace(self) -> None:
        """If Tier-1 and Tier-2 both fail verification, harness safely refuses."""
        def mock_transport(url: str, payload: dict[str, Any], api_key: str, timeout: int) -> tuple[int, Any, Any, float]:
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
            rate_limiter=self.no_wait_limiter,
        )
        harness = GroundingHarness(provider_manager=pm, min_bm25_score=0.5)

        response = harness.ask(
            question="What is the PM Kisan annual grant amount for small farmers?",
            knowledge_pack=self.knowledge_pack,
        )

        self.assertEqual(response["outcome"], "failed")
        self.assertEqual(response["model_calls"], 2)
        self.assertTrue(response["escalated"])
        self.assertEqual(response["answer"], get_safe_refusal("en"))

    def test_provider_error_recorded_in_trace_when_all_fail(self) -> None:
        """Fix 7: Provider error message is recorded in trace when all providers fail."""
        def failing_transport(*args: Any, **kwargs: Any) -> tuple[int, Any, str, float]:
            return 500, None, "Upstream 500 Internal Server Error", 0.1

        pm = ProviderManager(
            provider_order=["nvidia"],
            cache=self.cache,
            transport=failing_transport,
            rate_limiter=self.no_wait_limiter,
        )
        harness = GroundingHarness(provider_manager=pm, min_bm25_score=0.5)

        response = harness.ask(
            question="What is the PM Kisan benefit for small farmers?",
            knowledge_pack=self.knowledge_pack,
        )

        self.assertEqual(response["outcome"], "failed")
        self.assertIsNotNone(response["trace"]["provider_error"])
        self.assertIn("Upstream 500", response["trace"]["provider_error"])

    def test_replay_only_checks_all_providers_and_hits_fallback_provider(self) -> None:
        """Fix 2: REPLAY_ONLY checks ALL providers first; hits fallback provider without network."""
        replay_cache = DiskCache(cache_dir=self.temp_dir, replay_only=True)
        messages = [{"role": "user", "content": "hi"}]
        params = {"temperature": 0.0, "seed": 42, "max_tokens": 1024}

        # Cache is populated for 'openrouter', but NOT for 'nvidia'
        replay_cache.set(
            provider="openrouter",
            model="model-tier-1",
            messages=messages,
            params=params,
            completion_data={
                "text": "cached response from openrouter",
                "provider": "openrouter",
                "model": "model-tier-1",
                "latency": 0.0,
                "finish_reason": "stop",
            },
        )

        def no_network_transport(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("Network should not be called in REPLAY_ONLY mode!")

        pm = ProviderManager(
            provider_order=["nvidia", "openrouter"],
            cache=replay_cache,
            transport=no_network_transport,
            rate_limiter=self.no_wait_limiter,
        )

        res = pm.call_model(model_id="model-tier-1", messages=messages)
        self.assertEqual(res.provider, "openrouter")
        self.assertEqual(res.text, "cached response from openrouter")

    def test_harness_does_not_swallow_cache_miss_error(self) -> None:
        """Fix 2: Harness must propagate CacheMissError directly when in REPLAY_ONLY mode."""
        empty_replay_cache = DiskCache(cache_dir=self.temp_dir, replay_only=True)
        pm = ProviderManager(
            provider_order=["nvidia"],
            cache=empty_replay_cache,
            transport=lambda *args: (200, {}, None, 0.0),
            rate_limiter=self.no_wait_limiter,
        )
        harness = GroundingHarness(provider_manager=pm, min_bm25_score=0.5)

        with self.assertRaises(CacheMissError):
            harness.ask(
                question="What is the PM Kisan annual grant?",
                knowledge_pack=self.knowledge_pack,
            )

    def test_missing_model_raises_clear_value_error(self) -> None:
        """Fix 3: get_model_id raises clear ValueError if neither specific nor general model is set."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                get_model_id("tier1", "nvidia")
            self.assertIn("No model configured for tier 'tier1'", str(ctx.exception))

    def test_429_retry_with_exponential_backoff(self) -> None:
        """Fix 5: HTTP 429 retries with exponential backoff before succeeding."""
        attempts = 0
        sleep_durations: list[float] = []

        def rate_limited_transport(url: str, payload: dict[str, Any], api_key: str, timeout: int) -> tuple[int, Any, str, float]:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return 429, None, "Too Many Requests", 0.05
            body = {"choices": [{"finish_reason": "stop", "message": {"content": "Success"}}]}
            return 200, body, None, 0.05

        def mock_sleep(seconds: float) -> None:
            sleep_durations.append(seconds)

        pm = ProviderManager(
            provider_order=["nvidia"],
            cache=self.cache,
            transport=rate_limited_transport,
            rate_limiter=self.no_wait_limiter,
        )

        with patch("time.sleep", side_effect=mock_sleep):
            res = pm.call_model(model_id="test-model", messages=[{"role": "user", "content": "hi"}])

        self.assertEqual(res.text, "Success")
        self.assertEqual(attempts, 2)
        self.assertIn(1.0, sleep_durations)

    def test_timeout_falls_through_to_next_provider(self) -> None:
        """Fix 5: Timeout on first provider falls through to next provider in chain."""
        attempted_providers: list[str] = []

        def timeout_transport(url: str, payload: dict[str, Any], api_key: str, timeout: int) -> tuple[int, Any, str, float]:
            if "integrate.api.nvidia.com" in url:
                attempted_providers.append("nvidia")
                return 0, None, "Timed out waiting for response", 45.0
            attempted_providers.append("openrouter")
            body = {"choices": [{"finish_reason": "stop", "message": {"content": "OpenRouter Success"}}]}
            return 200, body, None, 0.5

        pm = ProviderManager(
            provider_order=["nvidia", "openrouter"],
            cache=self.cache,
            transport=timeout_transport,
            rate_limiter=self.no_wait_limiter,
        )

        res = pm.call_model(model_id="test-model", messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(res.provider, "openrouter")
        self.assertEqual(res.text, "OpenRouter Success")
        self.assertEqual(attempted_providers, ["nvidia", "openrouter"])

    def test_reasoning_effort_400_retry_without_extra_body(self) -> None:
        """Fix 6: If provider returns 400 with reasoning_effort, retries once without extra_body."""
        payloads_seen: list[dict[str, Any]] = []

        def mock_transport(url: str, payload: dict[str, Any], api_key: str, timeout: int) -> tuple[int, Any, str, float]:
            payloads_seen.append(dict(payload))
            if "extra_body" in payload:
                return 400, None, "Unrecognized field extra_body", 0.05
            body = {"choices": [{"finish_reason": "stop", "message": {"content": "Retried Success"}}]}
            return 200, body, None, 0.05

        with patch.dict(os.environ, {"REASONING_EFFORT": "medium"}):
            pm = ProviderManager(
                provider_order=["nvidia"],
                cache=self.cache,
                transport=mock_transport,
                rate_limiter=self.no_wait_limiter,
            )
            res = pm.call_model(model_id="test-model", messages=[{"role": "user", "content": "hi"}])

        self.assertEqual(res.text, "Retried Success")
        self.assertEqual(len(payloads_seen), 2)
        self.assertIn("extra_body", payloads_seen[0])
        self.assertNotIn("extra_body", payloads_seen[1])


if __name__ == "__main__":
    unittest.main()
