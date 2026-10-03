"""Unit tests for disk caching, silent error handling, and REPLAY_ONLY mode.

Why these tests exist:
Reliable offline replay and local caching protect against API exhaustion during hackathon
benchmarking. These tests verify caching mechanics, fallback robustness, and directory failures.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from groundgate.cache import CacheMissError, DiskCache


class TestCache(unittest.TestCase):
    """Test suite for DiskCache read/write and replay-only enforcement."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.cache = DiskCache(cache_dir=self.temp_dir, replay_only=False)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cache_miss_returns_none(self) -> None:
        """Cache miss returns None when replay_only is False."""
        result = self.cache.get(
            provider="nvidia",
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            params={"temperature": 0.0},
        )
        self.assertIsNone(result)

    def test_cache_set_and_hit(self) -> None:
        """Writing to cache allows identical requests to hit without network."""
        messages = [{"role": "user", "content": "How much subsidy?"}]
        params = {"temperature": 0.0, "max_tokens": 1024}
        comp_data = {
            "text": '{"answer": "5000", "sources": ["P1"], "refuse": false}',
            "provider": "openrouter",
            "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
            "latency": 1.25,
            "finish_reason": "stop",
        }

        self.cache.set("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free", messages, params, comp_data)

        # Retrieve
        hit = self.cache.get("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free", messages, params)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit["completion"]["text"], comp_data["text"])
        self.assertEqual(hit["completion"]["latency"], 1.25)

    def test_cache_set_fails_silently_on_exception(self) -> None:
        """Fix 4: cache.set must continue silently without raising if writing fails."""
        messages = [{"role": "user", "content": "Query"}]
        params = {"temperature": 0.0}
        comp_data = {"text": "Answer"}

        with patch("builtins.open", side_effect=PermissionError("Read only disk")):
            # Must not raise an exception
            self.cache.set("nvidia", "model-x", messages, params, comp_data)

    def test_cache_disabled_if_directory_creation_fails(self) -> None:
        """Fix 4: If cache directory cannot be created, cache is disabled without crashing."""
        with patch.object(Path, "mkdir", side_effect=PermissionError("Permission denied")):
            unwritable_cache = DiskCache(cache_dir="/nonexistent/root/cache")
            self.assertTrue(unwritable_cache.disabled)
            # get and set must be safe no-ops
            self.assertIsNone(unwritable_cache.get("p", "m", [], {}))
            unwritable_cache.set("p", "m", [], {}, {})

    def test_replay_only_raises_cache_miss_error(self) -> None:
        """When replay_only=True and raise_on_miss=True, uncached requests raise CacheMissError."""
        replay_cache = DiskCache(cache_dir=self.temp_dir, replay_only=True)
        with self.assertRaises(CacheMissError):
            replay_cache.get(
                provider="nvidia",
                model="test-model",
                messages=[{"role": "user", "content": "Uncached prompt"}],
                params={"temperature": 0.0},
                raise_on_miss=True,
            )


if __name__ == "__main__":
    unittest.main()
