"""Disk cache for model completions with SHA-256 key hashing and replay-only mode.

Why this module exists:
Free endpoints (OpenRouter :free and NVIDIA API Catalog) enforce strict rate limits
(e.g., 20 req/min, 50 req/day). Caching completion outputs on disk avoids exhausting quota
during testing and evaluation. Replay-only mode allows reproducible offline test runs
without making live network calls.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


class CacheMissError(Exception):
    """Raised when REPLAY_ONLY mode is active and a query is not found in cache."""


class DiskCache:
    """Persistent JSON disk cache keyed by deterministic SHA-256 hashes."""

    def __init__(self, cache_dir: str | Path = "cache", replay_only: bool = False) -> None:
        self.cache_dir = Path(cache_dir)
        self.replay_only = replay_only
        self.disabled = False

        # If cache directory cannot be created, gracefully disable cache rather than crashing
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            self.disabled = True

    @classmethod
    def from_env(cls, cache_dir: str | Path = "cache") -> DiskCache:
        """Instantiate cache reading REPLAY_ONLY configuration from environment."""
        env_val = os.environ.get("REPLAY_ONLY", "false").strip().lower()
        replay_only = env_val in {"1", "true", "yes"}
        return cls(cache_dir=cache_dir, replay_only=replay_only)

    def _compute_key(
        self,
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
        params: dict[str, Any],
    ) -> str:
        """Generate deterministic SHA-256 hex digest for request parameters."""
        payload = {
            "provider": provider.strip().lower(),
            "model": model.strip(),
            "messages": messages,
            "params": params,
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get(
        self,
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
        params: dict[str, Any],
        raise_on_miss: bool = False,
    ) -> dict[str, Any] | None:
        """Retrieve cached completion record if present.

        If raise_on_miss=True and replay_only=True, raises CacheMissError.
        """
        if self.disabled:
            if self.replay_only and raise_on_miss:
                raise CacheMissError("DiskCache is disabled and REPLAY_ONLY is active.")
            return None

        key = self._compute_key(provider, model, messages, params)
        file_path = self.cache_dir / f"{key}.json"

        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                # If cached file is corrupted, treat as miss
                return None

        if self.replay_only and raise_on_miss:
            raise CacheMissError(
                f"REPLAY_ONLY is enabled and cache key {key} ({provider}/{model}) was not found on disk."
            )

        return None

    def set(
        self,
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
        params: dict[str, Any],
        completion_data: dict[str, Any],
    ) -> None:
        """Write completion record to disk cache; silently continues on any write errors."""
        if self.disabled:
            return

        try:
            key = self._compute_key(provider, model, messages, params)
            file_path = self.cache_dir / f"{key}.json"

            record = {
                "key": key,
                "provider": provider,
                "model": model,
                "messages": messages,
                "params": params,
                "completion": completion_data,
            }

            temp_path = self.cache_dir / f"{key}.tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
            temp_path.replace(file_path)
        except Exception:
            # Continue silently on write failures (disk full, permissions, etc.)
            pass

    def clear(self) -> None:
        """Clear all cached entries."""
        if self.disabled:
            return
        for file in self.cache_dir.glob("*.json"):
            try:
                file.unlink(missing_ok=True)
            except Exception:
                pass
