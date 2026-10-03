"""OpenAI-compatible LLM provider client with fallback chains, retries, and rate limiting.

Why this module exists:
Free endpoints (NVIDIA API Catalog and OpenRouter) suffer from periodic outages,
severe concurrency caps (429 Too Many Requests), and occasional truncation (finish_reason='length').
This provider orchestrates multi-provider fallbacks, exponential backoff, rate limiting,
and disk caching so that downstream code receives reliable completions.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from groundgate.cache import CacheMissError, DiskCache

PROVIDER_METADATA: dict[str, dict[str, str]] = {
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "env_key": "NVIDIA_API_KEY",
        "name": "NVIDIA API Catalog",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "name": "OpenRouter",
    },
}


class ProviderError(Exception):
    """Raised when all configured providers in the chain fail."""


@dataclass(frozen=True)
class CompletionResult:
    """Standardized output from a model completion call."""

    text: str
    provider: str
    model: str
    latency: float
    finish_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "provider": self.provider,
            "model": self.model,
            "latency": self.latency,
            "finish_reason": self.finish_reason,
        }


class RateLimiter:
    """Thread-safe rate limiter enforcing maximum requests per minute with locking."""

    def __init__(self, requests_per_minute: int = 20) -> None:
        self.requests_per_minute = requests_per_minute
        self.interval = 60.0 / max(1, requests_per_minute) if requests_per_minute > 0 else 0.0
        self.last_request_time: float = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Pause execution if necessary to respect configured rate limits."""
        if self.requests_per_minute <= 0:
            return  # No-wait limiter for testing
        with self._lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_request_time = time.time()


def get_model_id(tier: str, provider: str) -> str:
    """Retrieve model ID for a specific tier and provider from environment.

    Checks:
    1. TIER{1|2}_MODEL_{PROVIDER} (e.g. TIER1_MODEL_NVIDIA)
    2. TIER{1|2}_MODEL (default fallback)
    Raises ValueError if no model ID is configured.
    """
    tier_clean = tier.lower().strip()
    tier_num = "1" if "1" in tier_clean else "2"
    prov_clean = provider.upper().strip()

    specific_env = f"TIER{tier_num}_MODEL_{prov_clean}"
    general_env = f"TIER{tier_num}_MODEL"

    model_id = os.environ.get(specific_env, "").strip() or os.environ.get(general_env, "").strip()
    if not model_id:
        raise ValueError(
            f"No model configured for tier '{tier}' on provider '{provider}'. "
            f"Set {specific_env} or {general_env} in environment."
        )
    return model_id


class ProviderManager:
    """Manages multi-provider execution with retries, caching, and fallback chains."""

    def __init__(
        self,
        provider_order: list[str] | None = None,
        cache: DiskCache | None = None,
        transport: Callable[..., dict[str, Any]] | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        raw_order = os.environ.get("PROVIDER_ORDER", "nvidia,openrouter")
        self.provider_order = provider_order or [
            p.strip().lower() for p in raw_order.split(",") if p.strip()
        ]
        self.cache = cache or DiskCache.from_env()
        self.transport = transport or self._default_http_transport

        if rate_limiter is not None:
            self.rate_limiter = rate_limiter
        else:
            rpm = int(os.environ.get("REQUESTS_PER_MINUTE", "20"))
            self.rate_limiter = RateLimiter(requests_per_minute=rpm)

        self.timeout_seconds = int(os.environ.get("REQUEST_TIMEOUT_SECONDS", "45"))
        self.reasoning_effort = os.environ.get("REASONING_EFFORT", "").strip()

    def _get_api_key(self, provider: str) -> str:
        """Fetch API key from environment for target provider."""
        meta = PROVIDER_METADATA.get(provider.lower(), {})
        env_key = meta.get("env_key", "")
        return os.environ.get(env_key, "").strip()

    def _default_http_transport(
        self,
        url: str,
        payload: dict[str, Any],
        api_key: str,
        timeout: int,
    ) -> tuple[int, dict[str, Any] | None, str | None, float]:
        """Execute HTTP request using standard library urllib."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "groundgate/0.1.0",
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                elapsed = time.perf_counter() - start
                body = resp.read().decode("utf-8")
                return resp.getcode(), json.loads(body), None, elapsed
        except urllib.error.HTTPError as exc:
            elapsed = time.perf_counter() - start
            err_msg = exc.read().decode("utf-8", errors="replace")
            return exc.code, None, f"HTTP {exc.code}: {err_msg}", elapsed
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return 0, None, str(exc), elapsed

    def call_model(
        self,
        model_id: str | None = None,
        tier: str = "tier1",
        messages: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int = 1024,
    ) -> CompletionResult:
        """Call model checking ALL provider cache entries first before executing calls.

        Fails and falls through to next provider if:
        - HTTP 429 (after exponential backoff)
        - HTTP 5xx or network timeout
        - content is None or empty string
        - finish_reason is 'length'
        """
        if messages is None:
            messages = []

        params = {
            "temperature": temperature,
            "seed": seed,
            "max_tokens": max_tokens,
        }

        # Step 1: Check ALL providers' cache entries first
        for provider in self.provider_order:
            prov_model = model_id or get_model_id(tier, provider)
            cached_data = self.cache.get(provider, prov_model, messages, params, raise_on_miss=False)
            if cached_data and "completion" in cached_data:
                comp = cached_data["completion"]
                return CompletionResult(
                    text=comp["text"],
                    provider=comp.get("provider", provider),
                    model=comp.get("model", prov_model),
                    latency=comp.get("latency", 0.0),
                    finish_reason=comp.get("finish_reason", "cached"),
                )

        # Step 2: If all providers missed and REPLAY_ONLY mode is active, raise CacheMissError
        if self.cache.replay_only:
            raise CacheMissError(
                f"REPLAY_ONLY is active and cache missed across all providers: {self.provider_order}"
            )

        provider_errors: list[str] = []

        # Step 3: Iterate through provider fallback chain
        for provider in self.provider_order:
            meta = PROVIDER_METADATA.get(provider)
            if not meta:
                continue

            api_key = self._get_api_key(provider)
            if not api_key:
                if self.transport == self._default_http_transport:
                    provider_errors.append(f"{provider}: Missing API key")
                    continue
                api_key = "mock-key"

            prov_model = model_id or get_model_id(tier, provider)
            endpoint = f"{meta['base_url']}/chat/completions"

            payload: dict[str, Any] = {
                "model": prov_model,
                "messages": messages,
                "temperature": temperature,
                "seed": seed,
                "max_tokens": max_tokens,
            }
            if self.reasoning_effort:
                payload["extra_body"] = {"reasoning_effort": self.reasoning_effort}

            max_retries = 2
            backoff = 1.0

            for attempt in range(max_retries + 1):
                self.rate_limiter.wait()
                status_code, resp_data, err_msg, elapsed = self.transport(
                    endpoint,
                    payload,
                    api_key,
                    self.timeout_seconds,
                )

                # If 400 Bad Request occurs and reasoning_effort was sent, retry once without it
                if status_code == 400 and "extra_body" in payload:
                    payload_without_extra = dict(payload)
                    payload_without_extra.pop("extra_body", None)
                    status_code, resp_data, err_msg, elapsed = self.transport(
                        endpoint,
                        payload_without_extra,
                        api_key,
                        self.timeout_seconds,
                    )

                if status_code == 429 and attempt < max_retries:
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue

                if status_code != 200 or not resp_data:
                    provider_errors.append(f"{provider}: {err_msg or f'Status {status_code}'}")
                    break  # Fall through to next provider

                choices = resp_data.get("choices", [])
                if not choices:
                    provider_errors.append(f"{provider}: Empty choices array")
                    break

                first_choice = choices[0]
                finish_reason = first_choice.get("finish_reason", "unknown")
                msg_content = first_choice.get("message", {}).get("content")

                if msg_content is None or not str(msg_content).strip():
                    provider_errors.append(f"{provider}: Received empty content from model")
                    break

                if finish_reason == "length":
                    provider_errors.append(f"{provider}: Response truncated by token length limit")
                    break

                # Valid completion
                result = CompletionResult(
                    text=str(msg_content).strip(),
                    provider=provider,
                    model=prov_model,
                    latency=round(elapsed, 2),
                    finish_reason=finish_reason,
                )

                self.cache.set(provider, prov_model, messages, params, result.to_dict())
                return result

        raise ProviderError(
            f"All providers in chain {self.provider_order} failed. Details: {'; '.join(provider_errors)}"
        )
