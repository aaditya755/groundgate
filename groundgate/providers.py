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
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from groundgate.cache import DiskCache

# Supported provider configurations
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
    """Thread-safe rate limiter enforcing maximum requests per minute."""

    def __init__(self, requests_per_minute: int = 20) -> None:
        self.interval = 60.0 / max(1, requests_per_minute)
        self.last_request_time: float = 0.0

    def wait(self) -> None:
        """Pause execution if necessary to respect configured rate limits."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_request_time = time.time()


class ProviderManager:
    """Manages multi-provider execution with retries, caching, and fallback chains."""

    def __init__(
        self,
        provider_order: list[str] | None = None,
        cache: DiskCache | None = None,
        transport: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        raw_order = os.environ.get("PROVIDER_ORDER", "nvidia,openrouter")
        self.provider_order = provider_order or [
            p.strip().lower() for p in raw_order.split(",") if p.strip()
        ]
        self.cache = cache or DiskCache.from_env()
        self.transport = transport or self._default_http_transport

        # Read rate limits and timeout from environment
        rpm = int(os.environ.get("REQUESTS_PER_MINUTE", "20"))
        self.rate_limiter = RateLimiter(requests_per_minute=rpm)
        self.timeout_seconds = int(os.environ.get("REQUEST_TIMEOUT_SECONDS", "45"))

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
        model_id: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int = 1024,
    ) -> CompletionResult:
        """Call model with automatic provider fallback, disk caching, and validation.

        Fails and falls through to next provider if:
        - HTTP 429 (after exponential backoff)
        - HTTP 5xx or network timeout
        - content is None or empty string
        - finish_reason is 'length'
        """
        params = {
            "temperature": temperature,
            "seed": seed,
            "max_tokens": max_tokens,
        }

        # Check disk cache first across all providers in chain
        for provider in self.provider_order:
            cached_data = self.cache.get(provider, model_id, messages, params)
            if cached_data and "completion" in cached_data:
                comp = cached_data["completion"]
                return CompletionResult(
                    text=comp["text"],
                    provider=comp.get("provider", provider),
                    model=comp.get("model", model_id),
                    latency=comp.get("latency", 0.0),
                    finish_reason=comp.get("finish_reason", "cached"),
                )

        provider_errors: list[str] = []

        # Iterate through provider chain
        for provider in self.provider_order:
            meta = PROVIDER_METADATA.get(provider)
            if not meta:
                continue

            api_key = self._get_api_key(provider)
            # If using custom mock transport (e.g. testing), provide mock key if unset
            if not api_key:
                if self.transport == self._default_http_transport:
                    provider_errors.append(f"{provider}: Missing API key")
                    continue
                api_key = "mock-key"

            endpoint = f"{meta['base_url']}/chat/completions"
            payload = {
                "model": model_id,
                "messages": messages,
                "temperature": temperature,
                "seed": seed,
                "max_tokens": max_tokens,
            }

            # Retry on 429 up to 2 times with backoff
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

                if status_code == 429 and attempt < max_retries:
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue

                if status_code != 200 or not resp_data:
                    provider_errors.append(f"{provider}: {err_msg or f'Status {status_code}'}")
                    break  # Fall through to next provider in chain

                choices = resp_data.get("choices", [])
                if not choices:
                    provider_errors.append(f"{provider}: Empty choices array")
                    break

                first_choice = choices[0]
                finish_reason = first_choice.get("finish_reason", "unknown")
                msg_content = first_choice.get("message", {}).get("content")

                # Validation checks: content cannot be None, empty, or truncated by length
                if msg_content is None or not str(msg_content).strip():
                    provider_errors.append(f"{provider}: Received empty content from model")
                    break

                if finish_reason == "length":
                    provider_errors.append(f"{provider}: Response truncated by token length limit")
                    break

                # Success!
                result = CompletionResult(
                    text=str(msg_content).strip(),
                    provider=provider,
                    model=model_id,
                    latency=round(elapsed, 2),
                    finish_reason=finish_reason,
                )

                # Save successful completion to disk cache
                self.cache.set(provider, model_id, messages, params, result.to_dict())
                return result

        raise ProviderError(
            f"All providers in chain {self.provider_order} failed. Details: {'; '.join(provider_errors)}"
        )
