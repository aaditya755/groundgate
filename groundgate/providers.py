"""OpenAI-compatible LLM provider client with fallback chains, retries, and rate limiting.

Why this module exists:
Free endpoints (NVIDIA API Catalog and OpenRouter) suffer from periodic outages,
severe concurrency caps (429 Too Many Requests), and occasional truncation (finish_reason='length').
This provider orchestrates multi-provider fallbacks, exponential backoff, rate limiting,
length-truncation retries, diagnostic error snippets, and disk caching so that downstream
code receives reliable completions.
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


def _extract_body_snippet(resp_data: Any, raw_text: str | None = None) -> str:
    """Extract up to 200 characters from the response body or its 'error' field.

    Why this helper exists:
    Provides diagnostic context when API providers fail, without leaking
    authorization headers, secret keys, or exceeding log limits.
    """
    if isinstance(resp_data, dict):
        if "error" in resp_data:
            err = resp_data["error"]
            if isinstance(err, dict):
                text = err.get("message") or json.dumps(err)
            else:
                text = str(err)
        else:
            text = json.dumps(resp_data)
        return str(text)[:200]
    if isinstance(resp_data, str):
        return resp_data[:200]
    if raw_text:
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict) and "error" in parsed:
                err = parsed["error"]
                text = err.get("message") if isinstance(err, dict) else str(err)
                return str(text)[:200]
        except Exception:
            pass
        return str(raw_text)[:200]
    return ""


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
            raw_body = exc.read().decode("utf-8", errors="replace").strip()
            snippet = _extract_body_snippet(None, raw_text=raw_body)
            err_msg = f"HTTP {exc.code}: {snippet}" if snippet else f"HTTP {exc.code}"
            return exc.code, None, err_msg, elapsed
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

        Why this design:
        1. Checks cache for max_tokens (default 1024) across all providers first.
        2. When a provider returns finish_reason == 'length', retries the SAME provider once
           with max_tokens multiplied by 4 (capped at 8192) before falling through.
        3. Each attempt is cached under its own max_tokens key so previous entries remain valid.
        4. When calls fail, the HTTP status and first 200 chars of response body (or error field)
           are included in diagnostics, without headers or authorization keys.
        """
        if messages is None:
            messages = []

        params = {
            "temperature": temperature,
            "seed": seed,
            "max_tokens": max_tokens,
        }

        # Step 1: Check ALL providers' cache entries first for initial max_tokens
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

            # When a provider returns finish_reason == 'length', retry the SAME provider once
            # with max_tokens * 4 (capped at 8192) before falling through to the next provider.
            current_max_tokens = max_tokens
            length_retry_attempted = False

            while True:
                current_params = {
                    "temperature": temperature,
                    "seed": seed,
                    "max_tokens": current_max_tokens,
                }

                # Check cache for this specific attempt on this provider
                cached_data = self.cache.get(
                    provider, prov_model, messages, current_params, raise_on_miss=False
                )
                if cached_data and "completion" in cached_data:
                    comp = cached_data["completion"]
                    return CompletionResult(
                        text=comp["text"],
                        provider=comp.get("provider", provider),
                        model=comp.get("model", prov_model),
                        latency=comp.get("latency", 0.0),
                        finish_reason=comp.get("finish_reason", "cached"),
                    )

                payload: dict[str, Any] = {
                    "model": prov_model,
                    "messages": messages,
                    "temperature": temperature,
                    "seed": seed,
                    "max_tokens": current_max_tokens,
                }
                if self.reasoning_effort:
                    payload["extra_body"] = {"reasoning_effort": self.reasoning_effort}

                max_retries = 2
                backoff = 1.0
                retry_for_length = False

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
                        snippet = _extract_body_snippet(resp_data, raw_text=err_msg)
                        if snippet:
                            detail = f"HTTP {status_code}: {snippet}"
                        elif err_msg:
                            detail = f"HTTP {status_code}: {err_msg[:200]}"
                        else:
                            detail = f"HTTP {status_code}"
                        provider_errors.append(f"{provider}: {detail}")
                        break

                    choices = resp_data.get("choices", [])
                    if not choices:
                        snippet = _extract_body_snippet(resp_data)
                        provider_errors.append(
                            f"{provider}: HTTP {status_code}: Empty choices array: {snippet}"
                        )
                        break

                    first_choice = choices[0]
                    finish_reason = first_choice.get("finish_reason", "unknown")
                    msg_content = first_choice.get("message", {}).get("content")

                    if msg_content is None or not str(msg_content).strip():
                        snippet = _extract_body_snippet(resp_data)
                        provider_errors.append(
                            f"{provider}: HTTP {status_code}: Received empty content from model: {snippet}"
                        )
                        break

                    if finish_reason == "length":
                        new_max_tokens = min(current_max_tokens * 4, 8192)
                        if not length_retry_attempted and new_max_tokens > current_max_tokens:
                            length_retry_attempted = True
                            current_max_tokens = new_max_tokens
                            retry_for_length = True
                            break  # Break attempt loop to re-enter while loop with 4x max_tokens
                        else:
                            snippet = _extract_body_snippet(resp_data)
                            provider_errors.append(
                                f"{provider}: HTTP {status_code}: Response truncated by token length limit (max_tokens={current_max_tokens}): {snippet}"
                            )
                            break

                    # Valid completion
                    result = CompletionResult(
                        text=str(msg_content).strip(),
                        provider=provider,
                        model=prov_model,
                        latency=round(elapsed, 2),
                        finish_reason=finish_reason,
                    )

                    self.cache.set(provider, prov_model, messages, current_params, result.to_dict())
                    return result

                # If retry_for_length is True, repeat while loop for the same provider
                if retry_for_length:
                    continue

                # Otherwise, this provider has exhausted its attempts; fall through to next provider
                break

        raise ProviderError(
            f"All providers in chain {self.provider_order} failed. Details: {'; '.join(provider_errors)}"
        )
