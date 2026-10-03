#!/usr/bin/env python3
"""Smoke test script for groundgate provider endpoints and multilingual JSON contracts.

Why this script exists:
Before relying on open-weight models (like NVIDIA Nemotron) for grounded QA,
we must verify:
1. Endpoint reachability and authentication (NVIDIA API and/or OpenRouter)
2. Model availability without guessing IDs
3. Marathi and Hindi instruction adherence: does the model obey the JSON schema and
   respond in the correct Devanagari script, or does it fall back to English?
This is the GO/NO-GO sanity check for Marathi and Hindi capabilities.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any

# Ensure project root is in sys.path so groundgate can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from groundgate.gate import parse_model_json
from groundgate.normalize import detect_language, detect_script
from groundgate.providers import PROVIDER_METADATA, get_model_id

# System prompt enforcing strict JSON contract
JSON_CONTRACT_SYSTEM_PROMPT = """You are a grounded question-answering assistant.
Answer ONLY from the provided facts. Do not assume or extrapolate.
You must return a single JSON object with exact keys:
{
  "answer": "string containing your concise answer in the same language and script as the question",
  "sources": ["list of passage IDs used to support your answer"],
  "refuse": false
}
If the context does not contain the answer, return:
{
  "answer": "I cannot answer this question based on the provided sources.",
  "sources": [],
  "refuse": true
}
Do not wrap your output in markdown fences. Return raw JSON only."""

# Smoke test questions with matching mock context passages (clearly labeled as FAKE TEST DATA)
SMOKE_TEST_CASES = [
    {
        "lang": "en",
        "language_name": "English",
        "question": "What is the annual financial support under the scheme?",
        "context": "[P1] FAKE TEST DATA - The Kisan Samman Scheme provides an annual financial grant of Rs 6,000 to eligible small farmers.",
        "expected_id": "P1",
    },
    {
        "lang": "hi",
        "language_name": "Hindi (हिन्दी)",
        "question": "योजना के तहत वार्षिक वित्तीय सहायता कितनी है?",
        "context": "[P1_HI] FAKE TEST DATA - किसान सम्मान योजना के तहत पात्र किसानों को प्रति वर्ष ₹6,000 की वित्तीय सहायता प्रदान की जाती है।",
        "expected_id": "P1_HI",
    },
    {
        "lang": "mr",
        "language_name": "Marathi (मराठी)",
        "question": "योजनेअंतर्गत वार्षिक आर्थिक सहाय्य किती दिले जाते?",
        "context": "[P1_MR] FAKE TEST DATA - किसान सन्मान योजनेअंतर्गत पात्र शेतकऱ्यांना दरवर्षी ६,००० रुपयांचे आर्थिक सहाय्य दिले जाते.",
        "expected_id": "P1_MR",
    },
]


def load_env_file() -> None:
    """Manually parse .env file if python-dotenv is not installed."""
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if not os.path.exists(env_path):
        return

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key not in os.environ:
                    os.environ[key] = val
    except Exception as exc:
        print(f"Warning: Could not read .env file: {exc}")


def http_post_json(
    url: str,
    payload: dict[str, Any],
    api_key: str,
    timeout: int = 45,
) -> tuple[int, dict[str, Any] | None, str | None, float]:
    """Perform HTTP POST request with Bearer authorization and time the roundtrip."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "groundgate-smoke/0.1.0",
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    start_time = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed = time.perf_counter() - start_time
            status_code = resp.getcode()
            body = resp.read().decode("utf-8")
            parsed = json.loads(body)
            return status_code, parsed, None, elapsed
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - start_time
        err_msg = exc.read().decode("utf-8", errors="replace")
        return exc.code, None, f"HTTP {exc.code}: {err_msg}", elapsed
    except urllib.error.URLError as exc:
        elapsed = time.perf_counter() - start_time
        return 0, None, f"Network error: {exc.reason}", elapsed
    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        return 0, None, f"Unexpected error: {exc}", elapsed


def http_get_json(
    url: str,
    api_key: str,
    timeout: int = 20,
) -> tuple[int, dict[str, Any] | None, str | None]:
    """Perform HTTP GET request to list models."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "groundgate-smoke/0.1.0",
    }
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            parsed = json.loads(body)
            return resp.getcode(), parsed, None
    except urllib.error.HTTPError as exc:
        err_msg = exc.read().decode("utf-8", errors="replace")
        return exc.code, None, f"HTTP {exc.code}: {err_msg}"
    except Exception as exc:
        return 0, None, f"Error: {exc}"


def check_models_for_provider(provider_key: str) -> None:
    """List available models from the provider endpoint."""
    config = PROVIDER_METADATA.get(provider_key)
    if not config:
        return

    api_key = os.environ.get(config["env_key"], "").strip()
    print(f"\n--- Checking Provider: {config['name']} ({provider_key}) ---")

    if not api_key:
        print(f"  [SKIPPED] Missing env var: {config['env_key']}. Set this in .env to test.")
        return

    models_url = f"{config['base_url']}/models"
    status, data, err = http_get_json(models_url, api_key)

    if status != 200 or not data:
        print(f"  [ERROR] Failed to fetch /v1/models: {err}")
        return

    model_entries = data.get("data", [])
    model_ids = [m.get("id", "") for m in model_entries if isinstance(m, dict)]
    print(f"  [SUCCESS] Endpoint returned {len(model_ids)} available models.")

    nemotron_models = [m for m in model_ids if "nemotron" in m.lower()]
    if nemotron_models:
        print("  Found Nemotron models:")
        for nm in nemotron_models[:10]:
            print(f"    - {nm}")
    else:
        print("  (No models with 'nemotron' in ID found. First 5 models in list:)")
        for m in model_ids[:5]:
            print(f"    - {m}")


def run_smoke_test_case(
    provider_key: str,
    model_id: str,
    tier_label: str,
    test_case: dict[str, str],
) -> dict[str, Any]:
    """Execute a single question against target model and verify output."""
    config = PROVIDER_METADATA[provider_key]
    api_key = os.environ.get(config["env_key"], "").strip()

    messages = [
        {"role": "system", "content": JSON_CONTRACT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{test_case['context']}\n\nQuestion: {test_case['question']}",
        },
    ]

    payload: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    reasoning_effort = os.environ.get("REASONING_EFFORT", "").strip()
    if reasoning_effort:
        payload["extra_body"] = {"reasoning_effort": reasoning_effort}

    endpoint_url = f"{config['base_url']}/chat/completions"
    status, resp_data, err, elapsed = http_post_json(endpoint_url, payload, api_key)

    # Retry once without extra_body if 400 occurred
    if status == 400 and "extra_body" in payload:
        payload_retry = dict(payload)
        payload_retry.pop("extra_body", None)
        status, resp_data, err, elapsed = http_post_json(endpoint_url, payload_retry, api_key)

    result_record = {
        "lang": test_case["lang"],
        "language_name": test_case["language_name"],
        "tier": tier_label,
        "provider": provider_key,
        "model": model_id,
        "latency_sec": round(elapsed, 2),
        "status_code": status,
        "finish_reason": "none",
        "json_valid": False,
        "script_ratio": 0.0,
        "detected_lang": "unknown",
        "answer_preview": "",
        "error": err,
    }

    if status != 200 or not resp_data:
        result_record["answer_preview"] = f"Error: {err}"
        return result_record

    choices = resp_data.get("choices", [])
    if not choices:
        result_record["answer_preview"] = "Empty choices array in response"
        return result_record

    first_choice = choices[0]
    result_record["finish_reason"] = first_choice.get("finish_reason", "unknown")

    raw_content = first_choice.get("message", {}).get("content", "")
    parsed_json, parse_err = parse_model_json(raw_content)

    if parsed_json and "answer" in parsed_json:
        result_record["json_valid"] = True
        answer_text = str(parsed_json.get("answer", ""))
        result_record["answer_preview"] = answer_text[:60] + ("..." if len(answer_text) > 60 else "")
        result_record["script_ratio"] = round(detect_script(answer_text), 3)
        result_record["detected_lang"] = detect_language(answer_text)
    else:
        result_record["json_valid"] = False
        result_record["answer_preview"] = f"Non-JSON or parse error: {parse_err}. Raw: {raw_content[:40]}"

    return result_record


def main() -> None:
    """Run full smoke test across providers, tiers, and en/hi/mr questions."""
    print("=" * 70)
    print("  groundgate Smoke Test & Marathi/Hindi Capability Check")
    print("=" * 70)

    load_env_file()

    raw_providers = os.environ.get("PROVIDER_ORDER", "nvidia,openrouter")
    providers = [p.strip().lower() for p in raw_providers.split(",") if p.strip()]

    print(f"Configured PROVIDER_ORDER : {providers}")

    for p in providers:
        if p in PROVIDER_METADATA:
            check_models_for_provider(p)

    active_providers = [
        p for p in providers
        if p in PROVIDER_METADATA and os.environ.get(PROVIDER_METADATA[p]["env_key"], "").strip()
    ]

    if not active_providers:
        print("\n" + "!" * 70)
        print("NOTICE: No valid API keys found in environment or .env.")
        print("To run live smoke calls against NVIDIA or OpenRouter:")
        print("  1. Copy .env.example to .env")
        print("  2. Set NVIDIA_API_KEY or OPENROUTER_API_KEY")
        print("  3. Set TIER1_MODEL / TIER2_MODEL (or per-provider models)")
        print("  4. Re-run: python scripts/smoke.py")
        print("!" * 70)
        return

    primary_provider = active_providers[0]
    print(f"\nRunning multilingual test cases via: {primary_provider.upper()}...")

    models_to_test: list[tuple[str, str]] = []
    try:
        t1_model = get_model_id("tier1", primary_provider)
        models_to_test.append(("Tier-1", t1_model))
    except ValueError as e:
        print(f"Warning: {e}")

    try:
        t2_model = get_model_id("tier2", primary_provider)
        if not models_to_test or t2_model != models_to_test[0][1]:
            models_to_test.append(("Tier-2", t2_model))
    except ValueError as e:
        print(f"Warning: {e}")

    if not models_to_test:
        print("ERROR: No valid models configured for primary provider.")
        return

    results: list[dict[str, Any]] = []

    for tier_label, model_id in models_to_test:
        print(f"\nEvaluating {tier_label} ({model_id}):")
        for test_case in SMOKE_TEST_CASES:
            print(f"  Testing {test_case['language_name']}...", end=" ", flush=True)
            res = run_smoke_test_case(primary_provider, model_id, tier_label, test_case)
            results.append(res)
            print(
                f"Done ({res['latency_sec']}s, finish={res['finish_reason']}, "
                f"JSON={'OK' if res['json_valid'] else 'FAIL'}, Lang={res['detected_lang']})"
            )

    print("\n" + "=" * 105)
    print(f"{'Lang':<8} {'Tier':<8} {'Latency':<8} {'Status':<10} {'Finish':<10} {'JSON?':<6} {'Script%':<8} {'Detected':<8} {'Preview'}")
    print("-" * 105)
    for r in results:
        status_str = f"HTTP {r['status_code']}" if r["status_code"] else "NET_ERR"
        json_str = "PASS" if r["json_valid"] else "FAIL"
        script_pct = f"{r['script_ratio']:.0%}"
        print(
            f"{r['lang']:<8} {r['tier']:<8} {r['latency_sec']:<6}s {status_str:<10} "
            f"{r['finish_reason']:<10} {json_str:<6} {script_pct:<8} {r['detected_lang']:<8} {r['answer_preview'][:25]}"
        )
    print("=" * 105)

    marathi_results = [r for r in results if r["lang"] == "mr"]
    if marathi_results:
        marathi_json_ok = all(r["json_valid"] for r in marathi_results)
        marathi_script_ok = all(r["script_ratio"] >= 0.40 for r in marathi_results)
        print("\nMarathi GO/NO-GO Assessment:")
        print(f"  JSON contract followed  : {'PASS' if marathi_json_ok else 'FAIL'}")
        print(f"  Devanagari script kept  : {'PASS' if marathi_script_ok else 'FAIL'}")
        if marathi_json_ok and marathi_script_ok:
            print("  VERDICT: GO - Model produces valid JSON and retains Marathi Devanagari script.")
        else:
            print("  VERDICT: NO-GO / INVESTIGATE - Model struggled with Marathi formatting or language retention.")


if __name__ == "__main__":
    main()
