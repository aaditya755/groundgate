"""FastAPI server for groundgate: a small JSON API plus the single-page UI.

Why this file exists:
It wraps the already-tested GroundingHarness so a browser page can ask questions,
load saved example runs and show benchmark results. It also protects the free
model tiers with simple public-demo limits.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# why: lets "import groundgate" work no matter where the server is started from
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# why: the harness reads API keys and model IDs from environment variables
load_dotenv(REPO_ROOT / ".env")

from groundgate.harness import GroundingHarness  # noqa: E402
from groundgate.providers import ProviderManager, RateLimiter  # noqa: E402

MAX_QUESTION_LENGTH = 500
PER_IP_LIMIT_PER_MINUTE = 10
GLOBAL_DAILY_LIMIT = 100

STATIC_DIR = REPO_ROOT / "app" / "static"
TRACES_FILE = STATIC_DIR / "example_traces.json"
RESULTS_FILE = REPO_ROOT / "eval" / "results" / "results.json"
PACK_FILE = REPO_ROOT / "data" / "knowledge_pack.jsonl"

FALLBACK_NOTICE = "Live call failed. Showing a pre-recorded example, not a live answer."

CAVEATS = [
    "The ungrounded-answer rate for the gate setup is circular: it is measured with the same deterministic checks the gate uses.",
    "With 30 questions the differences between setups are anecdotal, not statistically significant.",
    "The knowledge pack is placeholder text with fictional schemes and amounts.",
]

# ---------------------------------------------------------------------------
# Public-demo limits
# ---------------------------------------------------------------------------
_limit_lock = threading.Lock()
_ip_hits: dict[str, list[float]] = defaultdict(list)
_daily = {"day": time.strftime("%Y-%m-%d"), "count": 0}


def check_limits(ip: str) -> None:
    """Enforce a per-IP minute limit and a global daily cap.

    Why: free model tiers allow very few requests per day, so one noisy visitor
    must not use up the whole demo. The per-IP check runs first so rejected
    requests do not eat into the daily cap.
    """
    now = time.time()
    with _limit_lock:
        today = time.strftime("%Y-%m-%d")
        if _daily["day"] != today:
            _daily["day"] = today
            _daily["count"] = 0

        recent = [t for t in _ip_hits[ip] if t > now - 60.0]
        if len(recent) >= PER_IP_LIMIT_PER_MINUTE:
            _ip_hits[ip] = recent
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit: at most {PER_IP_LIMIT_PER_MINUTE} questions per minute per visitor.",
            )
        if _daily["count"] >= GLOBAL_DAILY_LIMIT:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"The public demo reached its daily limit of {GLOBAL_DAILY_LIMIT} questions. "
                    "Try the pre-recorded examples, or run the project locally."
                ),
            )
        recent.append(now)
        _ip_hits[ip] = recent
        _daily["count"] += 1


def client_ip(request: Request) -> str:
    """Return the caller's IP address.

    Why: behind a proxy every request shares the proxy's address, so the
    X-Forwarded-For header is only trusted when TRUST_PROXY is set.
    """
    if os.environ.get("TRUST_PROXY", "").strip().lower() in {"1", "true", "yes"}:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ---------------------------------------------------------------------------
# Data loading and harness setup
# ---------------------------------------------------------------------------
def load_pack() -> list[dict[str, Any]]:
    """Load data/knowledge_pack.jsonl (UTF-8, one JSON object per line)."""
    if not PACK_FILE.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(PACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            try:
                rows.append(json.loads(text))
            except ValueError:
                continue
    return rows


def read_traces() -> list[dict[str, Any]]:
    """Return the pre-recorded example runs, or an empty list if none exist."""
    if not TRACES_FILE.exists():
        return []
    try:
        with open(TRACES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def normalize_question(text: str) -> str:
    return " ".join(text.split()).lower()


KNOWLEDGE_PACK = load_pack()
RPM = int(os.environ.get("REQUESTS_PER_MINUTE", "20"))
# why: one shared rate limiter keeps all visitors inside the provider's per-minute limit
provider_mgr = ProviderManager(rate_limiter=RateLimiter(requests_per_minute=RPM))
harness = GroundingHarness(provider_manager=provider_mgr)

app = FastAPI(title="groundgate", version="0.1.0")


class AskRequest(BaseModel):
    question: str


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict[str, Any]:
    """Liveness probe. Never returns API keys."""
    return {
        "status": "healthy",
        "knowledge_pack_passages": len(KNOWLEDGE_PACK),
        "configured_providers": provider_mgr.provider_order,
        "requests_per_minute_cap": RPM,
    }


@app.get("/api/example-traces")
def example_traces() -> list[dict[str, Any]]:
    return read_traces()


@app.get("/api/results")
def results() -> dict[str, Any]:
    """Return eval/results/results.json plus the honesty caveats."""
    if not RESULTS_FILE.exists():
        return {
            "status": "not_found",
            "message": "Benchmark results have not been generated yet. Run: python eval/run_eval.py",
            "caveats": CAVEATS,
        }
    try:
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"status": "ok", "data": data, "caveats": CAVEATS}
    except Exception as exc:
        return {"status": "error", "message": f"Could not read results.json: {exc}", "caveats": CAVEATS}


@app.post("/api/ask")
def ask(body: AskRequest, request: Request) -> dict[str, Any]:
    """Run one question through the grounding harness."""
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(question) > MAX_QUESTION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Question is longer than {MAX_QUESTION_LENGTH} characters.",
        )

    check_limits(client_ip(request))

    try:
        response = harness.ask(question=question, knowledge_pack=KNOWLEDGE_PACK)
        response["is_cached_example"] = False
        return response
    except Exception as exc:
        # why: never show an answer that belongs to a different question
        print(f"[groundgate] harness error: {type(exc).__name__}: {str(exc)[:200]}")
        wanted = normalize_question(question)
        for example in read_traces():
            if normalize_question(str(example.get("question", ""))) == wanted:
                return {
                    "answer": example.get("answer", ""),
                    "sources": example.get("sources", []),
                    "outcome": example.get("outcome", "failed"),
                    "escalated": example.get("escalated", False),
                    "model_calls": example.get("model_calls", 0),
                    "trace": example.get("trace", {}),
                    "is_cached_example": True,
                    "notice": FALLBACK_NOTICE,
                }
        raise HTTPException(
            status_code=502,
            detail="The live model call failed and no pre-recorded example matches this question. Please try again.",
        )


# ---------------------------------------------------------------------------
# Static UI
# ---------------------------------------------------------------------------
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    page = STATIC_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="app/static/index.html not found.")
    return FileResponse(page)


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    print(f"groundgate server: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)