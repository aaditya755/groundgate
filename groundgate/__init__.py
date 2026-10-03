"""groundgate: Devanagari-aware deterministic grounding harness for open-weight LLMs.

This package provides deterministic, code-based grounding verification and
single-escalation cascades for Hindi, Marathi, and English.
"""

from __future__ import annotations

from groundgate.normalize import (
    normalize_text,
    devanagari_to_ascii_digits,
    strip_zero_width,
    tokenize,
    extract_numbers,
    detect_script,
    detect_language,
)
from groundgate.gate import (
    GateResult,
    verify_grounding,
)
from groundgate.retrieve import (
    BM25Index,
    retrieve_passages,
    char_ngrams,
    extract_features,
)
from groundgate.contract import (
    SYSTEM_PROMPT,
    format_context,
    build_messages,
    build_escalation_messages,
)
from groundgate.cache import (
    DiskCache,
    CacheMissError,
)
from groundgate.providers import (
    CompletionResult,
    ProviderManager,
    ProviderError,
)
from groundgate.trace import (
    ExecutionTrace,
    AttemptRecord,
)
from groundgate.harness import (
    GroundingHarness,
    SAFE_REFUSALS,
    get_safe_refusal,
)

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "normalize_text",
    "devanagari_to_ascii_digits",
    "strip_zero_width",
    "tokenize",
    "extract_numbers",
    "detect_script",
    "detect_language",
    "GateResult",
    "verify_grounding",
    "BM25Index",
    "retrieve_passages",
    "char_ngrams",
    "extract_features",
    "SYSTEM_PROMPT",
    "format_context",
    "build_messages",
    "build_escalation_messages",
    "DiskCache",
    "CacheMissError",
    "CompletionResult",
    "ProviderManager",
    "ProviderError",
    "ExecutionTrace",
    "AttemptRecord",
    "GroundingHarness",
    "SAFE_REFUSALS",
    "get_safe_refusal",
]
