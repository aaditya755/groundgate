"""Devanagari and multilingual text normalization, tokenization, and number extraction.

Why this module exists:
Devanagari script (used for Hindi and Marathi) has Unicode idiosyncrasies such as
zero-width joiners/non-joiners, Devanagari numeral glyphs (०-९), and danda punctuation (।, ॥).
Deterministic grounding checks require comparing numbers and tokens across generated answers
and source passages regardless of glyph variants or formatting differences.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

# Devanagari digits to ASCII digits translation table
# Why: Both Hindi and Marathi frequently mix Devanagari numerals (०-९) with ASCII numerals (0-9).
# Translating all digits to ASCII before extraction guarantees uniform number matching.
DEVANAGARI_DIGITS: Final[str] = "०१२३४५६७८९"
ASCII_DIGITS: Final[str] = "0123456789"
DIGIT_TRANSLATION_TABLE: Final[dict[int, int]] = str.maketrans(DEVANAGARI_DIGITS, ASCII_DIGITS)

# Zero-width characters and BOM markers to strip
# Why: LLMs and web scrapers often insert invisible Unicode joiners (ZWJ, ZWNJ, BOM)
# which cause byte-level string equality and regex boundaries to fail silently.
ZERO_WIDTH_CHARS: Final[re.Pattern[str]] = re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad]")

# Danda and double danda punctuation used as sentence terminators in Devanagari
# Why: Replacing dandas with whitespace/standard punctuation prevents fusing sentences into one token.
DANDA_PATTERN: Final[re.Pattern[str]] = re.compile(r"[\u0964\u0965]")

# Minimal stopword sets for English, Hindi, and Marathi
# Why: Token overlap in gate.py should measure substantive content tokens,
# not generic auxiliary verbs, postpositions, or conjunctions.
STOPWORDS_EN: Final[frozenset[str]] = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "as", "at", "by",
    "for", "in", "of", "on", "to", "with", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "this", "that",
    "these", "those", "it", "its", "from", "into", "about", "after", "before",
    "can", "could", "will", "would", "should", "not", "no", "only", "than",
})

STOPWORDS_HI: Final[frozenset[str]] = frozenset({
    "का", "के", "की", "है", "हैं", "था", "थे", "थी", "में", "से", "को", "पर",
    "ने", "और", "या", "यह", "वह", "इस", "उस", "जो", "तो", "ही", "भी", "नहीं",
    "तक", "द्वारा", "लिए", "साथ", "एक", "अपने", "अपनी", "अपने", "कर", "करते",
    "करता", "करती", "किया", "हो", "हुए", "हुआ", "हुई", "सकते", "सकता", "सकती",
    "दिया", "दिए", "दी", "गया", "गए", "गई", "इत्यादि", "आदि",
})

STOPWORDS_MR: Final[frozenset[str]] = frozenset({
    "आहे", "आहेत", "होता", "होते", "होती", "मध्ये", "चा", "ची", "चे", "च्या",
    "आणि", "किंवा", "हा", "ही", "हे", "या", "ती", "ते", "त्या", "नाही", "पण",
    "तर", "मग", "जर", "मुळे", "साठी", "कडून", "वर", "खाली", "पर्यंत", "केले",
    "केला", "केली", "करून", "एक", "स्वतः", "असून", "असल्यास", "होणारे", "होणारा",
    "दिले", "दिला", "दिली", "गेले", "गेला", "गेली", "इत्यादि",
})


def strip_zero_width(text: str) -> str:
    """Remove zero-width spaces, joiners, non-joiners, and BOM markers.

    Why: Invisible formatting characters break string comparison and tokenization.
    """
    if not text:
        return ""
    return ZERO_WIDTH_CHARS.sub("", text)


def devanagari_to_ascii_digits(text: str) -> str:
    """Translate Devanagari numeral glyphs (०-९) to ASCII digits (0-9).

    Why: LLMs may output numbers in either script; translating to ASCII allows
    uniform mathematical and factual consistency checks.
    """
    if not text:
        return ""
    return text.translate(DIGIT_TRANSLATION_TABLE)


def normalize_text(text: str) -> str:
    """Normalize text with Unicode NFC, stripped zero-width chars, and ASCII digits.

    Why: Unicode NFC composes separate base characters and combining matras into canonical
    precomposed forms, preventing false mismatches in Devanagari text.
    """
    if not text:
        return ""
    # NFC canonical composition
    normalized = unicodedata.normalize("NFC", text)
    # Strip invisible zero-width noise
    normalized = strip_zero_width(normalized)
    # Convert Devanagari digits to ASCII for uniform parsing
    normalized = devanagari_to_ascii_digits(normalized)
    return normalized


def tokenize(text: str, remove_stopwords: bool = True, lang: str | None = None) -> list[str]:
    """Tokenize multilingual text, preserving complete Devanagari and Latin words.

    Splits on punctuation (including danda । and ॥) and whitespace.
    Removes commas between digits (e.g. '6,000' -> '6000') before tokenizing so
    formatted numerals remain single tokens.

    Why: Standard regex word boundaries (\\b) do not work properly across Devanagari matras
    and combining characters. Matching character ranges [\\u0900-\\u097f]+ keeps Devanagari words intact.
    """
    if not text:
        return []

    norm = normalize_text(text)
    # Normalize dandas to spaces before regex matching
    norm = DANDA_PATTERN.sub(" ", norm)

    # Remove commas between digits so '6,000' becomes '6000' and doesn't split into '6' and '000'
    norm = re.sub(r"(?<=\d),(?=\d)", "", norm)

    # Word pattern: sequence of Devanagari characters or ASCII letters/digits
    # \\u0900-\\u097f covers all Devanagari vowels, consonants, signs, virama, and nukta
    word_pattern = re.compile(r"[\u0900-\u097f]+|[a-zA-Z0-9]+", re.UNICODE)
    raw_tokens = word_pattern.findall(norm)

    tokens: list[str] = []
    # Determine which stopword sets apply
    stopwords: set[str] = set()
    if remove_stopwords:
        if lang == "en":
            stopwords = set(STOPWORDS_EN)
        elif lang == "hi":
            stopwords = set(STOPWORDS_HI)
        elif lang == "mr":
            stopwords = set(STOPWORDS_MR)
        else:
            # Combine all stopwords if language is unknown
            stopwords = set(STOPWORDS_EN) | set(STOPWORDS_HI) | set(STOPWORDS_MR)

    for tok in raw_tokens:
        clean_tok = tok.lower()
        if remove_stopwords and clean_tok in stopwords:
            continue
        tokens.append(clean_tok)

    return tokens


def extract_numbers(text: str) -> list[str]:
    """Extract factual numbers (integers, decimals, percentages, Indian/Western commas).

    Ignores ordered-list item markers strictly at the start of a line (using re.MULTILINE with ^\\s*)
    so sentences like 'is 6000. It' keep 6000.

    Why: In grounded QA, numbers (grants, percentages, year, quota) carry critical facts.
    Devanagari numbers are normalized to ASCII, formatting commas are removed, and canonical
    numerical strings are returned.
    """
    if not text:
        return []

    norm = normalize_text(text)

    # Step 1: Strip out ordered list markers ONLY at the start of a line.
    # Matches patterns like:
    # ^\s*1.  or  ^\s*1)  or  ^\s*(1)  or  ^\s*1 -
    # Why: This prevents sentence-internal numbers like 'is 6000. It' from being stripped.
    list_marker_pattern = re.compile(
        r"^\s*(?:\(?\d+[\.\)\-:]|\d+\.)\s+",
        re.MULTILINE,
    )
    cleaned_for_numbers = list_marker_pattern.sub(" ", norm)

    # Step 2: Extract candidate numbers:
    # Supports:
    # - Indian comma formatting: 1,00,000
    # - Western comma formatting: 100,000
    # - Decimals: 3.14, 0.5
    # - Percentages: 50%, 12.5%
    # - Plain integers: 5000
    number_pattern = re.compile(
        r"(?<![a-zA-Z0-9_])(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d+)?%?(?![a-zA-Z0-9_])"
    )

    matches = number_pattern.findall(cleaned_for_numbers)
    extracted: list[str] = []

    for match in matches:
        # Strip commas to obtain canonical number representation (e.g. "1,00,000" -> "100000")
        canonical = match.replace(",", "").strip()
        if canonical:
            extracted.append(canonical)

    return extracted


def detect_script(text: str) -> float:
    """Return the Devanagari character ratio in the input text (between 0.0 and 1.0).

    Why: Used by gate.py to ensure the model responds in the same script family
    as the user's question (preventing unprompted language switches).
    """
    if not text:
        return 0.0

    devanagari_count = 0
    total_alphabetic = 0

    for char in text:
        # Check if character is in Devanagari block U+0900 - U+097F
        code = ord(char)
        if 0x0900 <= code <= 0x097F:
            devanagari_count += 1
            total_alphabetic += 1
        elif char.isalpha():
            total_alphabetic += 1

    if total_alphabetic == 0:
        return 0.0

    return devanagari_count / total_alphabetic


def detect_language(text: str) -> str:
    """Detect whether text is English ('en'), Marathi ('mr'), or Hindi ('hi').

    Why: Retrieval and stopword filtering need language tagging. If script is Latin,
    it classifies as 'en'. If Devanagari, it uses language-specific lexical markers.
    """
    ratio = detect_script(text)
    if ratio < 0.25:
        return "en"

    tokens = set(tokenize(text, remove_stopwords=False))

    # Distinct Marathi marker words (interrogatives, auxiliaries, case markers, verbs)
    marathi_markers = {
        "आहे", "आहेत", "होता", "होते", "होती", "मध्ये", "च्या", "ची", "चे", "चा",
        "आणि", "किंवा", "नाही", "मुळे", "साठी", "कडून", "केले", "केला", "केली",
        "करून", "दिले", "दिला", "दिली", "जाते", "जातात", "येते", "येतात",
        "मिळते", "मिळतात", "किती", "कसे", "केव्हा", "कुठे", "कोण", "काय",
        "शेतकऱ्यांना", "शेतकरी", "रुपये", "योजनेअंतर्गत", "बाबत", "विषयी", "सहाय्य",
    }

    # Distinct Hindi marker words (interrogatives, auxiliaries, postpositions, verbs)
    hindi_markers = {
        "है", "हैं", "था", "थे", "थी", "में", "का", "के", "की",
        "और", "या", "नहीं", "लिए", "साथ", "किया", "गया", "गए", "गई",
        "दिया", "दिए", "दी", "जाता", "जाती", "जाते", "मिलता", "मिलती", "मिलते",
        "कितना", "कितनी", "कितने", "कैसे", "कहाँ", "कब", "क्या", "सहायता",
    }

    marathi_hits = len(tokens & marathi_markers)
    hindi_hits = len(tokens & hindi_markers)

    if marathi_hits > hindi_hits:
        return "mr"
    if hindi_hits > marathi_hits:
        return "hi"

    # Default to Hindi if script is Devanagari but markers are tied or sparse
    return "hi"
