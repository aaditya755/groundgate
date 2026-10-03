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
DEVANAGARI_DIGITS: Final[str] = "०१२३४५६७८९"
ASCII_DIGITS: Final[str] = "0123456789"
DIGIT_TRANSLATION_TABLE: Final[dict[int, int]] = str.maketrans(DEVANAGARI_DIGITS, ASCII_DIGITS)

# Zero-width characters and BOM markers to strip
ZERO_WIDTH_CHARS: Final[re.Pattern[str]] = re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad]")

# Danda and double danda punctuation used as sentence terminators in Devanagari
DANDA_PATTERN: Final[re.Pattern[str]] = re.compile(r"[\u0964\u0965]")

# Minimal stopword sets for English, Hindi, and Marathi
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

# Distinct Marathi lexical markers (grammatical words, verbal forms, pronouns)
MARATHI_MARKERS: Final[frozenset[str]] = frozenset({
    "किती", "दिले", "जाते", "काय", "कसे", "आहे", "आहेत", "होता", "होते", "होती",
    "मध्ये", "च्या", "ची", "चे", "चा", "आणि", "किंवा", "नाही", "मुळे", "साठी",
    "कडून", "केले", "केला", "केली", "करून", "दिला", "दिली", "जातात", "येते",
    "येतात", "मिळते", "मिळतात", "केव्हा", "कुठे", "कोण", "शेतकऱ्यांना",
    "शेतकरी", "रुपये", "योजनेअंतर्गत", "बाबत", "विषयी", "अल्पभूधारक",
})

# Distinct Hindi lexical markers (grammatical words, verbal forms, pronouns)
HINDI_MARKERS: Final[frozenset[str]] = frozenset({
    "कितना", "कितनी", "कितने", "क्या", "कैसे", "जाती", "जाता", "जाते", "है",
    "हैं", "था", "थे", "थी", "में", "का", "के", "की", "और", "या", "नहीं",
    "लिए", "साथ", "किया", "गया", "गए", "गई", "दिया", "दिए", "दी", "मिलता",
    "मिलती", "मिलते", "कहाँ", "कब", "सकता", "सकती", "सकते",
})


def strip_zero_width(text: str) -> str:
    """Remove zero-width spaces, joiners, non-joiners, and BOM markers."""
    if not text:
        return ""
    return ZERO_WIDTH_CHARS.sub("", text)


def devanagari_to_ascii_digits(text: str) -> str:
    """Translate Devanagari numeral glyphs (०-९) to ASCII digits (0-9)."""
    if not text:
        return ""
    return text.translate(DIGIT_TRANSLATION_TABLE)


def normalize_text(text: str) -> str:
    """Normalize text with Unicode NFC, stripped zero-width chars, and ASCII digits."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFC", text)
    normalized = strip_zero_width(normalized)
    normalized = devanagari_to_ascii_digits(normalized)
    return normalized


def tokenize(text: str, remove_stopwords: bool = True, lang: str | None = None) -> list[str]:
    """Tokenize multilingual text, preserving complete Devanagari and Latin words.

    Splits on punctuation (including danda । and ॥) and whitespace.
    Removes commas between digits (e.g. '6,000' -> '6000') before tokenizing.
    """
    if not text:
        return []

    norm = normalize_text(text)
    norm = DANDA_PATTERN.sub(" ", norm)
    # Remove commas between digits so '6,000' becomes '6000'
    norm = re.sub(r"(?<=\d),(?=\d)", "", norm)

    word_pattern = re.compile(r"[\u0900-\u097f]+|[a-zA-Z0-9]+", re.UNICODE)
    raw_tokens = word_pattern.findall(norm)

    tokens: list[str] = []
    stopwords: set[str] = set()
    if remove_stopwords:
        if lang == "en":
            stopwords = set(STOPWORDS_EN)
        elif lang == "hi":
            stopwords = set(STOPWORDS_HI)
        elif lang == "mr":
            stopwords = set(STOPWORDS_MR)
        else:
            stopwords = set(STOPWORDS_EN) | set(STOPWORDS_HI) | set(STOPWORDS_MR)

    for tok in raw_tokens:
        clean_tok = tok.lower()
        if remove_stopwords and clean_tok in stopwords:
            continue
        tokens.append(clean_tok)

    return tokens


def char_ngrams(text: str, n: int = 3, lang: str | None = None) -> list[str]:
    """Build character n-grams per word (after normalize_text), skipping stopwords.

    Why per word:
    Generating n-grams across entire strings creates spurious boundary grams (e.g. word endings
    fused with next word starts) and indexes stopwords. Per-word grams over content words guarantee
    meaningful morphological root matching.
    """
    if not text:
        return []

    words = tokenize(text, remove_stopwords=True, lang=lang)
    ngrams: list[str] = []
    for word in words:
        clean_word = word.strip()
        if len(clean_word) < n:
            if clean_word:
                ngrams.append(clean_word)
        else:
            for i in range(len(clean_word) - n + 1):
                ngrams.append(clean_word[i : i + n])

    return ngrams


def extract_numbers(text: str) -> list[str]:
    """Extract factual numbers, ignoring ordered-list item markers strictly at line start."""
    if not text:
        return []

    norm = normalize_text(text)

    # Strip list markers only at start of line
    list_marker_pattern = re.compile(
        r"^\s*(?:\(?\d+[\.\)\-:]|\d+\.)\s+",
        re.MULTILINE,
    )
    cleaned_for_numbers = list_marker_pattern.sub(" ", norm)

    number_pattern = re.compile(
        r"(?<![a-zA-Z0-9_])(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d+)?%?(?![a-zA-Z0-9_])"
    )

    matches = number_pattern.findall(cleaned_for_numbers)
    extracted: list[str] = []

    for match in matches:
        canonical = match.replace(",", "").strip()
        if canonical:
            extracted.append(canonical)

    return extracted


def detect_script(text: str) -> float:
    """Return the Devanagari character ratio in the input text (between 0.0 and 1.0)."""
    if not text:
        return 0.0

    devanagari_count = 0
    total_alphabetic = 0

    for char in text:
        code = ord(char)
        if 0x0900 <= code <= 0x097F:
            devanagari_count += 1
            total_alphabetic += 1
        elif char.isalpha():
            total_alphabetic += 1

    if total_alphabetic == 0:
        return 0.0

    return devanagari_count / total_alphabetic


def candidate_languages(text: str) -> list[str]:
    """Identify candidate languages for a text: ['en'], ['hi'], ['mr'], or ['hi', 'mr'].

    Why candidate sets:
    Devanagari script is shared by both Hindi and Marathi. A short query lacking distinctive
    marker words (e.g., 'पीएम किसान योजना') cannot be definitively classified. Returning
    both ['hi', 'mr'] allows retrieval to search passages in both languages without premature
    filtering, deferring the final language choice to the top-ranked passage.
    """
    ratio = detect_script(text)
    if ratio < 0.25:
        return ["en"]

    tokens = set(tokenize(text, remove_stopwords=False))

    marathi_hits = len(tokens & MARATHI_MARKERS)
    hindi_hits = len(tokens & HINDI_MARKERS)

    if marathi_hits > hindi_hits:
        return ["mr"]
    if hindi_hits > marathi_hits:
        return ["hi"]

    # When marker words are tied or absent, both Devanagari languages are candidates
    return ["hi", "mr"]


def detect_language(text: str) -> str:
    """Detect single primary language for text ('en', 'mr', or 'hi').

    Defaults to 'hi' if Devanagari text is ambiguous.
    """
    candidates = candidate_languages(text)
    if len(candidates) == 1:
        return candidates[0]
    return "hi"
