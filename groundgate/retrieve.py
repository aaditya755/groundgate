"""BM25 information retrieval over multilingual passages with word tokens and character 3-grams.

Why this module exists:
Traditional tokenizers split Marathi and Hindi on boundaries that miss inflected forms
(e.g., 'शेतकऱ्यांना' vs 'शेतकरी'). Combining word tokens with character 3-grams allows BM25
to match morphological variants within a language without requiring an external lemmatizer.
Because character n-grams cannot cross language boundaries, retrieval filters to the question's
detected language.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from groundgate.normalize import detect_language, tokenize


def char_ngrams(text: str, n: int = 3) -> list[str]:
    """Generate character n-grams from text after stripping whitespace and punctuation.

    Why: In Devanagari, root words frequently carry agglutinative case markers (vibhakti).
    Character 3-grams capture the shared root across grammatical inflections.
    """
    cleaned = "".join(ch for ch in text.lower() if not ch.isspace() and ch.isalnum() or 0x0900 <= ord(ch) <= 0x097F)
    if len(cleaned) < n:
        return [cleaned] if cleaned else []
    return [cleaned[i : i + n] for i in range(len(cleaned) - n + 1)]


def extract_features(text: str, lang: str | None = None) -> list[str]:
    """Extract both word content tokens and character 3-grams for indexing and querying.

    Why: Hybrid representation gives high exact-word precision while character 3-grams
    provide robust recall for morphological variants in Hindi and Marathi.
    """
    words = tokenize(text, remove_stopwords=True, lang=lang)
    ngrams = char_ngrams(text, n=3)
    return words + ngrams


class BM25Index:
    """Pure-Python BM25 implementation supporting multilingual word and n-gram indexing.

    Why pure Python:
    Eliminates C/Rust compilation issues and external package dependencies for hackathon
    portability while maintaining sub-millisecond ranking speed on small knowledge packs.
    """

    def __init__(
        self,
        passages: Sequence[Mapping[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.passages = list(passages)
        self.doc_len: list[int] = []
        self.doc_term_freqs: list[dict[str, int]] = []
        self.doc_freqs: dict[str, int] = {}
        self.num_docs = len(self.passages)

        total_len = 0
        for p in self.passages:
            text = f"{p.get('title', '')} {p.get('text', '')}"
            lang = p.get("lang")
            terms = extract_features(text, lang=lang)
            length = len(terms)
            self.doc_len.append(length)
            total_len += length

            tf: dict[str, int] = {}
            for term in terms:
                tf[term] = tf.get(term, 0) + 1
            self.doc_term_freqs.append(tf)

            for term in tf:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.avg_doc_len = (total_len / self.num_docs) if self.num_docs > 0 else 0.0

    def score(self, query: str, lang: str | None = None) -> list[tuple[float, Mapping[str, Any]]]:
        """Compute BM25 relevance scores for all documents given a query string.

        Why IDF with +1 smoothing: Prevents negative IDF weights for terms that appear
        in more than half the corpus.
        """
        if self.num_docs == 0:
            return []

        query_terms = extract_features(query, lang=lang)
        if not query_terms:
            return [(0.0, p) for p in self.passages]

        scores: list[float] = [0.0] * self.num_docs

        for term in query_terms:
            df = self.doc_freqs.get(term, 0)
            if df == 0:
                continue

            # Robertson-Spärck Jones IDF formula with +1 floor
            idf = math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))

            for idx in range(self.num_docs):
                tf = self.doc_term_freqs[idx].get(term, 0)
                if tf == 0:
                    continue

                d_len = self.doc_len[idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (d_len / self.avg_doc_len))
                scores[idx] += idf * (tf * (self.k1 + 1.0) / denom)

        ranked = [(scores[i], self.passages[i]) for i in range(self.num_docs)]
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked


def retrieve_passages(
    query: str,
    passages: Sequence[Mapping[str, Any]],
    top_k: int = 3,
    min_score: float = 1.5,
    filter_language: bool = True,
) -> tuple[list[dict[str, Any]], float, bool]:
    """Retrieve top-k passages matching the query, filtered by the query's detected language.

    Returns:
    - retrieved: list of passages sorted by relevance
    - top_score: highest BM25 score achieved
    - precheck_passed: True if top_score >= min_score, False otherwise

    Why language filtering:
    Since BM25 matches tokens and character n-grams, cross-language matching will match
    noise or fail completely. Questions in Marathi only search Marathi passages.
    """
    if not passages:
        return [], 0.0, False

    detected_lang = detect_language(query)

    # Filter corpus to matching language if requested
    if filter_language:
        lang_passages = [p for p in passages if p.get("lang") == detected_lang]
        # If corpus has no passages in detected language, fallback to all passages
        candidate_passages = lang_passages if lang_passages else list(passages)
    else:
        candidate_passages = list(passages)

    index = BM25Index(candidate_passages)
    ranked = index.score(query, lang=detected_lang)

    if not ranked:
        return [], 0.0, False

    top_score = ranked[0][0]
    precheck_passed = top_score >= min_score

    top_results: list[dict[str, Any]] = []
    for score, passage in ranked[:top_k]:
        p_dict = dict(passage)
        p_dict["bm25_score"] = round(score, 3)
        top_results.append(p_dict)

    return top_results, round(top_score, 3), precheck_passed
