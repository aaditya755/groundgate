"""BM25 information retrieval over multilingual passages with word tokens and character 3-grams.

Why this module exists:
Traditional tokenizers split Marathi and Hindi on boundaries that miss inflected forms
(e.g., 'शेतकऱ्यांना' vs 'शेतकरी'). Combining word tokens with character 3-grams allows BM25
to match morphological variants within a language without requiring an external lemmatizer.
Because character n-grams cannot cross language boundaries, retrieval filters to candidate languages.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from groundgate.normalize import candidate_languages, char_ngrams, tokenize


def extract_features(text: str, lang: str | None = None) -> list[str]:
    """Extract both word content tokens and character 3-grams per word (skipping stopwords).

    Why per-word n-grams without stopwords:
    Prevents cross-word boundary artifacts and stopword contamination in n-gram indexing.
    """
    words = tokenize(text, remove_stopwords=True, lang=lang)
    ngrams = char_ngrams(text, n=3, lang=lang)
    return words + ngrams


class BM25Index:
    """Pure-Python BM25 implementation supporting multilingual word and n-gram indexing."""

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
        """Compute BM25 relevance scores for all documents given a query string."""
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
    """Retrieve top-k passages matching query across candidate languages.

    Returns:
    - retrieved: list of passages sorted by relevance
    - top_score: highest BM25 score achieved
    - precheck_passed: True if top_score >= min_score, False otherwise

    Why union of candidate languages:
    Short Devanagari questions may not have distinct Hindi or Marathi marker words.
    Searching the union of candidate languages allows the highest-scoring passage
    in the corpus to resolve the language ambiguity dynamically.
    """
    if not passages:
        return [], 0.0, False

    candidates = candidate_languages(query)

    if filter_language:
        candidate_passages = [p for p in passages if p.get("lang") in candidates]
        if not candidate_passages:
            candidate_passages = list(passages)
    else:
        candidate_passages = list(passages)

    # Use primary candidate for stopword filtering in query feature extraction
    primary_lang = candidates[0] if len(candidates) == 1 else None
    index = BM25Index(candidate_passages)
    ranked = index.score(query, lang=primary_lang)

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
