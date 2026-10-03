"""Unit tests for Devanagari and multilingual text normalization, tokenization, and number extraction.

Why these tests exist:
Devanagari scripts and multilingual QA have distinct edge cases (combining signs, zero-width
joiners, non-ASCII digits, danda sentence ends). These tests guarantee that normalization
and extraction remain rock-solid across English, Hindi, and Marathi.
"""

from __future__ import annotations

import unittest

from groundgate.normalize import (
    candidate_languages,
    char_ngrams,
    detect_language,
    detect_script,
    devanagari_to_ascii_digits,
    extract_numbers,
    normalize_text,
    strip_zero_width,
    tokenize,
)


class TestNormalize(unittest.TestCase):
    """Test suite for text normalization and Devanagari transformations."""

    def test_devanagari_to_ascii_digits(self) -> None:
        """Devanagari digits ०-९ must translate correctly to ASCII 0-9."""
        devanagari_str = "०१२३४५६७८९"
        expected = "0123456789"
        self.assertEqual(devanagari_to_ascii_digits(devanagari_str), expected)

        mixed = "अनुदान: ₹६,००० किंवा ₹५०० दरमहा (२०२४)"
        expected_mixed = "अनुदान: ₹6,000 किंवा ₹500 दरमहा (2024)"
        self.assertEqual(devanagari_to_ascii_digits(mixed), expected_mixed)

    def test_strip_zero_width(self) -> None:
        """Zero-width spaces, joiners, non-joiners, and BOM markers must be stripped."""
        text_with_zw = "मराठी\u200b\u200cभाषा\u200d\ufeff"
        cleaned = strip_zero_width(text_with_zw)
        self.assertEqual(cleaned, "मराठीभाषा")

    def test_normalize_text(self) -> None:
        """normalize_text combines NFC composition, zero-width stripping, and digit mapping."""
        sample = "शेतकरी\u200b अनुदान: ₹६,०००"
        normalized = normalize_text(sample)
        self.assertEqual(normalized, "शेतकरी अनुदान: ₹6,000")

    def test_tokenize_devanagari_words(self) -> None:
        """Tokenizer must preserve complete Devanagari words and split on dandas (। and ॥)."""
        sentence = "शेतकऱ्यांना आर्थिक सहाय्य मिळते। ते समाधानी आहेत॥"
        raw_tokens = tokenize(sentence, remove_stopwords=False)
        self.assertIn("शेतकऱ्यांना", raw_tokens)
        self.assertIn("आर्थिक", raw_tokens)
        self.assertIn("सहाय्य", raw_tokens)
        self.assertNotIn("।", raw_tokens)
        self.assertNotIn("॥", raw_tokens)

        filtered_tokens = tokenize(sentence, remove_stopwords=True, lang="mr")
        self.assertIn("शेतकऱ्यांना", filtered_tokens)
        self.assertIn("सहाय्य", filtered_tokens)
        self.assertNotIn("आहेत", filtered_tokens)

    def test_tokenize_multilingual(self) -> None:
        """Tokenizer handles mixed English and Devanagari terms cleanly."""
        text = "PM-Kisan योजनेअंतर्गत 6000 rupees benefit."
        tokens = tokenize(text, remove_stopwords=False)
        self.assertIn("pm", tokens)
        self.assertIn("kisan", tokens)
        self.assertIn("योजनेअंतर्गत", tokens)
        self.assertIn("6000", tokens)
        self.assertIn("benefit", tokens)

    def test_tokenize_digit_comma_removal(self) -> None:
        """Commas between digits are removed before tokenizing so '6,000' becomes '6000'."""
        text = "The grant amount is 6,000 rupees and 1,50,000 max."
        tokens = tokenize(text, remove_stopwords=False)
        self.assertIn("6000", tokens)
        self.assertIn("150000", tokens)
        self.assertNotIn("000", tokens)

    def test_char_ngrams_per_word_skips_stopwords(self) -> None:
        """char_ngrams builds n-grams per word (not across spaces) and skips stopwords."""
        # 'आणि' is a Marathi stopword
        phrase = "शेतकरी आणि अनुदान"
        ngrams = char_ngrams(phrase, n=3, lang="mr")

        # Must have n-grams from content words
        self.assertIn("शेत", ngrams)
        self.assertIn("करी", ngrams)
        self.assertIn("अनु", ngrams)
        self.assertIn("दान", ngrams)

        # Must NOT have ngrams from stopword 'आणि'
        self.assertNotIn("आणि", ngrams)

        # Must NOT have cross-word boundary n-grams (e.g. ending of word 1 with start of word 2)
        cross_word = "रीआ"  # last char of शेतकरी + first of आणि
        self.assertNotIn(cross_word, ngrams)

    def test_extract_numbers_various_formats(self) -> None:
        """extract_numbers extracts Indian commas, decimals, percentages, and ASCII numbers."""
        text = "Under scheme 12, financial support is 1,50,000 with 10.5% interest and 50% subsidy."
        numbers = extract_numbers(text)
        self.assertIn("12", numbers)
        self.assertIn("150000", numbers)
        self.assertIn("10.5%", numbers)
        self.assertIn("50%", numbers)

    def test_extract_numbers_devanagari(self) -> None:
        """extract_numbers converts Devanagari numerals to canonical ASCII figures."""
        text = "शेतकऱ्यांना ₹६,००० किंवा ₹५०,००० पर्यंत ७५% अनुदान मिळते."
        numbers = extract_numbers(text)
        self.assertIn("6000", numbers)
        self.assertIn("50000", numbers)
        self.assertIn("75%", numbers)

    def test_extract_numbers_start_of_line_only_list_markers(self) -> None:
        """Strip list markers only at line start; keep mid-sentence numbers."""
        mid_sentence = "The total grant is 6000. It is distributed in installments."
        numbers_mid = extract_numbers(mid_sentence)
        self.assertIn("6000", numbers_mid)

        multiline = (
            "1. First benefit is 5000 rupees.\n"
            "2) Second benefit covers 40%.\n"
            "   3. Third benefit is 2026."
        )
        numbers_multi = extract_numbers(multiline)
        self.assertIn("5000", numbers_multi)
        self.assertIn("40%", numbers_multi)
        self.assertIn("2026", numbers_multi)
        self.assertNotIn("1", numbers_multi)
        self.assertNotIn("2", numbers_multi)
        self.assertNotIn("3", numbers_multi)

    def test_detect_script(self) -> None:
        """detect_script computes Devanagari character ratio accurately."""
        english = "This is a purely English sentence with no Devanagari characters."
        self.assertEqual(detect_script(english), 0.0)

        marathi = "हे संपूर्ण वाक्य मराठी देवनागरी लिपीमध्ये लिहिलेले आहे."
        ratio = detect_script(marathi)
        self.assertGreater(ratio, 0.75)

        empty = ""
        self.assertEqual(detect_script(empty), 0.0)

    def test_candidate_languages_routing(self) -> None:
        """candidate_languages routes English, distinct Marathi/Hindi, and ambiguous queries."""
        # English
        self.assertEqual(candidate_languages("What is the annual subsidy?"), ["en"])

        # Marathi with distinct markers (किती, दिले, जाते, काय, कसे)
        mr_text = "योजनेअंतर्गत शेतकऱ्यांना किती अनुदान दिले जाते?"
        self.assertEqual(candidate_languages(mr_text), ["mr"])

        # Hindi with distinct markers (कितनी, क्या, कैसे, जाती)
        hi_text = "योजना के तहत किसानों को कितनी सहायता दी जाती है?"
        self.assertEqual(candidate_languages(hi_text), ["hi"])

        # Ambiguous Devanagari text without marker words
        ambiguous = "पीएम किसान योजना माहिती"
        candidates = candidate_languages(ambiguous)
        self.assertIn("hi", candidates)
        self.assertIn("mr", candidates)


if __name__ == "__main__":
    unittest.main()
