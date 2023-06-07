import logging
from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, List, Tuple

from ..core_tools.data_types import DocResult, Document, Span
from ..core_tools.registry import TaggerRegistry
from ..core_tools.taggers import BaseTagger

REQUIRED_ENGLISH_WORDS = {"the", "be", "to", "of", "and", "that", "have", "with"}
SYMBOLS = {"#", "\u2026"}
BULLET_POINTS = {"*", "-"}


@dataclass
class GopherAttributes:
    fraction_of_characters_in_most_common_ngram: List[Tuple[int, float]]
    fraction_of_characters_in_duplicate_ngrams: List[Tuple[int, float]]
    character_count: int = 0
    word_count: int = 0
    median_word_length: float = False
    symbol_to_word_ratio: float = 0.0
    fraction_of_words_with_alpha_character: float = 0.0
    required_word_count: int = 0
    fraction_of_lines_starting_with_bullet_point: float = 0.0
    fraction_of_lines_ending_with_ellipsis: float = 0.0
    fraction_of_duplicate_lines: float = 0.0
    fraction_of_characters_in_duplicate_lines: float = 0.0

    def as_spans(self) -> List[Span]:
        spans = []
        spans.extend(
            [
                Span(0, self.character_count, f"fraction_of_characters_in_most_common_{n}grams", v)
                for n, v in self.fraction_of_characters_in_most_common_ngram
            ]
        )
        spans.extend(
            [
                Span(0, self.character_count, f"fraction_of_characters_in_duplicate_{n}grams", v)
                for n, v in self.fraction_of_characters_in_duplicate_ngrams
            ]
        )
        spans.append(Span(0, self.character_count, type="character_count", score=self.character_count))
        spans.append(Span(0, self.character_count, type="word_count", score=self.word_count))
        spans.append(Span(0, self.character_count, type="median_word_length", score=self.median_word_length))
        spans.append(Span(0, self.character_count, type="symbol_to_word_ratio", score=self.symbol_to_word_ratio))
        spans.append(
            Span(
                0,
                self.character_count,
                type="fraction_of_words_with_alpha_character",
                score=self.fraction_of_words_with_alpha_character,
            )
        )
        spans.append(Span(0, self.character_count, type="required_word_count", score=self.required_word_count))
        spans.append(
            Span(
                0,
                self.character_count,
                type="fraction_of_lines_starting_with_bullet_point",
                score=self.fraction_of_lines_starting_with_bullet_point,
            )
        )
        spans.append(
            Span(
                0,
                self.character_count,
                type="fraction_of_lines_ending_with_ellipsis",
                score=self.fraction_of_lines_ending_with_ellipsis,
            )
        )
        spans.append(
            Span(
                0, self.character_count, type="fraction_of_duplicate_lines", score=self.fraction_of_duplicate_lines
            )
        )
        spans.append(
            Span(
                0,
                self.character_count,
                type="fraction_of_characters_in_duplicate_lines",
                score=self.fraction_of_characters_in_duplicate_lines,
            )
        )
        return spans


def get_attributes(text: str) -> GopherAttributes:
    attrs = GopherAttributes([], [])
    attrs.character_count = len(text)
    if attrs.character_count == 0:
        return attrs

    try:
        words = text.split()
        word_count = len(words)
        character_count = sum(len(word) for word in words)

        attrs.word_count = word_count
        attrs.median_word_length = median([len(word) for word in words])
        attrs.symbol_to_word_ratio = sum(1 for word in words if any(s in word for s in SYMBOLS)) / word_count
        attrs.fraction_of_words_with_alpha_character = (
            sum(1 for word in words if any(c.isalpha() for c in word)) / word_count
        )
        attrs.required_word_count = sum(1 for word in words if word in REQUIRED_ENGLISH_WORDS)

        for n in range(2, 5):
            value = 0.0
            ngrams = find_ngrams(words, n)
            if ngrams:
                ngram_counts = occurrence_counts(ngrams)
                most_common_ngram, count = max(ngram_counts.items(), key=lambda x: x[1])
                value = sum(len(s) for s in most_common_ngram) / character_count
            attrs.fraction_of_characters_in_most_common_ngram.append((n, value))

        for n in range(5, 11):
            value = 0.0
            ngrams = find_ngrams(words, n)
            if ngrams:
                ngram_counts = occurrence_counts(ngrams)
                # Over-count the characters in the same way for the denominator and numerator
                ng_char_count = sum(sum(len(w) for w in ng) for ng in ngrams)
                value = (
                    sum(sum(len(w) for w in ng) * count for ng, count in ngram_counts.items() if count > 1)
                    / ng_char_count
                )
            attrs.fraction_of_characters_in_duplicate_ngrams.append((n, value))

        lines = text.split("\n")
        line_count = len(lines)

        attrs.fraction_of_lines_starting_with_bullet_point = (
            sum(1 for line in lines if any(line.startswith(s) for s in BULLET_POINTS)) / line_count
        )
        attrs.fraction_of_lines_ending_with_ellipsis = sum(1 for line in lines if line.endswith("\u2026")) / len(
            lines
        )
        attrs.fraction_of_duplicate_lines = (
            sum(count for line, count in occurrence_counts(lines).items() if count > 1) / line_count
        )
        attrs.fraction__of_characters_in_duplicate_lines = (
            sum(len(line) * count for line, count in occurrence_counts(lines).items() if count > 1)
            / character_count
        )
    except Exception as e:
        logging.exception(f"Error processing text: {text[:200]}")

    return attrs


def occurrence_counts(objects: List[Any]) -> Dict[Any, int]:
    counts = defaultdict(int)
    for line in objects:
        counts[line] += 1
    return counts


def find_ngrams(words: List[str], n: int) -> List[Tuple[str, ...]]:
    return list(zip(*[words[i:] for i in range(n)]))


@TaggerRegistry.add("gopher_v1")
class GopherTagger(BaseTagger):
    def predict(self, doc: Document) -> DocResult:
        attrs = get_attributes(doc.text)
        result = DocResult(doc=doc, spans=attrs.as_spans())
        return result