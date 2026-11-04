"""Deterministic lexical evidence retrieval for FinQA."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*", re.IGNORECASE)

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "many",
    "much",
    "of",
    "on",
    "the",
    "that",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


@dataclass(frozen=True)
class EvidenceItem:
    """One complete source evidence unit selected for a question."""

    kind: str
    index: int
    content: str
    score: float
    token_count: int


@dataclass(frozen=True)
class EvidenceSelection:
    """Evidence selected under a strict token budget."""

    items: tuple[EvidenceItem, ...]
    budget: int
    used_tokens: int

    @property
    def remaining_tokens(self) -> int:
        return self.budget - self.used_tokens


@dataclass(frozen=True)
class _Candidate:
    kind: str
    index: int
    content: str
    document: tuple[str, ...]
    token_count: int
    source_order: int


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(
        token.lower()
        for token in _TOKEN_PATTERN.findall(text)
        if token.lower() not in _STOP_WORDS
    )


def _bm25_scores(
    query: str,
    documents: list[tuple[str, ...]],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Calculate BM25 scores without an external retrieval dependency."""
    query_terms = sorted(set(_tokenize(query)))

    if not query_terms or not documents:
        return [0.0] * len(documents)

    document_frequencies: Counter[str] = Counter()
    term_frequencies: list[Counter[str]] = []
    lengths: list[int] = []

    for document in documents:
        frequencies = Counter(document)
        term_frequencies.append(frequencies)
        lengths.append(len(document))
        document_frequencies.update(frequencies.keys())

    document_count = len(documents)
    average_length = sum(lengths) / document_count

    if average_length == 0:
        return [0.0] * document_count

    inverse_document_frequencies = {
        term: math.log(
            1.0
            + (document_count - document_frequencies[term] + 0.5)
            / (document_frequencies[term] + 0.5)
        )
        for term in query_terms
    }

    scores: list[float] = []

    for frequencies, length in zip(term_frequencies, lengths):
        score = 0.0
        length_normalization = k1 * (1.0 - b + b * length / average_length)

        for term in query_terms:
            frequency = frequencies[term]
            if frequency == 0:
                continue

            denominator = frequency + length_normalization
            score += inverse_document_frequencies[term] * (
                frequency * (k1 + 1.0) / denominator
            )

        scores.append(score)

    return scores


def select_evidence(
    question: str,
    passages: list[str],
    table: list[list[str]],
    *,
    token_budget: int,
    token_counter: Callable[[str], int],
) -> EvidenceSelection:
    """Select complete passages and header-grounded table rows.

    The evidence budget counts selected evidence content only. It excludes
    the question, prompt instructions, and generated target.

    Text indexes refer to ``passages``, which should be pre_text followed
    by post_text. Table indexes refer to original table row indexes.

    The table's first row is treated as its header. Every table candidate
    includes both the header and one data row, so no selected row loses
    its column context.

    This function deliberately accepts no dataset record or gold
    annotation argument.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    if not isinstance(token_budget, int) or isinstance(token_budget, bool):
        raise ValueError("token_budget must be an integer")

    if token_budget < 0:
        raise ValueError("token_budget must be non-negative")

    if not callable(token_counter):
        raise ValueError("token_counter must be callable")

    if not isinstance(passages, list) or any(
        not isinstance(passage, str) for passage in passages
    ):
        raise ValueError("passages must be a list of strings")

    if not isinstance(table, list) or any(
        not isinstance(row, list) or any(not isinstance(cell, str) for cell in row)
        for row in table
    ):
        raise ValueError("table must be a list of string rows")

    if table:
        header_width = len(table[0])
        if any(len(row) != header_width for row in table):
            raise ValueError("table rows must have a consistent width")

    candidates: list[_Candidate] = []
    source_order = 0

    for index, passage in enumerate(passages):
        if passage.strip():
            count = token_counter(passage)
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError("token_counter must return non-negative integers")

            candidates.append(
                _Candidate(
                    kind="text",
                    index=index,
                    content=passage,
                    document=_tokenize(passage),
                    token_count=count,
                    source_order=source_order,
                )
            )
            source_order += 1

    if table:
        header = " | ".join(table[0])

        for index, row in enumerate(table[1:], start=1):
            if not any(cell.strip() for cell in row):
                continue

            row_content = " | ".join(row)
            content = f"{header}\n{row_content}"

            count = token_counter(content)
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError("token_counter must return non-negative integers")

            candidates.append(
                _Candidate(
                    kind="table",
                    index=index,
                    content=content,
                    document=_tokenize(content),
                    token_count=count,
                    source_order=source_order,
                )
            )
            source_order += 1

    scores = _bm25_scores(
        question,
        [candidate.document for candidate in candidates],
    )

    ranked = sorted(
        zip(candidates, scores),
        key=lambda pair: (
            -pair[1],
            pair[0].source_order,
        ),
    )

    selected: list[EvidenceItem] = []
    used_tokens = 0

    for candidate, score in ranked:
        if score <= 0:
            continue

        if used_tokens + candidate.token_count > token_budget:
            continue

        selected.append(
            EvidenceItem(
                kind=candidate.kind,
                index=candidate.index,
                content=candidate.content,
                score=score,
                token_count=candidate.token_count,
            )
        )
        used_tokens += candidate.token_count

    return EvidenceSelection(
        items=tuple(selected),
        budget=token_budget,
        used_tokens=used_tokens,
    )
