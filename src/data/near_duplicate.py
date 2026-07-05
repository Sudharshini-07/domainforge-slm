"""Deterministic utilities for near-duplicate text detection."""

from __future__ import annotations

import re


DEFAULT_SHINGLE_SIZE = 5


def tokenize_for_deduplication(text: str) -> list[str]:
    """Normalize text into deterministic lowercase word tokens."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    return re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)


def create_word_shingles(
    text: str,
    shingle_size: int = DEFAULT_SHINGLE_SIZE,
) -> set[tuple[str, ...]]:
    """Create unique contiguous word shingles from text."""
    if shingle_size <= 0:
        raise ValueError("shingle_size must be greater than zero")

    tokens = tokenize_for_deduplication(text)

    if len(tokens) < shingle_size:
        return set()

    return {
        tuple(tokens[index : index + shingle_size])
        for index in range(len(tokens) - shingle_size + 1)
    }


def jaccard_similarity(
    left: set[tuple[str, ...]],
    right: set[tuple[str, ...]],
) -> float:
    """Return Jaccard similarity for two shingle sets."""
    if not left and not right:
        return 0.0

    union = left | right
    if not union:
        return 0.0

    return len(left & right) / len(union)


def text_similarity(
    left_text: str,
    right_text: str,
    shingle_size: int = DEFAULT_SHINGLE_SIZE,
) -> float:
    """Compute deterministic shingle-based Jaccard similarity."""
    left_shingles = create_word_shingles(left_text, shingle_size)
    right_shingles = create_word_shingles(right_text, shingle_size)

    return jaccard_similarity(left_shingles, right_shingles)


def is_near_duplicate(
    left_text: str,
    right_text: str,
    threshold: float = 0.85,
    shingle_size: int = DEFAULT_SHINGLE_SIZE,
) -> bool:
    """Return whether two texts meet the near-duplicate threshold."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0")

    return (
        text_similarity(
            left_text,
            right_text,
            shingle_size=shingle_size,
        )
        >= threshold
    )