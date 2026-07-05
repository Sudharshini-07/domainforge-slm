from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

QUALITY_RULES_PATH = (
    PROJECT_ROOT
    / "configs"
    / "quality_rules.yaml"
)


def load_quality_rules() -> dict[str, Any]:
    """Load corpus quality rules."""
    with QUALITY_RULES_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def repeated_line_ratio(text: str) -> float:
    """
    Measure the fraction of non-empty lines that are
    repeated occurrences beyond their first appearance.
    """
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return 0.0

    counts = Counter(lines)

    repeated_occurrences = sum(
        count - 1
        for count in counts.values()
        if count > 1
    )

    return repeated_occurrences / len(lines)


def repeated_word_ratio(text: str) -> float:
    """
    Measure the dominance of the most frequent
    whitespace-delimited word.
    """
    words = [
        word.casefold()
        for word in text.split()
        if word.strip()
    ]

    if not words:
        return 0.0

    counts = Counter(words)
    highest_frequency = max(counts.values())

    return highest_frequency / len(words)


def non_alphanumeric_ratio(text: str) -> float:
    """
    Measure non-whitespace characters that are
    neither alphabetic nor numeric.
    """
    characters = [
        char
        for char in text
        if not char.isspace()
    ]

    if not characters:
        return 0.0

    non_alphanumeric = sum(
        1
        for char in characters
        if not char.isalnum()
    )

    return non_alphanumeric / len(characters)


def alphabetic_ratio(text: str) -> float:
    """
    Measure alphabetic characters among
    non-whitespace characters.
    """
    characters = [
        char
        for char in text
        if not char.isspace()
    ]

    if not characters:
        return 0.0

    alphabetic = sum(
        1
        for char in characters
        if char.isalpha()
    )

    return alphabetic / len(characters)


def evaluate_quality(
    text: str,
    rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Evaluate deterministic document-quality signals
    and return metrics, rule outcomes, and score.
    """
    if rules is None:
        rules = load_quality_rules()

    document_rules = rules["document_rules"]
    weights = rules["quality_scoring"]["weights"]

    character_count = len(text)
    word_count = len(text.split())

    repeated_lines = repeated_line_ratio(text)
    repeated_words = repeated_word_ratio(text)
    non_alphanumeric = non_alphanumeric_ratio(text)
    alphabetic = alphabetic_ratio(text)

    checks = {
        "minimum_characters": (
            character_count
            >= document_rules[
                "minimum_characters"
            ]["threshold"]
        ),
        "minimum_words": (
            word_count
            >= document_rules[
                "minimum_words"
            ]["threshold"]
        ),
        "repeated_line_ratio": (
            repeated_lines
            <= document_rules[
                "maximum_repeated_line_ratio"
            ]["threshold"]
        ),
        "repeated_word_ratio": (
            repeated_words
            <= document_rules[
                "maximum_repeated_word_ratio"
            ]["threshold"]
        ),
        "non_alphanumeric_ratio": (
            non_alphanumeric
            <= document_rules[
                "maximum_non_alphanumeric_ratio"
            ]["threshold"]
        ),
        "alphabetic_ratio": (
            alphabetic
            >= document_rules[
                "minimum_alphabetic_ratio"
            ]["threshold"]
        ),
    }

    score = sum(
        weights[check_name]
        for check_name, passed in checks.items()
        if passed
    )

    score = round(score, 6)

    approval_threshold = rules[
        "quality_scoring"
    ]["approval_threshold"]

    return {
        "metrics": {
            "character_count": character_count,
            "word_count": word_count,
            "repeated_line_ratio": repeated_lines,
            "repeated_word_ratio": repeated_words,
            "non_alphanumeric_ratio": non_alphanumeric,
            "alphabetic_ratio": alphabetic,
        },
        "checks": checks,
        "score": score,
        "passed": score >= approval_threshold,
    }