"""Tests for deterministic near-duplicate text detection."""

import pytest

from src.data.near_duplicate import (
    create_word_shingles,
    is_near_duplicate,
    jaccard_similarity,
    text_similarity,
    tokenize_for_deduplication,
)


def test_tokenization_normalizes_case_and_punctuation():
    text = "Self-Attention, improves TRANSFORMER representations!"

    tokens = tokenize_for_deduplication(text)

    assert tokens == [
        "self",
        "attention",
        "improves",
        "transformer",
        "representations",
    ]


def test_tokenization_rejects_non_string_input():
    with pytest.raises(TypeError, match="text must be a string"):
        tokenize_for_deduplication(123)  # type: ignore[arg-type]


def test_create_word_shingles_is_deterministic():
    text = "one two three four five six seven"

    first = create_word_shingles(text, shingle_size=3)
    second = create_word_shingles(text, shingle_size=3)

    assert first == second
    assert len(first) == 5


def test_create_word_shingles_returns_empty_for_short_text():
    shingles = create_word_shingles(
        "too short",
        shingle_size=5,
    )

    assert shingles == set()


def test_create_word_shingles_rejects_invalid_size():
    with pytest.raises(
        ValueError,
        match="shingle_size must be greater than zero",
    ):
        create_word_shingles("some text", shingle_size=0)


def test_jaccard_similarity_for_identical_sets():
    shingles = {
        ("self", "attention"),
        ("attention", "mechanism"),
    }

    assert jaccard_similarity(shingles, shingles) == 1.0


def test_jaccard_similarity_for_disjoint_sets():
    left = {("self", "attention")}
    right = {("skin", "lesion")}

    assert jaccard_similarity(left, right) == 0.0


def test_jaccard_similarity_for_empty_sets():
    assert jaccard_similarity(set(), set()) == 0.0


def test_identical_text_has_full_similarity():
    text = (
        "Self attention computes contextual token representations "
        "using learned query key and value projections."
    )

    assert text_similarity(text, text) == 1.0


def test_case_and_punctuation_variation_has_full_similarity():
    left = (
        "Self attention computes contextual token representations "
        "using learned query key and value projections."
    )
    right = (
        "SELF ATTENTION computes contextual token representations, "
        "using learned query, key, and value projections!"
    )

    assert text_similarity(left, right) == 1.0


def test_lightly_edited_copy_is_detected_as_near_duplicate():
    left = (
        "Self attention allows each token to aggregate information "
        "from other tokens in the sequence using learned attention "
        "weights derived from query key similarity scores."
    )
    right = (
        "Self attention allows each token to aggregate information "
        "from other tokens in the sequence using learned attention "
        "weights derived from query key similarity values."
    )

    assert is_near_duplicate(
        left,
        right,
        threshold=0.80,
    )


def test_unrelated_documents_are_not_near_duplicates():
    left = (
        "Self attention computes relationships between tokens "
        "using query key and value projections in transformers."
    )
    right = (
        "Melanoma classification systems analyze dermoscopic images "
        "to distinguish malignant lesions from benign skin findings."
    )

    assert not is_near_duplicate(
        left,
        right,
        threshold=0.80,
    )


def test_short_texts_do_not_become_near_duplicates():
    assert not is_near_duplicate(
        "self attention",
        "self attention",
    )


@pytest.mark.parametrize(
    "threshold",
    [-0.01, 1.01],
)
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(
        ValueError,
        match="threshold must be between 0.0 and 1.0",
    ):
        is_near_duplicate(
            "some sufficiently long source text here",
            "some sufficiently long source text here",
            threshold=threshold,
        )


def test_similarity_is_symmetric():
    left = (
        "Transformer models use attention mechanisms "
        "to construct contextual representations."
    )
    right = (
        "Transformer models use attention mechanisms "
        "to build contextual representations."
    )

    assert text_similarity(left, right) == text_similarity(right, left)