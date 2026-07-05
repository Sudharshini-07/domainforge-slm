from src.data.quality import (
    alphabetic_ratio,
    evaluate_quality,
    non_alphanumeric_ratio,
    repeated_line_ratio,
    repeated_word_ratio,
)


def test_repeated_line_ratio_no_repetition():
    text = "First line\nSecond line\nThird line"

    assert repeated_line_ratio(text) == 0.0


def test_repeated_line_ratio_detects_repetition():
    text = "Repeat\nRepeat\nUnique"

    assert repeated_line_ratio(text) == 1 / 3


def test_repeated_word_ratio_detects_dominance():
    text = "attention attention attention model"

    assert repeated_word_ratio(text) == 0.75


def test_non_alphanumeric_ratio():
    text = "abc!!!"

    assert non_alphanumeric_ratio(text) == 0.5


def test_alphabetic_ratio():
    text = "abc123"

    assert alphabetic_ratio(text) == 0.5


def test_high_quality_document_passes():
    text = (
        "Self-attention computes contextual token "
        "representations by comparing query and key "
        "vectors and combining value vectors. "
    ) * 8

    result = evaluate_quality(text)

    assert result["passed"] is True
    assert result["score"] >= 0.80


def test_short_document_fails_length_checks():
    text = "Short technical note."

    result = evaluate_quality(text)

    assert (
        result["checks"]["minimum_characters"]
        is False
    )

    assert (
        result["checks"]["minimum_words"]
        is False
    )


def test_repetitive_document_detected():
    text = (
        "spam spam spam spam spam spam "
        "spam spam spam useful"
    )

    result = evaluate_quality(text)

    assert (
        result["checks"]["repeated_word_ratio"]
        is False
    )