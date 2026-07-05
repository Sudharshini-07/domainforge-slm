from src.data.corpus_index import (
    find_exact_duplicate,
    load_corpus_index,
    upsert_corpus_record,
)
from src.data.near_duplicate import is_near_duplicate


def test_exact_duplicate_found_through_persistent_index(tmp_path):
    index_path = tmp_path / "corpus_index.jsonl"

    upsert_corpus_record(
        record={
            "document_id": "doc-1",
            "content_hash": "hash-1",
            "normalized_text": (
                "transformers use self attention mechanisms"
            ),
        },
        index_path=index_path,
    )

    records = load_corpus_index(index_path)

    result = find_exact_duplicate(
        content_hash="hash-1",
        records=records,
    )

    assert result is not None
    assert result["document_id"] == "doc-1"


def test_near_duplicate_detected_from_persistent_index(tmp_path):
    index_path = tmp_path / "corpus_index.jsonl"

    stored_text = (
        "transformer models use self attention mechanisms "
        "to process long sequences in parallel efficiently"
    )

    upsert_corpus_record(
        record={
            "document_id": "doc-1",
            "content_hash": "hash-1",
            "normalized_text": stored_text,
        },
        index_path=index_path,
    )

    records = load_corpus_index(index_path)

    query_text = (
        "transformer models use self attention mechanisms "
        "to process long sequences in parallel effectively"
    )

    result = is_near_duplicate(
        query_text,
        records[0]["normalized_text"],
        threshold=0.8,
    )

    assert result is True


def test_unrelated_document_is_not_near_duplicate(tmp_path):
    index_path = tmp_path / "corpus_index.jsonl"

    stored_text = (
        "transformer models use self attention mechanisms "
        "to process long sequences in parallel efficiently"
    )

    upsert_corpus_record(
        record={
            "document_id": "doc-1",
            "content_hash": "hash-1",
            "normalized_text": stored_text,
        },
        index_path=index_path,
    )

    records = load_corpus_index(index_path)

    query_text = (
        "database transactions preserve consistency "
        "through atomicity isolation and durability guarantees"
    )

    result = is_near_duplicate(
        query_text,
        records[0]["normalized_text"],
        threshold=0.8,
    )

    assert result is False


def test_exact_duplicate_can_exclude_current_document(tmp_path):
    index_path = tmp_path / "corpus_index.jsonl"

    upsert_corpus_record(
        record={
            "document_id": "doc-1",
            "content_hash": "hash-1",
            "normalized_text": (
                "small language models can be adapted "
                "to specialized technical domains"
            ),
        },
        index_path=index_path,
    )

    records = load_corpus_index(index_path)

    result = find_exact_duplicate(
        content_hash="hash-1",
        records=records,
        exclude_document_id="doc-1",
    )

    assert result is None