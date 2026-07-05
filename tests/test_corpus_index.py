import json

import pytest

from src.data.corpus_index import (
    find_exact_duplicate,
    load_corpus_index,
    upsert_corpus_record,
)


def test_missing_index_loads_as_empty(tmp_path):
    index_path = tmp_path / "missing.jsonl"

    records = load_corpus_index(index_path)

    assert records == []


def test_load_corpus_index_reads_jsonl(tmp_path):
    index_path = tmp_path / "index.jsonl"

    first = {
        "document_id": "doc-1",
        "content_hash": "aaa",
    }

    second = {
        "document_id": "doc-2",
        "content_hash": "bbb",
    }

    index_path.write_text(
        json.dumps(first)
        + "\n"
        + json.dumps(second)
        + "\n",
        encoding="utf-8",
    )

    records = load_corpus_index(index_path)

    assert records == [first, second]


def test_invalid_json_is_rejected(tmp_path):
    index_path = tmp_path / "index.jsonl"

    index_path.write_text(
        '{"document_id": "doc-1"}\n'
        'not-valid-json\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="line 2",
    ):
        load_corpus_index(index_path)


def test_non_object_record_is_rejected(tmp_path):
    index_path = tmp_path / "index.jsonl"

    index_path.write_text(
        '["not", "an", "object"]\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="not a JSON object",
    ):
        load_corpus_index(index_path)


def test_find_exact_duplicate():
    records = [
        {
            "document_id": "doc-1",
            "content_hash": "abc123",
        },
        {
            "document_id": "doc-2",
            "content_hash": "def456",
        },
    ]

    duplicate = find_exact_duplicate(
        content_hash="abc123",
        records=records,
    )

    assert duplicate is not None
    assert duplicate["document_id"] == "doc-1"


def test_find_exact_duplicate_returns_none():
    records = [
        {
            "document_id": "doc-1",
            "content_hash": "abc123",
        }
    ]

    duplicate = find_exact_duplicate(
        content_hash="different",
        records=records,
    )

    assert duplicate is None


def test_find_exact_duplicate_can_exclude_current_document():
    records = [
        {
            "document_id": "doc-1",
            "content_hash": "abc123",
        }
    ]

    duplicate = find_exact_duplicate(
        content_hash="abc123",
        records=records,
        exclude_document_id="doc-1",
    )

    assert duplicate is None


def test_upsert_inserts_record(tmp_path):
    index_path = tmp_path / "index.jsonl"

    record = {
        "document_id": "doc-1",
        "content_hash": "abc123",
    }

    upsert_corpus_record(
        record=record,
        index_path=index_path,
    )

    records = load_corpus_index(index_path)

    assert records == [record]


def test_upsert_replaces_same_document_id(tmp_path):
    index_path = tmp_path / "index.jsonl"

    first = {
        "document_id": "doc-1",
        "content_hash": "old-hash",
    }

    updated = {
        "document_id": "doc-1",
        "content_hash": "new-hash",
    }

    upsert_corpus_record(
        record=first,
        index_path=index_path,
    )

    upsert_corpus_record(
        record=updated,
        index_path=index_path,
    )

    records = load_corpus_index(index_path)

    assert len(records) == 1
    assert records[0] == updated