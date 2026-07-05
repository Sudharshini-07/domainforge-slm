from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INDEX_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "corpus_index.jsonl"
)


def load_corpus_index(
    index_path: Path = DEFAULT_INDEX_PATH,
) -> list[dict[str, Any]]:
    """
    Load a JSONL corpus index.

    Missing index files are treated as an empty corpus.
    """
    if not index_path.exists():
        return []

    records: list[dict[str, Any]] = []

    with index_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "Invalid JSON in corpus index "
                    f"at line {line_number}: {error.msg}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    "Corpus index record at line "
                    f"{line_number} is not a JSON object."
                )

            records.append(record)

    return records


def find_exact_duplicate(
    *,
    content_hash: str,
    records: list[dict[str, Any]],
    exclude_document_id: str | None = None,
) -> dict[str, Any] | None:
    """
    Find an existing record with the same content hash.

    Optionally exclude the current document ID so that
    re-registering the same document is idempotent.
    """
    for record in records:
        if (
            exclude_document_id is not None
            and record.get("document_id")
            == exclude_document_id
        ):
            continue

        if record.get("content_hash") == content_hash:
            return record

    return None


def upsert_corpus_record(
    *,
    record: dict[str, Any],
    index_path: Path = DEFAULT_INDEX_PATH,
) -> None:
    """
    Insert or replace a corpus-index record by document_id.

    The complete index is rewritten atomically.
    """
    records = load_corpus_index(index_path)

    updated_records = [
        existing
        for existing in records
        if existing.get("document_id")
        != record["document_id"]
    ]

    updated_records.append(record)

    index_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = index_path.with_suffix(
        index_path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for item in updated_records:
            file.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            file.write("\n")

    temporary_path.replace(index_path)