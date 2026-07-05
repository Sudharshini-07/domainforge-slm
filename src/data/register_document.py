from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TAXONOMY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "domain_taxonomy.yaml"
)

MANIFEST_SCHEMA_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifest_schema.json"
)

PIPELINE_VERSION = "0.1.0"


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file."""
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file."""
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)

def validate_manifest_record(
    record: dict[str, Any],
) -> None:
    """
    Validate a manifest record against the
    Draft 2020-12 JSON Schema.
    """
    schema = load_json(
        MANIFEST_SCHEMA_PATH
    )

    validator = Draft202012Validator(
        schema
    )

    errors = sorted(
        validator.iter_errors(record),
        key=lambda error: list(error.path),
    )

    if errors:
        messages = []

        for error in errors:
            location = ".".join(
                str(part)
                for part in error.path
            )

            if not location:
                location = "<root>"

            messages.append(
                f"{location}: {error.message}"
            )

        raise ValidationError(
            "Manifest validation failed:\n"
            + "\n".join(messages)
        )

def normalize_text(text: str) -> str:
    """
    Apply deterministic lightweight normalization.

    This intentionally avoids aggressive cleaning because
    technical punctuation and formatting may carry meaning.
    """
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = "\n".join(
        line.rstrip()
        for line in text.split("\n")
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def compute_content_hash(
    normalized_text: str,
) -> str:
    """Compute SHA-256 over normalized UTF-8 text."""
    return hashlib.sha256(
        normalized_text.encode("utf-8")
    ).hexdigest()


def generate_document_id(
    source_name: str,
    content_hash: str,
) -> str:
    """
    Generate a stable document ID using
    source name plus content hash prefix.
    """
    source_slug = re.sub(
        r"[^a-z0-9]+",
        "-",
        source_name.lower(),
    ).strip("-")

    return (
        f"{source_slug}-"
        f"{content_hash[:16]}"
    )


def validate_taxonomy_path(
    taxonomy: dict[str, Any],
    domain: str,
    subdomain: str,
    topic: str,
) -> None:
    """
    Validate domain → subdomain → topic
    against domain_taxonomy.yaml.
    """
    domains = taxonomy.get(
        "domains",
        {},
    )

    if domain not in domains:
        raise ValueError(
            f"Unknown domain: {domain}"
        )

    subdomains = domains[domain].get(
        "subdomains",
        {},
    )

    if subdomain not in subdomains:
        raise ValueError(
            f"Unknown subdomain "
            f"'{subdomain}' under domain "
            f"'{domain}'"
        )

    topics = subdomains[subdomain].get(
        "topics",
        [],
    )

    if topic not in topics:
        raise ValueError(
            f"Unknown topic "
            f"'{topic}' under "
            f"'{domain}/{subdomain}'"
        )


def count_words(text: str) -> int:
    """Count whitespace-delimited words."""
    return len(text.split())


def build_manifest_record(
    *,
    text: str,
    source_family: str,
    source_name: str,
    source_type: str,
    source_reference: str,
    title: str,
    domain: str,
    subdomain: str,
    topic: str,
    language: str = "en",
    publication_or_update_date: str | None = None,
    usage_status: str = "pending_review",
    usage_note: str = "Pending source usage review.",
) -> dict[str, Any]:
    """
    Build a document-level manifest record.
    """
    taxonomy = load_yaml(
        TAXONOMY_PATH
    )

    validate_taxonomy_path(
        taxonomy=taxonomy,
        domain=domain,
        subdomain=subdomain,
        topic=topic,
    )

    normalized_text = normalize_text(text)

    if not normalized_text:
        raise ValueError(
            "Document is empty after normalization."
        )

    content_hash = compute_content_hash(
        normalized_text
    )

    document_id = generate_document_id(
        source_name=source_name,
        content_hash=content_hash,
    )
    normalized_artifact_path = (
        "data/interim/normalized/"
        f"{document_id}.txt"
        )

    retrieved_at = datetime.now(
        timezone.utc
    ).isoformat()

    record = {
        "document_id": document_id,
        "source_family": source_family,
        "source_name": source_name,
        "source_type": source_type,
        "source_reference": source_reference,
        "title": title,
        "domain": domain,
        "subdomain": subdomain,
        "topic": topic,
        "language": language,
        "publication_or_update_date":
            publication_or_update_date,
        "retrieved_at": retrieved_at,
        "usage_status": usage_status,
        "usage_note": usage_note,
        "content_hash": content_hash,
        "normalized_artifact_path":
        normalized_artifact_path,
        "pipeline_version": PIPELINE_VERSION,
        "processing_status": "registered",
        "quality": {
            "score": None,
            "language_verified": False,
            "minimum_length_passed": False,
            "boilerplate_ratio": None,
        },
        "deduplication": {
            "exact_duplicate": False,
            "near_duplicate": False,
            "duplicate_of": None,
        },
        "contamination": {
            "status": "not_checked",
            "matched_benchmark_ids": [],
        },
        "statistics": {
            "character_count": len(
                normalized_text
            ),
            "word_count": count_words(
                normalized_text
            ),
            "token_count": None,
        },
    }

    validate_manifest_record(record)

    return record


def register_text_file(
    *,
    input_path: Path,
    output_path: Path,
    source_family: str,
    source_name: str,
    source_type: str,
    source_reference: str,
    title: str,
    domain: str,
    subdomain: str,
    topic: str,
) -> dict[str, Any]:
    """
    Read a UTF-8 text file, persist the exact normalized
    text artifact, and write its manifest record as JSON.
    """
    text = input_path.read_text(
        encoding="utf-8"
    )

    normalized_text = normalize_text(text)

    record = build_manifest_record(
        text=text,
        source_family=source_family,
        source_name=source_name,
        source_type=source_type,
        source_reference=source_reference,
        title=title,
        domain=domain,
        subdomain=subdomain,
        topic=topic,
    )

    normalized_output_path = (
        PROJECT_ROOT
        / record["normalized_artifact_path"]
        )

    normalized_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalized_output_path.write_text(
        normalized_text,
        encoding="utf-8",
    )

    persisted_text = normalized_output_path.read_text(
        encoding="utf-8"
    )

    persisted_hash = compute_content_hash(
        persisted_text
    )

    if persisted_hash != record["content_hash"]:
        raise ValueError(
            "Persisted normalized artifact hash "
            "does not match manifest content hash."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            record,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return record


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Register a text document in the "
            "DomainForge-SLM corpus pipeline."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--source-family",
        required=True,
    )

    parser.add_argument(
        "--source-name",
        required=True,
    )

    parser.add_argument(
        "--source-type",
        required=True,
    )

    parser.add_argument(
        "--source-reference",
        required=True,
    )

    parser.add_argument(
        "--title",
        required=True,
    )

    parser.add_argument(
        "--domain",
        required=True,
    )

    parser.add_argument(
        "--subdomain",
        required=True,
    )

    parser.add_argument(
        "--topic",
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    record = register_text_file(
        input_path=args.input,
        output_path=args.output,
        source_family=args.source_family,
        source_name=args.source_name,
        source_type=args.source_type,
        source_reference=args.source_reference,
        title=args.title,
        domain=args.domain,
        subdomain=args.subdomain,
        topic=args.topic,
    )

    print(
        "Document registered successfully."
    )
    print(
        f"Document ID: "
        f"{record['document_id']}"
    )
    print(
        f"Content hash: "
        f"{record['content_hash']}"
    )
    print(
        f"Word count: "
        f"{record['statistics']['word_count']}"
    )


if __name__ == "__main__":
    main()