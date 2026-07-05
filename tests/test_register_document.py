from jsonschema.exceptions import ValidationError
import pytest

from src.data.register_document import (
    build_manifest_record,
    compute_content_hash,
    generate_document_id,
    normalize_text,
    validate_taxonomy_path,
    validate_manifest_record,
)


def test_normalize_text_line_endings():
    text = "Line one\r\nLine two\rLine three"

    normalized = normalize_text(text)

    assert normalized == (
        "Line one\nLine two\nLine three"
    )


def test_normalize_text_removes_excess_blank_lines():
    text = "First\n\n\n\nSecond"

    normalized = normalize_text(text)

    assert normalized == "First\n\nSecond"


def test_content_hash_is_deterministic():
    text = "Scaled dot-product attention"

    first_hash = compute_content_hash(text)
    second_hash = compute_content_hash(text)

    assert first_hash == second_hash


def test_different_content_produces_different_hash():
    first_hash = compute_content_hash(
        "Self-attention"
    )

    second_hash = compute_content_hash(
        "Cross-attention"
    )

    assert first_hash != second_hash


def test_document_id_is_deterministic():
    content_hash = compute_content_hash(
        "Transformer architecture"
    )

    first_id = generate_document_id(
        source_name="DomainForge Original Corpus",
        content_hash=content_hash,
    )

    second_id = generate_document_id(
        source_name="DomainForge Original Corpus",
        content_hash=content_hash,
    )

    assert first_id == second_id


def test_document_id_contains_source_slug():
    content_hash = compute_content_hash(
        "Transformer architecture"
    )

    document_id = generate_document_id(
        source_name="DomainForge Original Corpus",
        content_hash=content_hash,
    )

    assert document_id.startswith(
        "domainforge-original-corpus-"
    )


def test_valid_taxonomy_path():
    taxonomy = {
        "domains": {
            "transformers": {
                "subdomains": {
                    "attention_mechanisms": {
                        "topics": [
                            "self_attention",
                            "cross_attention",
                        ]
                    }
                }
            }
        }
    }

    validate_taxonomy_path(
        taxonomy=taxonomy,
        domain="transformers",
        subdomain="attention_mechanisms",
        topic="self_attention",
    )


def test_unknown_domain_is_rejected():
    taxonomy = {
        "domains": {
            "transformers": {
                "subdomains": {}
            }
        }
    }

    with pytest.raises(
        ValueError,
        match="Unknown domain",
    ):
        validate_taxonomy_path(
            taxonomy=taxonomy,
            domain="computer_vision",
            subdomain="cnn",
            topic="resnet",
        )


def test_unknown_subdomain_is_rejected():
    taxonomy = {
        "domains": {
            "transformers": {
                "subdomains": {
                    "attention_mechanisms": {
                        "topics": [
                            "self_attention"
                        ]
                    }
                }
            }
        }
    }

    with pytest.raises(
        ValueError,
        match="Unknown subdomain",
    ):
        validate_taxonomy_path(
            taxonomy=taxonomy,
            domain="transformers",
            subdomain="wrong_subdomain",
            topic="self_attention",
        )


def test_unknown_topic_is_rejected():
    taxonomy = {
        "domains": {
            "transformers": {
                "subdomains": {
                    "attention_mechanisms": {
                        "topics": [
                            "self_attention"
                        ]
                    }
                }
            }
        }
    }

    with pytest.raises(
        ValueError,
        match="Unknown topic",
    ):
        validate_taxonomy_path(
            taxonomy=taxonomy,
            domain="transformers",
            subdomain="attention_mechanisms",
            topic="wrong_topic",
        )


def test_empty_document_is_rejected():
    with pytest.raises(
        ValueError,
        match="empty after normalization",
    ):
        build_manifest_record(
            text="   \n\n   ",
            source_family="curated_original_material",
            source_name="DomainForge Original Corpus",
            source_type="original_material",
            source_reference="internal:test",
            title="Empty Test Document",
            domain="transformers",
            subdomain="attention_mechanisms",
            topic="self_attention",
        )


def test_manifest_record_contains_expected_metadata():
    record = build_manifest_record(
        text=(
            "Self-attention allows each token "
            "to attend to other tokens."
        ),
        source_family="curated_original_material",
        source_name="DomainForge Original Corpus",
        source_type="original_material",
        source_reference="internal:test-document",
        title="Introduction to Self-Attention",
        domain="transformers",
        subdomain="attention_mechanisms",
        topic="self_attention",
    )

    assert record["domain"] == "transformers"

    assert (
        record["subdomain"]
        == "attention_mechanisms"
    )

    assert record["topic"] == "self_attention"

    assert (
        record["usage_status"]
        == "pending_review"
    )

    assert (
        record["processing_status"]
        == "registered"
    )

    assert (
        record["contamination"]["status"]
        == "not_checked"
    )

    assert record["statistics"]["word_count"] > 0

    assert len(record["content_hash"]) == 64

def test_manifest_rejects_invalid_usage_status():
    record = build_manifest_record(
        text="Self-attention computes contextual token representations.",
        source_family="curated_original_material",
        source_name="DomainForge Original Corpus",
        source_type="original_material",
        source_reference="internal:schema-test",
        title="Schema Validation Test",
        domain="transformers",
        subdomain="attention_mechanisms",
        topic="self_attention",
    )

    record["usage_status"] = "maybe_ok"

    with pytest.raises(
        ValidationError,
        match="Manifest validation failed",
    ):
        validate_manifest_record(record)


def test_manifest_rejects_invalid_content_hash():
    record = build_manifest_record(
        text="Transformers use attention mechanisms.",
        source_family="curated_original_material",
        source_name="DomainForge Original Corpus",
        source_type="original_material",
        source_reference="internal:hash-test",
        title="Hash Validation Test",
        domain="transformers",
        subdomain="attention_mechanisms",
        topic="self_attention",
    )

    record["content_hash"] = "not-a-valid-sha256"

    with pytest.raises(
        ValidationError,
        match="Manifest validation failed",
    ):
        validate_manifest_record(record)


def test_manifest_rejects_unexpected_field():
    record = build_manifest_record(
        text="KV caching reduces repeated computation during decoding.",
        source_family="curated_original_material",
        source_name="DomainForge Original Corpus",
        source_type="original_material",
        source_reference="internal:extra-field-test",
        title="Unexpected Field Test",
        domain="transformers",
        subdomain="inference_mechanics",
        topic="kv_cache",
    )

    record["random_unexpected_field"] = "should fail"

    with pytest.raises(
        ValidationError,
        match="Manifest validation failed",
    ):
        validate_manifest_record(record)

def test_manifest_contains_normalized_artifact_path():
    record = build_manifest_record(
        text="Self-attention creates contextual representations.",
        source_family="curated_original_material",
        source_name="DomainForge Original Corpus",
        source_type="original_material",
        source_reference="internal:path-test",
        title="Artifact Path Test",
        domain="transformers",
        subdomain="attention_mechanisms",
        topic="self_attention",
    )

    expected_path = (
        "data/interim/normalized/"
        f"{record['document_id']}.txt"
    )

    assert (
        record["normalized_artifact_path"]
        == expected_path
    )


def test_manifest_rejects_invalid_artifact_path():
    record = build_manifest_record(
        text="Attention masks control visible token interactions.",
        source_family="curated_original_material",
        source_name="DomainForge Original Corpus",
        source_type="original_material",
        source_reference="internal:invalid-path-test",
        title="Invalid Artifact Path Test",
        domain="transformers",
        subdomain="attention_mechanisms",
        topic="self_attention",
    )

    record["normalized_artifact_path"] = (
        "C:/Users/example/random.txt"
    )

    with pytest.raises(
        ValidationError,
        match="Manifest validation failed",
    ):
        validate_manifest_record(record)