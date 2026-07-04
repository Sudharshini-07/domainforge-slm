from src.validation.validate_configs import (
    validate_domain_alignment,
    validate_domain_item_total,
    validate_percentage_group,
)


def test_valid_domain_alignment():
    taxonomy = {
        "domains": {
            "machine_learning": {},
            "deep_learning": {},
        }
    }

    benchmark_plan = {
        "domain_allocation": {
            "machine_learning": {},
            "deep_learning": {},
        }
    }

    errors = validate_domain_alignment(
        taxonomy,
        benchmark_plan,
    )

    assert errors == []


def test_unknown_domain_is_detected():
    taxonomy = {
        "domains": {
            "machine_learning": {},
        }
    }

    benchmark_plan = {
        "domain_allocation": {
            "machine_learning": {},
            "unknown_domain": {},
        }
    }

    errors = validate_domain_alignment(
        taxonomy,
        benchmark_plan,
    )

    assert len(errors) == 1
    assert "unknown_domain" in errors[0]


def test_valid_domain_item_total():
    benchmark_plan = {
        "target_total_items": 100,
        "domain_allocation": {
            "domain_a": {
                "target_items": 40
            },
            "domain_b": {
                "target_items": 60
            },
        },
    }

    errors = validate_domain_item_total(
        benchmark_plan
    )

    assert errors == []


def test_invalid_domain_item_total_is_detected():
    benchmark_plan = {
        "target_total_items": 100,
        "domain_allocation": {
            "domain_a": {
                "target_items": 40
            },
            "domain_b": {
                "target_items": 50
            },
        },
    }

    errors = validate_domain_item_total(
        benchmark_plan
    )

    assert len(errors) == 1
    assert "expected 100, got 90" in errors[0]


def test_valid_percentage_group():
    group = {
        "basic": {
            "target_percentage": 25
        },
        "intermediate": {
            "target_percentage": 45
        },
        "advanced": {
            "target_percentage": 30
        },
    }

    errors = validate_percentage_group(
        group,
        "Difficulty",
    )

    assert errors == []


def test_invalid_percentage_group_is_detected():
    group = {
        "basic": {
            "target_percentage": 25
        },
        "intermediate": {
            "target_percentage": 40
        },
        "advanced": {
            "target_percentage": 30
        },
    }

    errors = validate_percentage_group(
        group,
        "Difficulty",
    )

    assert len(errors) == 1
    assert "sum to 95, not 100" in errors[0]