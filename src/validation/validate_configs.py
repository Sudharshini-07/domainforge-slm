from pathlib import Path
import json
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "configs"
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmarks"

DOMAIN_SCOPE_PATH = CONFIG_DIR / "domain_scope.yaml"
DOMAIN_TAXONOMY_PATH = CONFIG_DIR / "domain_taxonomy.yaml"
BENCHMARK_PLAN_PATH = CONFIG_DIR / "benchmark_plan.yaml"
BENCHMARK_SCHEMA_PATH = BENCHMARK_DIR / "benchmark_schema.json"


def load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a dictionary."""
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_json(path: Path) -> dict:
    """Load a JSON file and return its contents as a dictionary."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_required_files() -> list[str]:
    """Check that all required configuration files exist."""
    required_paths = [
        DOMAIN_SCOPE_PATH,
        DOMAIN_TAXONOMY_PATH,
        BENCHMARK_PLAN_PATH,
        BENCHMARK_SCHEMA_PATH,
    ]

    errors = []

    for path in required_paths:
        if not path.exists():
            errors.append(f"Missing required file: {path}")

    return errors


def validate_domain_alignment(
    taxonomy: dict,
    benchmark_plan: dict,
) -> list[str]:
    """
    Check that benchmark-plan domains exist
    in the domain taxonomy.
    """
    errors = []

    taxonomy_domains = set(
        taxonomy.get("domains", {}).keys()
    )

    planned_domains = set(
        benchmark_plan.get("domain_allocation", {}).keys()
    )

    unknown_domains = planned_domains - taxonomy_domains

    for domain in sorted(unknown_domains):
        errors.append(
            f"Benchmark plan contains unknown domain: {domain}"
        )

    return errors


def validate_domain_item_total(
    benchmark_plan: dict,
) -> list[str]:
    """
    Check that domain target counts sum
    to target_total_items.
    """
    errors = []

    expected_total = benchmark_plan.get(
        "target_total_items", 0
    )

    domain_allocation = benchmark_plan.get(
        "domain_allocation", {}
    )

    actual_total = sum(
        domain_config.get("target_items", 0)
        for domain_config in domain_allocation.values()
    )

    if actual_total != expected_total:
        errors.append(
            "Domain allocation mismatch: "
            f"expected {expected_total}, got {actual_total}"
        )

    return errors


def validate_percentage_group(
    group: dict,
    group_name: str,
) -> list[str]:
    """Check that target percentages sum to 100."""
    errors = []

    total = sum(
        item.get("target_percentage", 0)
        for item in group.values()
    )

    if total != 100:
        errors.append(
            f"{group_name} percentages sum to {total}, not 100"
        )

    return errors


def validate_task_types(
    domain_scope: dict,
    benchmark_plan: dict,
    benchmark_schema: dict,
) -> list[str]:
    """
    Check task-type consistency across:
    - domain scope
    - benchmark plan
    - benchmark schema
    """
    errors = []

    scope_task_types = set(
        domain_scope.get("task_taxonomy", {}).keys()
    )

    plan_task_types = set(
        benchmark_plan.get("task_type_targets", {}).keys()
    )

    schema_task_types = set(
        benchmark_schema
        .get("fields", {})
        .get("task_type", {})
        .get("allowed_values", [])
    )

    if scope_task_types != plan_task_types:
        errors.append(
            "Task types differ between "
            "domain_scope.yaml and benchmark_plan.yaml"
        )

    if scope_task_types != schema_task_types:
        errors.append(
            "Task types differ between "
            "domain_scope.yaml and benchmark_schema.json"
        )

    return errors


def main() -> None:
    print("Validating DomainForge-SLM configuration...\n")

    errors = validate_required_files()

    if errors:
        for error in errors:
            print(f"[ERROR] {error}")
        sys.exit(1)

    domain_scope = load_yaml(DOMAIN_SCOPE_PATH)
    taxonomy = load_yaml(DOMAIN_TAXONOMY_PATH)
    benchmark_plan = load_yaml(BENCHMARK_PLAN_PATH)
    benchmark_schema = load_json(BENCHMARK_SCHEMA_PATH)

    errors.extend(
        validate_domain_alignment(
            taxonomy,
            benchmark_plan,
        )
    )

    errors.extend(
        validate_domain_item_total(
            benchmark_plan
        )
    )

    errors.extend(
        validate_percentage_group(
            benchmark_plan.get("task_type_targets", {}),
            "Task type",
        )
    )

    errors.extend(
        validate_percentage_group(
            benchmark_plan.get("difficulty_targets", {}),
            "Difficulty",
        )
    )

    errors.extend(
        validate_percentage_group(
            benchmark_plan.get("split_policy", {}),
            "Split",
        )
    )

    errors.extend(
        validate_task_types(
            domain_scope,
            benchmark_plan,
            benchmark_schema,
        )
    )

    if errors:
        print("Validation failed:\n")

        for error in errors:
            print(f"[ERROR] {error}")

        sys.exit(1)

    print("[OK] All required files exist")
    print("[OK] Benchmark domains match taxonomy")
    print("[OK] Domain item allocation is valid")
    print("[OK] Task percentages sum to 100")
    print("[OK] Difficulty percentages sum to 100")
    print("[OK] Split percentages sum to 100")
    print("[OK] Task types are consistent")
    print("\nConfiguration validation passed.")


if __name__ == "__main__":
    main()