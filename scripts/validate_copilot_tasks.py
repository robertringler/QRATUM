#!/usr/bin/env python3
"""

Validate GitHub Copilot task definitions.

This script validates YAML structure and required fields for Copilot task files.
"""

import sys
from pathlib import Path

import yaml


def validate_task_file(filepath: Path) -> tuple[bool, list[str]]:
    """

    Validate a single Copilot task YAML file.

    Args:
        filepath: Path to the YAML file to validate

    Returns:
        Tuple of (is_valid, list_of_errors)
    """

    errors = []

    try:
        with open(filepath) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return False, [f"YAML parsing error: {e}"]
    except Exception as e:
        return False, [f"Error reading file: {e}"]

    # Check required top-level fields (support both 'name' and 'task' formats)
    if "name" not in data and "task" not in data:
        errors.append("Missing required field: 'name' or 'task'")
    if "description" not in data:
        errors.append("Missing required field: 'description'")
    if "steps" not in data:
        errors.append("Missing required field: 'steps'")

    # Validate steps structure
    if "steps" in data:
        if not isinstance(data["steps"], list):
            errors.append("'steps' must be a list")
        else:
            for i, step in enumerate(data["steps"], 1):
                if not isinstance(step, dict):
                    errors.append(f"Step {i} must be a dictionary")
                    continue

                # Check required step fields
                if "name" not in step:
                    errors.append(f"Step {i} missing 'name' field")
                # 'id' field is optional for compatibility with different formats
                # 'run' field is optional for compatibility with different formats

    return len(errors) == 0, errors


def main():
    """Main validation function."""

    repo_root = Path(__file__).parent.parent
    tasks_dir = repo_root / ".github" / "copilot-tasks"

    if not tasks_dir.exists():
        print(f"❌ Tasks directory not found: {tasks_dir}")
        return 1

    yaml_files = list(tasks_dir.glob("*.yaml")) + list(tasks_dir.glob("*.yml"))

    if not yaml_files:
        print(f"⚠️  No YAML files found in {tasks_dir}")
        return 0

    print(f"Validating {len(yaml_files)} Copilot task file(s)...\n")

    all_valid = True
    for filepath in yaml_files:
        is_valid, errors = validate_task_file(filepath)

        if is_valid:
            print(f"✅ {filepath.name}: Valid")
        else:
            print(f"❌ {filepath.name}: Invalid")
            for error in errors:
                print(f"   - {error}")
            all_valid = False

    print()
    if all_valid:
        print("✅ All Copilot task files are valid!")
        return 0
    else:
        print("❌ Some Copilot task files have validation errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())
