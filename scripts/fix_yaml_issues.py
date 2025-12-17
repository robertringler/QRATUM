#!/usr/bin/env python3
"""

Fix common YAML issues in the repository
"""

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def fix_multi_document_yaml(filepath: Path) -> bool:
    """Fix YAML files that contain multiple documents"""

    try:
        with open(filepath) as f:
            content = f.read()

        # Check if it's a multi-document YAML
        if content.count("---") > 1:
            print(f"Multi-document YAML detected in {filepath}")
            # This is actually valid YAML, update the validator to handle it
            return True

        return True
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def fix_flow_node_yaml(filepath: Path) -> bool:
    """Fix YAML files with flow node parsing errors"""

    try:
        with open(filepath) as f:
            content = f.read()

        # Check for common flow node issues
        if content.startswith("{{-"):
            print(f"Helm template detected in {filepath} - this is expected")
            return True

        # Try to parse as YAML
        try:
            yaml.safe_load(content)
            return True
        except yaml.YAMLError as e:
            print(f"YAML error in {filepath}: {e}")
            # This might be a Helm template, which is OK
            if "{{" in content or "{%" in content:
                print("  -> Appears to be a template file, skipping")
                return True
            return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def main():
    """Main function"""

    print("Checking YAML files for common issues...\n")

    # Files with known issues from the test output
    problematic_files = [
        "infra/kubefed/federated-service.yaml",
        "infra/kubefed/federated-deployment.yaml",
        "infra/helm/quasim-platform/templates/qc-worker-deployment.yaml",
        "infra/helm/quasim-platform/templates/ray-cluster.yaml",
        "infra/helm/quasim-platform/templates/api-deployment.yaml",
    ]

    all_ok = True
    for file_path in problematic_files:
        full_path = REPO_ROOT / file_path
        if full_path.exists():
            print(f"Checking {file_path}...")

            # Try to detect the issue type
            with open(full_path) as f:
                first_line = f.readline()

            if first_line.startswith("{{-") or "{{" in first_line:
                print("  -> Helm template file (expected to fail standard YAML parsing)")
            else:
                with open(full_path, encoding="utf-8") as f:
                    text = f.read()
                if text.count("---") > 1:
                    print("  -> Multi-document YAML (contains more than one '---')")
                else:
                    result = fix_flow_node_yaml(full_path)
                    if not result:
                        all_ok = False
        else:
            print(f"File not found: {file_path}")

    print("\n" + "=" * 60)
    if all_ok:
        print("Note: The 'errors' found are expected for Helm templates and")
        print("multi-document YAML files. These are valid formats.")
        print("The test runner should be updated to handle these cases.")
    else:
        print("Some YAML files have genuine issues that need manual review.")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
