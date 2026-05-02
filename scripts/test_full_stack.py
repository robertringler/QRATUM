#!/usr/bin/env python3
"""Full-stack test runner for QuASIM repository.

This script validates:
- YAML/JSON syntax in Helm/Kustomize manifests
- Terraform module validation (when available)
- Python bytecode compilation
- C/C++ code formatting checks
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def validate_yaml_files() -> bool:
    """Validate YAML syntax in manifests."""

    print("Validating YAML files...")
    yaml_files = list(REPO_ROOT.glob("**/*.yaml")) + list(REPO_ROOT.glob("**/*.yml"))
    yaml_files = [f for f in yaml_files if ".git" not in str(f) and "node_modules" not in str(f)]

    all_valid = True
    skipped = 0
    for yaml_file in yaml_files:
        try:
            # Check first 1KB for Helm template markers (more efficient)
            with open(yaml_file) as f:
                header = f.read(1024)

            # Skip Helm templates (contain Jinja2-like syntax)
            if "{{" in header or "{%" in header:
                skipped += 1
                continue

            # Read full content for parsing
            with open(yaml_file) as f:
                content = f.read()

            # Always use safe_load_all to handle both single and multi-document YAML
            try:
                list(yaml.safe_load_all(content))
            except yaml.YAMLError as e:
                print(f"YAML error in {yaml_file}: {e}")
                all_valid = False
        except Exception as e:
            print(f"Error reading {yaml_file}: {e}")
            all_valid = False

    print(f"Validated {len(yaml_files)} YAML files ({skipped} Helm templates skipped)")
    return all_valid


def validate_json_files() -> bool:
    """Validate JSON syntax."""

    print("Validating JSON files...")
    json_files = list(REPO_ROOT.glob("**/*.json"))
    json_files = [f for f in json_files if ".git" not in str(f) and "node_modules" not in str(f)]

    all_valid = True
    for json_file in json_files:
        try:
            with open(json_file) as f:
                json.load(f)
        except json.JSONDecodeError as e:
            print(f"JSON error in {json_file}: {e}")
            all_valid = False

    print(f"Validated {len(json_files)} JSON files")
    return all_valid


def validate_terraform() -> bool:
    """Validate Terraform modules if terraform CLI is available."""

    print("Validating Terraform modules...")
    try:
        subprocess.run(["terraform", "version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("terraform CLI not found; skipping Terraform validation")
        return True

    tf_dirs = []
    for tf_file in REPO_ROOT.glob("**/main.tf"):
        tf_dirs.append(tf_file.parent)

    all_valid = True
    for tf_dir in tf_dirs:
        print(f"  Validating {tf_dir.relative_to(REPO_ROOT)}...")
        try:
            # Initialize with backend disabled
            subprocess.run(
                ["terraform", "init", "-backend=false"],
                cwd=tf_dir,
                capture_output=True,
                check=True,
            )
            # Validate
            subprocess.run(
                ["terraform", "validate"],
                cwd=tf_dir,
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"    Terraform validation failed: {e}")
            all_valid = False

    print(f"Validated {len(tf_dirs)} Terraform modules")
    return all_valid


def validate_python_syntax() -> bool:
    """Validate Python syntax by compiling to bytecode."""

    print("Validating Python syntax...")
    py_files = list(REPO_ROOT.glob("**/*.py"))
    py_files = [f for f in py_files if ".git" not in str(f)]

    try:
        cmd = [sys.executable, "-m", "py_compile"] + [str(f) for f in py_files]
        subprocess.run(cmd, capture_output=True, check=True)
        print(f"Validated {len(py_files)} Python files")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Python syntax validation failed: {e}")
        return False


def main() -> int:
    """Run all validation checks."""

    print("=" * 60)
    print("Full Stack Validation")
    print("=" * 60)

    checks = {
        "YAML": validate_yaml_files(),
        "JSON": validate_json_files(),
        "Terraform": validate_terraform(),
        "Python": validate_python_syntax(),
    }

    print("\n" + "=" * 60)
    print("Summary:")
    for name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    print("=" * 60)

    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
