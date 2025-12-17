#!/usr/bin/env python3
"""

Export Control Scanner
Scans codebase for ITAR/EAR controlled technology patterns
"""

import re
from pathlib import Path

import yaml


def load_patterns():
    """Load export control patterns from configuration"""

    config_file = Path(__file__).parent.parent / "config" / "export-patterns.yml"
    if config_file.exists():
        with open(config_file) as f:
            return yaml.safe_load(f)
    return {}


def scan_file(file_path, patterns):
    """Scan a single file for export-controlled content"""

    findings = []

    try:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            content = f.read()

            # Check for ITAR keywords
            for category, data in patterns.get("itar_patterns", {}).items():
                for keyword in data.get("keywords", []):
                    if re.search(r"\b" + re.escape(keyword) + r"\b", content, re.IGNORECASE):
                        findings.append(
                            {
                                "file": str(file_path),
                                "type": "ITAR",
                                "category": category,
                                "keyword": keyword,
                                "severity": "HIGH",
                            }
                        )

            # Check for EAR keywords
            for eccn, data in patterns.get("ear_patterns", {}).items():
                for keyword in data.get("keywords", []):
                    if re.search(r"\b" + re.escape(keyword) + r"\b", content, re.IGNORECASE):
                        findings.append(
                            {
                                "file": str(file_path),
                                "type": "EAR",
                                "eccn": eccn,
                                "keyword": keyword,
                                "severity": "MEDIUM",
                            }
                        )

    except Exception as e:
        print(f"Error scanning {file_path}: {e}")

    return findings


def scan_repository():
    """Scan entire repository for export-controlled content"""

    patterns = load_patterns()
    all_findings = []

    # Scan Python files
    for py_file in Path(".").rglob("*.py"):
        if ".git" in str(py_file) or "node_modules" in str(py_file):
            continue
        findings = scan_file(py_file, patterns)
        all_findings.extend(findings)

    # Print results
    print("\n" + "=" * 60)
    print("Export Control Scan Results")
    print("=" * 60)

    if all_findings:
        print(f"\n⚠️  Found {len(all_findings)} potential export-controlled items:\n")
        for finding in all_findings:
            print(f"  [{finding['severity']}] {finding['type']}: {finding.get('keyword', '')}")
            print(f"      File: {finding['file']}")
            print()
        print("\n⚠️  Please review these findings with export compliance team")
    else:
        print("\n✓ No export-controlled patterns detected")

    print("=" * 60)

    return all_findings


if __name__ == "__main__":
    scan_repository()
