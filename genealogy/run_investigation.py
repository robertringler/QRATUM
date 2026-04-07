#!/usr/bin/env python3
"""
Execute genealogical proof investigation for Robert Ringler Jr.

Multi-agent orchestration system for professional-grade genealogical research.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genealogy.orchestrator import ChiefResearchOrchestrator


def main():
    """Execute complete genealogical research investigation."""

    print("\n" + "=" * 80)
    print("QRATUM GENEALOGICAL RESEARCH ORCHESTRATION SYSTEM")
    print("Multi-Agent Investigation for Royal & Noble Ancestry Proof")
    print("=" * 80 + "\n")

    # Initialize orchestrator
    orchestrator = ChiefResearchOrchestrator(subject_name="Robert Ringler Jr.")

    # Execute research (parallel agent execution)
    results = orchestrator.execute_research(parallel=True)

    # Create outputs directory if it doesn't exist
    outputs_dir = Path(__file__).parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    # Export results in multiple formats
    print("\nExporting results...")

    json_path = outputs_dir / "genealogical_proof_investigation.json"
    orchestrator.export_results(str(json_path), format="json")
    print(f"  ✓ JSON report exported to: {json_path}")

    txt_path = outputs_dir / "genealogical_proof_investigation.txt"
    orchestrator.export_results(str(txt_path), format="txt")
    print(f"  ✓ Text report exported to: {txt_path}")

    html_path = outputs_dir / "genealogical_proof_investigation.html"
    orchestrator.export_results(str(html_path), format="html")
    print(f"  ✓ HTML report exported to: {html_path}")

    # Display summary
    print("\n" + "=" * 80)
    print("INVESTIGATION SUMMARY")
    print("=" * 80 + "\n")

    synthesis = results.get("synthesis", {})
    print(synthesis.get("executive_summary", "No summary available"))

    print("\n" + "=" * 80)
    print(f"Full reports available in: {outputs_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
