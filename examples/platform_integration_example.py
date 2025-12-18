#!/usr/bin/env python3
"""Example demonstrating QRATUM platform integration layer.

This example shows how to use the new platform integration API
while maintaining backwards compatibility with existing code.

Classification: UNCLASSIFIED // CUI
"""

from qratum import create_platform


def example_vqe_workflow():
    """Example VQE workflow with compliance hooks."""
    print("=" * 70)
    print("Example 1: VQE Workflow with DO-178C Compliance")
    print("=" * 70)

    # Initialize platform with DO-178C compliance
    platform = create_platform(
        quantum_backend="simulator",
        seed=42,
        do178c_enabled=True,
        audit_enabled=True,
        shots=1024,
    )

    # Run VQE workflow for H2 molecule
    print("\nRunning VQE for H2 molecule at bond length 0.735 Å...")
    result = platform.run_vqe(molecule="H2", bond_length=0.735, basis="sto3g")

    print(f"\n✓ VQE Results:")
    print(f"  Molecule: {result['molecule']}")
    print(f"  Bond Length: {result['bond_length']} Å")
    print(f"  Energy: {result['energy']:.6f} Hartree")
    print(f"  Execution ID: {result['execution_id']}")
    print(f"  Backend: {result['backend']}")
    print(f"  DO-178C Enabled: {result['compliance']['do178c_enabled']}")
    print()


def example_qaoa_workflow():
    """Example QAOA workflow for MaxCut problem."""
    print("=" * 70)
    print("Example 2: QAOA Workflow for MaxCut Problem")
    print("=" * 70)

    # Initialize platform
    platform = create_platform(
        quantum_backend="simulator",
        seed=42,
        do178c_enabled=True,
        audit_enabled=True,
    )

    # Define MaxCut problem (simple 4-node graph)
    problem_data = {"edges": [[0, 1], [1, 2], [2, 3], [3, 0]], "num_nodes": 4}

    print("\nRunning QAOA for MaxCut problem (4-node cycle graph)...")
    result = platform.run_qaoa(
        problem_type="maxcut", problem_data=problem_data, p_layers=3
    )

    print(f"\n✓ QAOA Results:")
    print(f"  Problem Type: {result['problem_type']}")
    print(f"  Solution: {result['solution']}")
    print(f"  Energy: {result['energy']:.3f}")
    print(f"  Execution ID: {result['execution_id']}")
    print(f"  P-Layers: {result['p_layers']}")
    print()


def example_backend_selection():
    """Example demonstrating intelligent backend selection."""
    print("=" * 70)
    print("Example 3: Intelligent Backend Selection")
    print("=" * 70)

    platform = create_platform(quantum_backend="simulator", seed=42)

    print("\nBackend selection based on problem size:")
    print("  2 qubits  → " + platform.select_backend("VQE", 2))
    print("  8 qubits  → " + platform.select_backend("VQE", 8))
    print("  15 qubits → " + platform.select_backend("VQE", 15))
    print("  25 qubits → " + platform.select_backend("VQE", 25))
    print()


def example_compliance_report():
    """Example compliance report generation."""
    print("=" * 70)
    print("Example 4: Compliance Report Generation")
    print("=" * 70)

    platform = create_platform(
        quantum_backend="simulator",
        seed=42,
        do178c_enabled=True,
        audit_enabled=True,
    )

    # Run a workflow
    print("\nRunning workflow to generate audit trail...")
    platform.run_vqe(molecule="H2", bond_length=0.735)

    # Generate compliance report
    report_path = "/tmp/qratum_compliance_example.json"
    print(f"Generating compliance report: {report_path}")
    report = platform.generate_compliance_report(report_path)

    print(f"\n✓ Compliance Report Generated:")
    print(f"  Report ID: {report['report_id']}")
    print(f"  Generated At: {report['generated_at']}")
    print(f"  Platform Version: {report['platform_version']}")
    print(f"  DO-178C Enabled: {report['configuration']['do178c_enabled']}")
    print(f"  Audit Trail Entries: {len(report['audit_trail'])}")
    print()


def example_backwards_compatibility():
    """Example showing backwards compatibility with existing imports."""
    print("=" * 70)
    print("Example 5: Backwards Compatibility")
    print("=" * 70)

    # Old imports still work
    from qratum import QRATUMConfig, Simulator

    print("\n✓ Existing imports work:")
    print(f"  QRATUMConfig: {QRATUMConfig}")
    print(f"  Simulator: {Simulator}")

    # New imports also work
    from qratum import PlatformConfig, QRATUMPlatform

    print("\n✓ New platform imports work:")
    print(f"  PlatformConfig: {PlatformConfig}")
    print(f"  QRATUMPlatform: {QRATUMPlatform}")

    print("\n✓ Both old and new APIs coexist without conflicts")
    print()


if __name__ == "__main__":
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║       QRATUM Platform Integration Examples                           ║")
    print("║       Version 2.0 - Task 1 Implementation                            ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print("\n")

    # Run all examples
    example_vqe_workflow()
    example_qaoa_workflow()
    example_backend_selection()
    example_compliance_report()
    example_backwards_compatibility()

    print("=" * 70)
    print("✓ All examples completed successfully!")
    print("=" * 70)
    print()
