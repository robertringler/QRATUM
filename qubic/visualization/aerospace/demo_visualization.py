"""Demo script to showcase aerospace visualization capabilities.

Generates sample visualizations for documentation and verification.
"""

from __future__ import annotations

import numpy as np

from qubic.visualization.aerospace import (AerospaceVisualizer,
                                           AerospaceVizConfig, ComplianceMode)


def generate_all_demos():
    """Generate all demo visualizations."""
    print("=" * 70)
    print("QRATUM Aerospace Visualization Module - Demo")
    print("=" * 70)
    print()

    # Initialize visualizer with compliance mode
    config = AerospaceVizConfig(
        compliance_mode=ComplianceMode.DO178C_LEVEL_A,
        seed=42,
        enable_audit_log=True,
        dpi=150,
    )
    viz = AerospaceVisualizer(config)

    # Demo 1: Flight Trajectory with Vapor Trail
    print("1. Generating Flight Trajectory with Vapor Trail...")
    t = np.linspace(0, 4 * np.pi, 200)
    trajectory = np.column_stack([100 * np.cos(t), 100 * np.sin(t), 50 * t])
    velocity = np.gradient(trajectory, axis=0) * 10

    fig1 = viz.render_flight_trajectory(
        trajectory=trajectory,
        velocity=velocity,
        title="F-35 Spiral Climb Maneuver",
        output_path="demo_trajectory.png",
        show_vapor_trail=True,
    )
    if fig1:
        print("   ✓ Saved: demo_trajectory.png")

    # Demo 2: Airflow Streamlines
    print("\n2. Generating Airflow Streamlines...")
    nx, ny, nz = 20, 20, 20
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(-1, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Vortex flow pattern
    vx = -Y.flatten()
    vy = X.flatten()
    vz = 0.5 * np.sin(2 * np.pi * Z.flatten())
    velocity_field = np.column_stack([vx, vy, vz])

    fig2 = viz.render_airflow_streamlines(
        velocity_field=velocity_field,
        grid_shape=(nx, ny, nz),
        density=80,
        title="Wing Tip Vortex Airflow",
        output_path="demo_airflow.png",
    )
    if fig2:
        print("   ✓ Saved: demo_airflow.png")

    # Demo 3: FEM Mesh with Stress
    print("\n3. Generating FEM Mesh with Stress Analysis...")
    # Create a simple beam mesh
    n_nodes = 20
    nodes = np.array([[i * 0.5, 0, 0] for i in range(n_nodes)], dtype=float)

    # Create tetrahedral elements
    elements = []
    for i in range(n_nodes - 1):
        elements.append([i, i + 1, min(i + 2, n_nodes - 1), min(i + 3, n_nodes - 1)])
    elements = np.array(elements, dtype=int)

    # Generate stress tensor (simulated bending stress)
    stress = np.zeros((n_nodes, 6))
    for i in range(n_nodes):
        x = nodes[i, 0]
        stress[i, 0] = 100e6 * (x / 10) ** 2  # xx
        stress[i, 1] = 50e6 * (x / 10) ** 2  # yy
        stress[i, 2] = 30e6 * (x / 10) ** 2  # zz
        stress[i, 3] = 10e6 * (x / 10)  # xy
        stress[i, 4] = 5e6 * (x / 10)  # xz
        stress[i, 5] = 5e6 * (x / 10)  # yz

    fig3 = viz.render_fem_mesh(
        nodes=nodes,
        elements=elements,
        stress_tensor=stress,
        title="Wing Spar Stress Analysis",
        output_path="demo_fem_mesh.png",
        show_wireframe=True,
    )
    if fig3:
        print("   ✓ Saved: demo_fem_mesh.png")

    # Demo 4: Modal Analysis
    print("\n4. Generating Modal Analysis Visualization...")
    n_nodes = 30
    nodes = np.array([[i, 0, 0] for i in range(n_nodes)], dtype=float)

    # First three bending modes
    mode_shapes = []
    for mode in range(3):
        mode_shape = np.array(
            [[0, np.sin((mode + 1) * i * np.pi / (n_nodes - 1)), 0] for i in range(n_nodes)]
        )
        mode_shapes.append(mode_shape)

    eigenvectors = np.stack(mode_shapes, axis=1)  # Shape: (n_nodes, 3_modes, 3)
    eigenfrequencies = np.array([15.2, 42.8, 98.5])  # Hz

    fig4 = viz.render_modal_analysis(
        nodes=nodes,
        eigenvectors=eigenvectors,
        eigenfrequencies=eigenfrequencies,
        mode_index=0,
        amplitude_scale=2.0,
        output_path="demo_modal_analysis.png",
    )
    if fig4:
        print("   ✓ Saved: demo_modal_analysis.png")

    # Demo 5: Thermal Field
    print("\n5. Generating Thermal Field Visualization...")
    np.random.seed(42)
    n_points = 500
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    r = np.random.uniform(0.8, 1.2, n_points)

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    geometry = np.column_stack([x, y, z])

    # Temperature distribution (hot at equator, cooler at poles)
    temperature = 300 + 500 * np.sin(phi) ** 2

    fig5 = viz.render_thermal_field(
        temperature=temperature,
        geometry=geometry,
        title="Reentry Heating Profile",
        output_path="demo_thermal_field.png",
        colormap="hot",
        show_isotherms=False,
    )
    if fig5:
        print("   ✓ Saved: demo_thermal_field.png")

    # Demo 6: Heat Flux
    print("\n6. Generating Heat Flux Visualization...")
    n_points = 100
    theta = np.linspace(0, 2 * np.pi, n_points)
    geometry = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(n_points)])

    # Heat flux vectors (radial outward)
    heat_flux = geometry * 10000  # W/m^2
    surface_normals = geometry / (np.linalg.norm(geometry, axis=1, keepdims=True) + 1e-10)

    fig6 = viz.render_heat_flux(
        heat_flux=heat_flux,
        surface_normals=surface_normals,
        geometry=geometry,
        title="Heat Flux on Cylindrical Surface",
        output_path="demo_heat_flux.png",
    )
    if fig6:
        print("   ✓ Saved: demo_heat_flux.png")

    # Demo 7: Sensor FOV
    print("\n7. Generating Sensor FOV Visualization...")
    sensor_pos = np.array([0, 0, 100], dtype=float)
    sensor_dir = np.array([1, 0, -0.3], dtype=float)

    fig7 = viz.render_sensor_fov(
        sensor_position=sensor_pos,
        sensor_orientation=sensor_dir,
        fov_horizontal=90,
        fov_vertical=45,
        range_m=500,
        title="Forward LIDAR Field of View",
        output_path="demo_sensor_fov.png",
        cone_color="cyan",
        cone_alpha=0.4,
    )
    if fig7:
        print("   ✓ Saved: demo_sensor_fov.png")

    # Demo 8: Radar Cross Section
    print("\n8. Generating Radar Cross Section Plot...")
    n_points = 360
    theta = np.linspace(0, 2 * np.pi, n_points)
    geometry = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(n_points)])

    # RCS pattern (stealth aircraft has low RCS from front/back, higher from sides)
    rcs_db = -20 + 15 * np.sin(2 * theta) ** 2

    fig8 = viz.render_radar_cross_section(
        geometry=geometry,
        rcs_db=rcs_db,
        frequency_ghz=10.0,
        azimuth_range=(0, 360),
        title="F-35 Radar Cross Section @ 10 GHz",
        output_path="demo_rcs.png",
    )
    if fig8:
        print("   ✓ Saved: demo_rcs.png")

    # Export audit trail
    print("\n9. Exporting Audit Trail...")
    viz.export_audit_trail("demo_audit_trail.json")
    print("   ✓ Saved: demo_audit_trail.json")

    # Generate compliance report
    print("\n10. Generating Compliance Report...")
    report = viz.generate_compliance_report()
    print("\n   Compliance Report:")
    print(f"   - Mode: {report['compliance_mode']}")
    print(f"   - Total Frames: {report['total_frames']}")
    print(f"   - Seed: {report['seed']}")
    print(f"   - Avg Render Time: {report['render_time_stats']['average_ms']:.2f} ms")
    print(f"   - Config Hash: {report['config_hash'][:16]}...")

    print()
    print("=" * 70)
    print("Demo Complete! All visualizations generated successfully.")
    print("=" * 70)
    print()
    print("Generated Files:")
    print("  • demo_trajectory.png        - Flight trajectory with vapor trail")
    print("  • demo_airflow.png           - Airflow streamlines")
    print("  • demo_fem_mesh.png          - FEM mesh with stress")
    print("  • demo_modal_analysis.png    - Modal analysis eigenmode")
    print("  • demo_thermal_field.png     - Thermal field distribution")
    print("  • demo_heat_flux.png         - Heat flux vectors")
    print("  • demo_sensor_fov.png        - Sensor field of view")
    print("  • demo_rcs.png               - Radar cross section")
    print("  • demo_audit_trail.json      - Compliance audit trail")
    print()


if __name__ == "__main__":
    generate_all_demos()
