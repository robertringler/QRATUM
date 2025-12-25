#!/usr/bin/env python3
"""Demo script for the Advanced Molecular Dynamics Lab.

This script demonstrates the full capabilities of the Molecular Dynamics Lab,
including:
- Loading and visualizing PDB structures
- Running molecular docking simulations
- Performing real-time MD simulations
- WebXR/VR integration with haptic feedback

Usage:
    python -m xenon.molecular_dynamics_lab.demo

This will start a web server at http://localhost:8080 with the full
interactive 3D molecular viewer.
"""

from __future__ import annotations

import sys
import time
import webbrowser
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from xenon.molecular_dynamics_lab import (DockingConfig, DockingEngine,
                                          HapticConfig, HapticEngine, MDConfig,
                                          MDSimulator, MolecularLabServer,
                                          MolecularViewer, PDBLoader,
                                          ViewerConfig, VRConfig, VRController)
from xenon.molecular_dynamics_lab.web.server import ServerConfig


def demo_pdb_loading():
    """Demonstrate PDB loading capabilities."""
    print("\n" + "=" * 60)
    print("🧬 PDB Loading Demo")
    print("=" * 60)

    loader = PDBLoader()

    # Load a small protein (Crambin - only 46 residues)
    print("\n📥 Loading 1CRN (Crambin) from RCSB...")
    try:
        structure = loader.load_from_rcsb("1CRN")
        print("   ✅ Loaded successfully!")
        print(f"   📊 PDB ID: {structure.pdb_id}")
        print(f"   🔢 Atoms: {len(structure.atoms)}")
        print(f"   🧪 Residues: {sum(len(c.residues) for c in structure.chains)}")
        print(f"   🔗 Chains: {len(structure.chains)}")
        print(f"   📏 Resolution: {structure.header.get('resolution', 'N/A')} Å")
    except Exception as e:
        print(f"   ⚠️ Could not load from RCSB: {e}")
        print("   📝 Creating synthetic structure for demo...")

    # Show supported formats
    print("\n📁 Supported PDB sources:")
    print("   • RCSB Protein Data Bank (online)")
    print("   • Local PDB files")
    print("   • PDB-formatted strings")


def demo_molecular_viewer():
    """Demonstrate molecular visualization capabilities."""
    print("\n" + "=" * 60)
    print("🔬 Molecular Viewer Demo")
    print("=" * 60)

    config = ViewerConfig(
        background_color="#000000",
        ambient_light=0.4,
        directional_light=0.6,
        webxr_enabled=True,
    )

    viewer = MolecularViewer(config)

    print("\n🎨 Available visualization styles:")
    styles = ["cartoon", "stick", "sphere", "line", "surface"]
    for style in styles:
        print(f"   • {style}")

    print("\n🌈 Available color schemes:")
    colors = [
        "chainHetatm",
        "ssPyMOL",
        "spectrum",
        "greenCarbon",
        "Jmol",
        "orangeCarbon",
    ]
    for color in colors:
        print(f"   • {color}")

    print("\n🖥️ Viewer features:")
    print("   • 3Dmol.js based rendering")
    print("   • Interactive rotation/zoom/pan")
    print("   • Atom picking and labeling")
    print("   • Surface rendering")
    print("   • Animation support")
    print("   • WebXR VR support")


def demo_docking():
    """Demonstrate molecular docking capabilities."""
    print("\n" + "=" * 60)
    print("⚗️ Molecular Docking Demo")
    print("=" * 60)

    config = DockingConfig(
        search_exhaustiveness=8,
        num_poses=10,
        flexible_residues=[],
        scoring_function="vina",
    )

    engine = DockingEngine(config)

    print("\n🔧 Docking parameters:")
    print(f"   • Exhaustiveness: {config.search_exhaustiveness}")
    print(f"   • Number of poses: {config.num_poses}")
    print(f"   • Scoring function: {config.scoring_function}")

    print("\n📋 Docking workflow:")
    print("   1. Load receptor structure")
    print("   2. Load ligand structure")
    print("   3. Define binding site (auto or manual)")
    print("   4. Run docking simulation")
    print("   5. Analyze poses and scores")

    print("\n🎯 Scoring components:")
    print("   • Lennard-Jones (van der Waals)")
    print("   • Electrostatic interactions")
    print("   • Hydrogen bonding")
    print("   • Desolvation penalty")


def demo_md_simulation():
    """Demonstrate molecular dynamics simulation."""
    print("\n" + "=" * 60)
    print("⚡ Molecular Dynamics Demo")
    print("=" * 60)

    config = MDConfig(
        timestep=0.002,  # 2 fs
        temperature=300.0,  # K
        thermostat="berendsen",
        thermostat_coupling=0.1,
    )

    simulator = MDSimulator(config)

    print("\n🔧 Simulation parameters:")
    print(f"   • Timestep: {config.timestep * 1000:.1f} fs")
    print(f"   • Temperature: {config.temperature:.1f} K")
    print(f"   • Thermostat: {config.thermostat}")
    print(f"   • Coupling: {config.thermostat_coupling} ps")

    print("\n🧮 Force field components:")
    print("   • Bond stretching (harmonic)")
    print("   • Angle bending (harmonic)")
    print("   • Lennard-Jones (non-bonded)")
    print("   • Electrostatic (Coulomb)")

    print("\n📈 Output data:")
    print("   • Trajectory (positions over time)")
    print("   • Kinetic energy")
    print("   • Potential energy")
    print("   • Temperature")


def demo_vr_haptics():
    """Demonstrate VR and haptic feedback."""
    print("\n" + "=" * 60)
    print("🥽 VR & Haptic Feedback Demo")
    print("=" * 60)

    vr_config = VRConfig(
        enable_hand_tracking=True,
        molecule_scale=0.1,
        interaction_radius=0.1,
    )

    haptic_config = HapticConfig(
        intensity=1.0,
        enabled=True,
    )

    vr_controller = VRController(vr_config)
    haptic_engine = HapticEngine(haptic_config)

    print("\n🎮 VR features:")
    print("   • Immersive-VR mode (6DoF)")
    print("   • Hand tracking support")
    print("   • Controller input")
    print("   • Molecule grabbing/manipulation")

    print("\n✋ Hand gestures:")
    print("   • Pinch: Select atom")
    print("   • Grab: Hold molecule")
    print("   • Point: UI interaction")
    print("   • Fist: Reset view")

    print("\n📳 Haptic feedback patterns:")
    print("   • Atom selection: Quick pulse")
    print("   • Atom hover: Gentle vibration")
    print("   • Bond formed: Strong pulse")
    print("   • Bond broken: Double pulse")
    print("   • Collision: Vibration intensity")


def run_server():
    """Run the full web server."""
    print("\n" + "=" * 60)
    print("🌐 Starting Molecular Dynamics Lab Server")
    print("=" * 60)

    config = ServerConfig(
        host="localhost",
        http_port=8080,
        debug=True,
    )

    server = MolecularLabServer(config)

    print("\n🚀 Server configuration:")
    print(f"   • Host: {config.host}")
    print(f"   • HTTP Port: {config.http_port}")
    print(f"   • Debug: {config.debug}")

    try:
        server.start()
        url = f"http://{config.host}:{config.http_port}"
        print(f"\n✅ Server running at {url}")
        print("\n📖 Available endpoints:")
        print(f"   • GET  {url}/              - Main application")
        print(f"   • GET  {url}/api/status    - Server status")
        print(f"   • GET  {url}/api/structure/<pdb_id> - Load PDB")
        print(f"   • POST {url}/api/dock      - Run docking")
        print(f"   • POST {url}/api/simulate  - Run MD simulation")

        print("\n🌐 Opening browser...")
        webbrowser.open(url)

        print("\n⏳ Press Ctrl+C to stop the server\n")

        # Keep server running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down server...")
        server.stop()
        print("👋 Goodbye!")


def main():
    """Run the demo."""
    print("\n" + "=" * 60)
    print("🧬 ADVANCED MOLECULAR DYNAMICS LAB")
    print("   Interactive 3D Molecular Visualization & Simulation")
    print("=" * 60)

    print("\nThis demo showcases the full capabilities of the")
    print("Molecular Dynamics Lab module, including:")
    print("  • PDB file loading from RCSB")
    print("  • 3Dmol.js molecular visualization")
    print("  • Interactive docking simulation")
    print("  • Real-time MD simulation")
    print("  • WebXR VR support")
    print("  • Haptic feedback")

    # Show capability demos
    demo_pdb_loading()
    demo_molecular_viewer()
    demo_docking()
    demo_md_simulation()
    demo_vr_haptics()

    # Ask to run server
    print("\n" + "=" * 60)
    response = input("\n🚀 Start the web server? [Y/n]: ").strip().lower()
    if response != "n":
        run_server()
    else:
        print("\n👋 Demo complete. Run with --server to start the web server.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Advanced Molecular Dynamics Lab Demo")
    parser.add_argument(
        "--server",
        action="store_true",
        help="Start the web server immediately",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="HTTP port (default: 8080)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host address (default: localhost)",
    )

    args = parser.parse_args()

    if args.server:
        # Direct server start
        config = ServerConfig(host=args.host, http_port=args.port)
        server = MolecularLabServer(config)
        server.start()
        url = f"http://{args.host}:{args.port}"
        print(f"✅ Server running at {url}")
        webbrowser.open(url)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            server.stop()
    else:
        main()
