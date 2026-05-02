#!/usr/bin/env python3
"""
QRATUM Platform Integration Layer
Connects all QRATUM components with QUBIC visualization modules
"""

import json
import sys
from pathlib import Path

# QRATUM root
QRATUM_ROOT = Path("/workspaces/QRATUM")

# Platform components registry
PLATFORM_COMPONENTS = {
    "quasim": {
        "name": "QuASIM",
        "description": "Quantum-Inspired Autonomous Simulation Engine",
        "version": "2.0.0",
        "path": QRATUM_ROOT / "quasim",
        "modules": [
            "quasim.api",
            "quasim.opt",
            "quasim.sim",
            "quasim.hcal",
            "quasim.qc",
            "quasim.distributed",
        ],
        "data_feeds": ["quantum_state", "optimization_progress", "simulation_metrics"],
    },
    "xenon": {
        "name": "XENON",
        "description": "Quantum Bioinformatics Platform",
        "version": "5.0.0",
        "path": QRATUM_ROOT / "xenon",
        "modules": ["xenon.core", "xenon.genome", "xenon.protein", "xenon.cli"],
        "data_feeds": ["genome_sequence", "protein_fold", "mutation_analysis"],
    },
    "qubic": {
        "name": "QUBIC",
        "description": "Advanced Visualization Suite",
        "version": "2.0.0",
        "path": QRATUM_ROOT / "qubic",
        "modules": ["qubic.visualization", "qubic.modules", "qubic.dashboard"],
        "data_feeds": ["render_metrics", "visualization_state"],
    },
    "autonomous": {
        "name": "Autonomous Systems",
        "description": "RL Optimization Platform",
        "version": "1.0.0",
        "path": QRATUM_ROOT / "autonomous_systems_platform",
        "modules": [
            "autonomous.services.backend",
            "autonomous.services.frontend",
            "autonomous.infra",
        ],
        "data_feeds": ["rl_state", "policy_updates", "reward_signals"],
    },
    "compliance": {
        "name": "Compliance Framework",
        "description": "DO-178C / NIST / CMMC Compliance",
        "version": "1.0.0",
        "path": QRATUM_ROOT / "compliance",
        "modules": ["compliance.scripts", "compliance.policies"],
        "data_feeds": ["compliance_status", "audit_logs"],
    },
}

# Module to Component Mapping
MODULE_COMPONENT_MAP = {
    # Quantum category -> QuASIM
    "quantum_state_viewer": "quasim",
    "qubit_simulator": "quasim",
    "entanglement_lab": "quasim",
    "quantum_gate_designer": "quasim",
    "decoherence_monitor": "quasim",
    "bell_state_analyzer": "quasim",
    "quantum_walk": "quasim",
    "grover_search": "quasim",
    "shor_factoring": "quasim",
    "vqe_optimizer": "quasim",
    # Bioinformatics category -> XENON
    "dna_sequencer": "xenon",
    "protein_folder": "xenon",
    "genome_browser": "xenon",
    "phylo_tree": "xenon",
    "mutation_tracker": "xenon",
    "rna_structure": "xenon",
    "crispr_designer": "xenon",
    "metabolic_pathway": "xenon",
    "expression_heatmap": "xenon",
    "variant_caller": "xenon",
    # Neural category -> Autonomous
    "neural_network": "autonomous",
    "activation_maps": "autonomous",
    "gradient_flow": "autonomous",
    "attention_viz": "autonomous",
    "embedding_space": "autonomous",
    "loss_landscape": "autonomous",
    "gan_generator": "autonomous",
    "reinforcement_env": "autonomous",
    "autoencoder_latent": "autonomous",
    "transformer_layers": "autonomous",
    # Physics category -> QuASIM
    "particle_collider": "quasim",
    "wave_equation": "quasim",
    "electromagnetic_field": "quasim",
    "fluid_dynamics": "quasim",
    "pendulum_chaos": "quasim",
    "orbital_mechanics": "quasim",
    "black_hole": "quasim",
    "string_vibrations": "quasim",
    "plasma_dynamics": "quasim",
    "superconductor": "quasim",
    # Chemistry category -> QuASIM
    "molecular_viewer": "quasim",
    "reaction_kinetics": "quasim",
    "periodic_explorer": "quasim",
    "orbital_viewer": "quasim",
    "bond_analyzer": "quasim",
    "spectroscopy": "quasim",
    "catalyst_sim": "quasim",
    "crystal_structure": "quasim",
    "solubility_map": "quasim",
    "thermodynamics": "quasim",
    # Crypto category -> QuASIM (quantum crypto)
    "hash_visualizer": "quasim",
    "blockchain_explorer": "quasim",
    "encryption_flow": "quasim",
    "key_exchange": "quasim",
    "merkle_tree": "quasim",
    "zero_knowledge": "quasim",
    "signature_verify": "quasim",
    "entropy_analyzer": "quasim",
    "cipher_wheel": "quasim",
    "quantum_crypto": "quasim",
    # Network category -> Autonomous
    "network_topology": "autonomous",
    "packet_flow": "autonomous",
    "firewall_monitor": "compliance",
    "latency_map": "autonomous",
    "ddos_simulator": "compliance",
    "protocol_analyzer": "autonomous",
    "dns_resolver": "autonomous",
    "load_balancer": "autonomous",
    "vpn_tunnel": "compliance",
    "mesh_network": "autonomous",
    # Space category -> QuASIM
    "solar_system": "quasim",
    "star_map": "quasim",
    "galaxy_merger": "quasim",
    "exoplanet_finder": "quasim",
    "asteroid_tracker": "quasim",
    "cosmic_web": "quasim",
    "pulsar_timing": "quasim",
    "redshift_map": "quasim",
    "dark_matter": "quasim",
    "gravitational_waves": "quasim",
    # Financial category -> Autonomous
    "market_depth": "autonomous",
    "volatility_surface": "autonomous",
    "correlation_matrix": "autonomous",
    "risk_dashboard": "compliance",
    "monte_carlo_sim": "quasim",
    "candlestick_chart": "autonomous",
    "sentiment_tracker": "autonomous",
    "flow_analyzer": "autonomous",
    "yield_curve": "autonomous",
    "portfolio_optimizer": "autonomous",
    # Data category -> QUBIC
    "scatter_3d": "qubic",
    "parallel_coords": "qubic",
    "treemap_viz": "qubic",
    "sankey_flow": "qubic",
    "chord_diagram": "qubic",
    "force_graph": "qubic",
    "sunburst_chart": "qubic",
    "radar_chart": "qubic",
    "stream_graph": "qubic",
    "hexbin_map": "qubic",
}

# API Endpoints registry
API_ENDPOINTS = {
    "platform": {
        "base": "http://localhost:9000",
        "endpoints": {
            "status": "/api/status",
            "metrics": "/api/metrics",
            "modules": "/api/modules",
            "stream": "/api/stream/{type}",
        },
    },
    "quasim": {
        "base": "http://localhost:8000",
        "endpoints": {"health": "/health", "kernel": "/kernel", "metrics": "/metrics"},
    },
    "xenon": {
        "base": "http://localhost:8099",
        "endpoints": {"simulation": "/simulation", "genome": "/genome", "status": "/status"},
    },
    "qubic": {
        "base": "http://localhost:8100",
        "endpoints": {"dashboard": "/dashboard", "modules": "/modules"},
    },
}


class QRATUMIntegration:
    """Integration layer for QRATUM platform components"""

    def __init__(self):
        self.components = PLATFORM_COMPONENTS
        self.module_map = MODULE_COMPONENT_MAP
        self.api_endpoints = API_ENDPOINTS

    def get_component_for_module(self, module_id: str) -> dict:
        """Get the backend component for a visualization module"""
        component_key = self.module_map.get(module_id, "qubic")
        return self.components.get(component_key, {})

    def get_api_endpoint(self, component: str, endpoint: str) -> str:
        """Get full API endpoint URL"""
        if component in self.api_endpoints:
            base = self.api_endpoints[component]["base"]
            path = self.api_endpoints[component]["endpoints"].get(endpoint, "")
            return f"{base}{path}"
        return ""

    def get_module_data_source(self, module_id: str) -> dict:
        """Get data source configuration for a module"""
        component_key = self.module_map.get(module_id, "qubic")
        component = self.components.get(component_key, {})

        return {
            "component": component_key,
            "component_name": component.get("name", "Unknown"),
            "data_feeds": component.get("data_feeds", []),
            "api_base": self.api_endpoints.get(component_key, {}).get("base", ""),
        }

    def generate_integration_config(self) -> dict:
        """Generate full integration configuration"""
        config = {
            "platform": "QRATUM",
            "version": "2.0.0",
            "components": self.components,
            "module_mapping": self.module_map,
            "api_endpoints": self.api_endpoints,
            "total_modules": len(self.module_map),
            "component_counts": {},
        }

        # Count modules per component
        for module_id, component in self.module_map.items():
            config["component_counts"][component] = config["component_counts"].get(component, 0) + 1

        return config

    def export_integration_manifest(self, output_path: str = None):
        """Export integration manifest to JSON"""
        if output_path is None:
            output_path = QRATUM_ROOT / "qubic" / "integration_manifest.json"

        config = self.generate_integration_config()

        with open(output_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

        print(f"✅ Integration manifest exported to: {output_path}")
        return config


def update_modules_with_integration():
    """Update all QUBIC modules with integration endpoints"""
    integration = QRATUMIntegration()
    modules_dir = QRATUM_ROOT / "qubic" / "modules"

    # Load modules manifest
    manifest_path = modules_dir / "modules.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            modules = json.load(f)
    else:
        print("⚠️ modules.json not found")
        return

    # Update each module with integration info
    updated_modules = []
    for module in modules:
        module_id = module.get("id")
        if module_id:
            data_source = integration.get_module_data_source(module_id)
            module["integration"] = data_source
        updated_modules.append(module)

    # Save updated manifest
    with open(manifest_path, "w") as f:
        json.dump(updated_modules, f, indent=2)

    print(f"✅ Updated {len(updated_modules)} modules with integration info")

    # Export full integration manifest
    integration.export_integration_manifest()


def print_platform_summary():
    """Print platform integration summary"""
    integration = QRATUMIntegration()
    config = integration.generate_integration_config()

    print()
    print("=" * 70)
    print("🚀 QRATUM PLATFORM INTEGRATION SUMMARY")
    print("=" * 70)
    print()

    print("📦 COMPONENTS:")
    for key, comp in config["components"].items():
        print(f"  • {comp['name']} v{comp['version']}")
        print(f"    {comp['description']}")
        print(f"    Data Feeds: {', '.join(comp['data_feeds'])}")
        print()

    print("=" * 70)
    print()

    print("📊 MODULE DISTRIBUTION:")
    for comp, count in sorted(config["component_counts"].items(), key=lambda x: -x[1]):
        bar = "█" * (count // 2)
        print(f"  {comp:15} {count:3} modules {bar}")
    print()

    print(f"  Total: {config['total_modules']} modules")
    print()

    print("=" * 70)
    print()

    print("🌐 API ENDPOINTS:")
    for comp, api in config["api_endpoints"].items():
        print(f"  {comp}: {api['base']}")
        for name, path in api["endpoints"].items():
            print(f"    • {name}: {path}")
        print()

    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "update":
            update_modules_with_integration()
        elif sys.argv[1] == "export":
            integration = QRATUMIntegration()
            integration.export_integration_manifest()
    else:
        print_platform_summary()
        update_modules_with_integration()
