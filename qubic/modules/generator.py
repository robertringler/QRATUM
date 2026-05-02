#!/usr/bin/env python3
"""
QUBIC Module Generator - Creates 100 interactive visualization modules
with a unified centralized dashboard launcher.
"""

import json
import os

# Base directory for modules
BASE_DIR = "/workspaces/QRATUM/qubic/modules"

# Module categories and their modules
MODULES = {
    "quantum": [
        (
            "quantum_state_viewer",
            "Quantum State Viewer",
            "Visualize quantum state vectors and superposition",
            "#00f5ff",
            "⚛️",
        ),
        (
            "qubit_simulator",
            "Qubit Simulator",
            "Interactive qubit manipulation and measurement",
            "#ff00ff",
            "🔮",
        ),
        (
            "entanglement_lab",
            "Entanglement Lab",
            "Explore quantum entanglement between particles",
            "#00ff88",
            "🔗",
        ),
        (
            "quantum_gate_designer",
            "Gate Designer",
            "Design and test quantum gate circuits",
            "#ffaa00",
            "🚪",
        ),
        (
            "decoherence_monitor",
            "Decoherence Monitor",
            "Track quantum decoherence over time",
            "#ff3366",
            "📉",
        ),
        (
            "bell_state_analyzer",
            "Bell State Analyzer",
            "Analyze Bell states and violations",
            "#9933ff",
            "🔔",
        ),
        ("quantum_walk", "Quantum Walk", "Visualize quantum random walks", "#33ffcc", "🚶"),
        (
            "grover_search",
            "Grover Search",
            "Interactive Grover's algorithm visualization",
            "#ff9933",
            "🔍",
        ),
        (
            "shor_factoring",
            "Shor Factoring",
            "Shor's algorithm for integer factorization",
            "#66ff33",
            "🔢",
        ),
        (
            "vqe_optimizer",
            "VQE Optimizer",
            "Variational Quantum Eigensolver optimization",
            "#ff66cc",
            "📈",
        ),
    ],
    "bioinformatics": [
        ("dna_sequencer", "DNA Sequencer", "Real-time DNA sequence analysis", "#00ff88", "🧬"),
        ("protein_folder", "Protein Folder", "3D protein folding visualization", "#ff6633", "🔬"),
        ("genome_browser", "Genome Browser", "Interactive genome navigation", "#33ccff", "📊"),
        (
            "phylo_tree",
            "Phylogenetic Tree",
            "Evolutionary relationship visualization",
            "#99ff33",
            "🌳",
        ),
        (
            "mutation_tracker",
            "Mutation Tracker",
            "Track genetic mutations in real-time",
            "#ff3399",
            "🦠",
        ),
        ("rna_structure", "RNA Structure", "RNA secondary structure prediction", "#ffcc00", "🔄"),
        ("crispr_designer", "CRISPR Designer", "Design CRISPR gene edits", "#00ffcc", "✂️"),
        (
            "metabolic_pathway",
            "Metabolic Pathway",
            "Metabolic network visualization",
            "#ff9966",
            "🔀",
        ),
        (
            "expression_heatmap",
            "Expression Heatmap",
            "Gene expression heatmap analysis",
            "#cc33ff",
            "🌡️",
        ),
        ("variant_caller", "Variant Caller", "Identify genetic variants", "#66ffcc", "📋"),
    ],
    "neural": [
        (
            "neural_network",
            "Neural Network",
            "Deep learning architecture visualization",
            "#ff00ff",
            "🧠",
        ),
        ("activation_maps", "Activation Maps", "CNN activation map visualization", "#00f5ff", "🗺️"),
        (
            "gradient_flow",
            "Gradient Flow",
            "Backpropagation gradient visualization",
            "#ffaa00",
            "➡️",
        ),
        ("attention_viz", "Attention Viz", "Transformer attention visualization", "#00ff88", "👁️"),
        (
            "embedding_space",
            "Embedding Space",
            "High-dimensional embedding projection",
            "#ff3366",
            "🌐",
        ),
        ("loss_landscape", "Loss Landscape", "3D loss function landscape", "#9933ff", "⛰️"),
        (
            "gan_generator",
            "GAN Generator",
            "Generative adversarial network output",
            "#33ffcc",
            "🎨",
        ),
        (
            "reinforcement_env",
            "RL Environment",
            "Reinforcement learning environment",
            "#ff9933",
            "🎮",
        ),
        ("autoencoder_latent", "Autoencoder Latent", "Latent space visualization", "#66ff33", "🔐"),
        (
            "transformer_layers",
            "Transformer Layers",
            "Layer-by-layer transformer analysis",
            "#ff66cc",
            "📚",
        ),
    ],
    "physics": [
        (
            "particle_collider",
            "Particle Collider",
            "High-energy particle collision simulation",
            "#ff3366",
            "💥",
        ),
        ("wave_equation", "Wave Equation", "2D wave equation visualization", "#00f5ff", "🌊"),
        (
            "electromagnetic_field",
            "EM Field",
            "Electromagnetic field visualization",
            "#ffaa00",
            "⚡",
        ),
        ("fluid_dynamics", "Fluid Dynamics", "Navier-Stokes fluid simulation", "#00ff88", "💧"),
        ("pendulum_chaos", "Pendulum Chaos", "Double pendulum chaotic motion", "#ff00ff", "🔄"),
        (
            "orbital_mechanics",
            "Orbital Mechanics",
            "Gravitational orbit simulation",
            "#33ccff",
            "🌍",
        ),
        ("black_hole", "Black Hole", "Black hole gravitational lensing", "#9933ff", "🕳️"),
        (
            "string_vibrations",
            "String Vibrations",
            "String theory vibration modes",
            "#ff9933",
            "🎸",
        ),
        (
            "plasma_dynamics",
            "Plasma Dynamics",
            "Magnetohydrodynamic plasma simulation",
            "#66ff33",
            "🔥",
        ),
        ("superconductor", "Superconductor", "Superconductivity visualization", "#00ffcc", "❄️"),
    ],
    "chemistry": [
        (
            "molecular_viewer",
            "Molecular Viewer",
            "3D molecular structure visualization",
            "#00ff88",
            "⚗️",
        ),
        (
            "reaction_kinetics",
            "Reaction Kinetics",
            "Chemical reaction rate visualization",
            "#ff6633",
            "⏱️",
        ),
        ("periodic_explorer", "Periodic Explorer", "Interactive periodic table", "#33ccff", "📋"),
        ("orbital_viewer", "Orbital Viewer", "Atomic orbital visualization", "#9933ff", "🔵"),
        ("bond_analyzer", "Bond Analyzer", "Chemical bond strength analysis", "#ffcc00", "🔗"),
        ("spectroscopy", "Spectroscopy", "Molecular spectroscopy simulation", "#ff3399", "🌈"),
        ("catalyst_sim", "Catalyst Sim", "Catalytic reaction simulation", "#00ffcc", "⚡"),
        (
            "crystal_structure",
            "Crystal Structure",
            "Crystal lattice visualization",
            "#ff9966",
            "💎",
        ),
        ("solubility_map", "Solubility Map", "Solubility phase diagram", "#66ffcc", "💧"),
        ("thermodynamics", "Thermodynamics", "Thermodynamic process visualization", "#cc33ff", "🌡️"),
    ],
    "crypto": [
        ("hash_visualizer", "Hash Visualizer", "Cryptographic hash visualization", "#ff3366", "🔐"),
        (
            "blockchain_explorer",
            "Blockchain Explorer",
            "Blockchain network visualization",
            "#ffaa00",
            "⛓️",
        ),
        (
            "encryption_flow",
            "Encryption Flow",
            "Encryption algorithm visualization",
            "#00f5ff",
            "🔒",
        ),
        ("key_exchange", "Key Exchange", "Diffie-Hellman key exchange viz", "#00ff88", "🔑"),
        ("merkle_tree", "Merkle Tree", "Merkle tree structure visualization", "#ff00ff", "🌲"),
        ("zero_knowledge", "Zero Knowledge", "Zero-knowledge proof visualization", "#9933ff", "❓"),
        ("signature_verify", "Signature Verify", "Digital signature verification", "#33ffcc", "✍️"),
        (
            "entropy_analyzer",
            "Entropy Analyzer",
            "Randomness and entropy analysis",
            "#ff9933",
            "🎲",
        ),
        ("cipher_wheel", "Cipher Wheel", "Classical cipher visualization", "#66ff33", "⚙️"),
        ("quantum_crypto", "Quantum Crypto", "Quantum cryptography simulation", "#ff66cc", "🔮"),
    ],
    "network": [
        ("network_topology", "Network Topology", "Network graph visualization", "#00f5ff", "🕸️"),
        ("packet_flow", "Packet Flow", "Network packet flow visualization", "#ff00ff", "📦"),
        ("firewall_monitor", "Firewall Monitor", "Firewall rule visualization", "#ffaa00", "🛡️"),
        ("latency_map", "Latency Map", "Global network latency heatmap", "#00ff88", "🌍"),
        ("ddos_simulator", "DDoS Simulator", "DDoS attack visualization", "#ff3366", "🎯"),
        ("protocol_analyzer", "Protocol Analyzer", "Network protocol analysis", "#9933ff", "📡"),
        ("dns_resolver", "DNS Resolver", "DNS resolution visualization", "#33ffcc", "🔍"),
        ("load_balancer", "Load Balancer", "Load distribution visualization", "#ff9933", "⚖️"),
        ("vpn_tunnel", "VPN Tunnel", "VPN tunnel visualization", "#66ff33", "🚇"),
        ("mesh_network", "Mesh Network", "Mesh network topology", "#ff66cc", "🔷"),
    ],
    "space": [
        ("solar_system", "Solar System", "Interactive solar system model", "#ffaa00", "☀️"),
        ("star_map", "Star Map", "3D star catalog visualization", "#00f5ff", "⭐"),
        ("galaxy_merger", "Galaxy Merger", "Galaxy collision simulation", "#ff00ff", "🌌"),
        ("exoplanet_finder", "Exoplanet Finder", "Exoplanet transit detection", "#00ff88", "🔭"),
        ("asteroid_tracker", "Asteroid Tracker", "Near-Earth asteroid tracking", "#ff3366", "☄️"),
        ("cosmic_web", "Cosmic Web", "Large-scale structure visualization", "#9933ff", "🕸️"),
        ("pulsar_timing", "Pulsar Timing", "Pulsar signal analysis", "#33ffcc", "📻"),
        ("redshift_map", "Redshift Map", "Cosmological redshift visualization", "#ff9933", "🔴"),
        ("dark_matter", "Dark Matter", "Dark matter distribution map", "#666699", "🌑"),
        (
            "gravitational_waves",
            "Gravitational Waves",
            "Gravitational wave detection",
            "#66ff33",
            "〰️",
        ),
    ],
    "financial": [
        ("market_depth", "Market Depth", "Order book depth visualization", "#00ff88", "📊"),
        ("volatility_surface", "Volatility Surface", "Options volatility surface", "#ff3366", "📈"),
        ("correlation_matrix", "Correlation Matrix", "Asset correlation heatmap", "#00f5ff", "🔢"),
        ("risk_dashboard", "Risk Dashboard", "Portfolio risk metrics", "#ffaa00", "⚠️"),
        ("monte_carlo_sim", "Monte Carlo Sim", "Monte Carlo price simulation", "#ff00ff", "🎲"),
        ("candlestick_chart", "Candlestick Chart", "Advanced candlestick analysis", "#9933ff", "🕯️"),
        ("sentiment_tracker", "Sentiment Tracker", "Market sentiment analysis", "#33ffcc", "😊"),
        ("flow_analyzer", "Flow Analyzer", "Money flow visualization", "#ff9933", "💰"),
        ("yield_curve", "Yield Curve", "Interest rate yield curve", "#66ff33", "📉"),
        (
            "portfolio_optimizer",
            "Portfolio Optimizer",
            "Efficient frontier visualization",
            "#ff66cc",
            "🎯",
        ),
    ],
    "data": [
        ("scatter_3d", "3D Scatter", "3D scatter plot visualization", "#00f5ff", "📍"),
        ("parallel_coords", "Parallel Coords", "Parallel coordinates plot", "#ff00ff", "📏"),
        ("treemap_viz", "Treemap", "Hierarchical treemap visualization", "#00ff88", "🗺️"),
        ("sankey_flow", "Sankey Flow", "Sankey diagram flow visualization", "#ffaa00", "➡️"),
        ("chord_diagram", "Chord Diagram", "Relationship chord diagram", "#ff3366", "🔄"),
        ("force_graph", "Force Graph", "Force-directed graph layout", "#9933ff", "🕸️"),
        ("sunburst_chart", "Sunburst Chart", "Hierarchical sunburst chart", "#33ffcc", "☀️"),
        ("radar_chart", "Radar Chart", "Multi-dimensional radar chart", "#ff9933", "📡"),
        ("stream_graph", "Stream Graph", "Time-series stream graph", "#66ff33", "🌊"),
        ("hexbin_map", "Hexbin Map", "Hexagonal binning heatmap", "#ff66cc", "🔷"),
    ],
}

# HTML template for each module - ENHANCED VERSION
MODULE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} | QUBIC</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/EffectComposer.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/RenderPass.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/UnrealBloomPass.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/ShaderPass.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/shaders/LuminosityHighPassShader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/shaders/CopyShader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Share Tech Mono', monospace;
            background: linear-gradient(135deg, #000010 0%, #001020 50%, #000010 100%);
            color: {color};
            min-height: 100vh;
            overflow: hidden;
        }}
        
        .header {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            padding: 15px 30px;
            background: rgba(0,0,20,0.95);
            border-bottom: 1px solid {color}40;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 100;
            backdrop-filter: blur(10px);
        }}
        
        .title {{
            font-family: 'Orbitron', sans-serif;
            font-size: 24px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .icon {{ font-size: 32px; }}
        
        .header-controls {{
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        
        .header-btn {{
            padding: 8px 16px;
            background: {color}15;
            border: 1px solid {color}40;
            color: {color};
            font-family: 'Share Tech Mono', monospace;
            font-size: 12px;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.3s;
            text-decoration: none;
        }}
        
        .header-btn:hover {{
            background: {color};
            color: #000;
        }}
        
        #canvas-container {{
            position: fixed;
            top: 65px;
            left: 0;
            right: 0;
            bottom: 0;
        }}
        
        /* Settings Panel */
        .settings-panel {{
            position: fixed;
            top: 80px;
            left: 20px;
            width: 320px;
            max-height: calc(100vh - 180px);
            overflow-y: auto;
            background: rgba(0,0,20,0.95);
            border: 1px solid {color}30;
            border-radius: 12px;
            backdrop-filter: blur(10px);
            z-index: 50;
        }}
        
        .panel-section {{
            padding: 15px;
            border-bottom: 1px solid {color}15;
        }}
        
        .panel-section:last-child {{ border-bottom: none; }}
        
        .section-header {{
            font-family: 'Orbitron', sans-serif;
            font-size: 11px;
            color: {color};
            letter-spacing: 2px;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .section-header::before {{
            content: '';
            width: 3px;
            height: 12px;
            background: {color};
            border-radius: 2px;
        }}
        
        .slider-group {{
            margin-bottom: 12px;
        }}
        
        .slider-label {{
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            margin-bottom: 5px;
            color: #888;
        }}
        
        .slider-value {{
            color: {color};
            font-family: 'Orbitron', sans-serif;
        }}
        
        .slider {{
            width: 100%;
            height: 4px;
            -webkit-appearance: none;
            background: {color}20;
            border-radius: 2px;
            outline: none;
        }}
        
        .slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 14px;
            height: 14px;
            background: {color};
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 0 10px {color}80;
        }}
        
        .toggle-group {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
        }}
        
        .toggle-label {{ font-size: 12px; color: #aaa; }}
        
        .toggle {{
            width: 40px;
            height: 20px;
            background: #333;
            border-radius: 10px;
            position: relative;
            cursor: pointer;
            transition: background 0.3s;
        }}
        
        .toggle.active {{ background: {color}; }}
        
        .toggle::after {{
            content: '';
            position: absolute;
            width: 16px;
            height: 16px;
            background: #fff;
            border-radius: 50%;
            top: 2px;
            left: 2px;
            transition: left 0.3s;
        }}
        
        .toggle.active::after {{ left: 22px; }}
        
        .color-picker-group {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 8px;
        }}
        
        .color-swatch {{
            width: 28px;
            height: 28px;
            border-radius: 6px;
            cursor: pointer;
            border: 2px solid transparent;
            transition: all 0.3s;
        }}
        
        .color-swatch:hover, .color-swatch.active {{
            transform: scale(1.1);
            border-color: #fff;
        }}
        
        .btn-group {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}
        
        .mode-btn {{
            flex: 1;
            min-width: 60px;
            padding: 8px 12px;
            background: {color}15;
            border: 1px solid {color}30;
            color: {color};
            font-family: 'Share Tech Mono', monospace;
            font-size: 11px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.3s;
        }}
        
        .mode-btn:hover, .mode-btn.active {{
            background: {color};
            color: #000;
        }}
        
        /* Metrics Panel */
        .metrics-panel {{
            position: fixed;
            top: 80px;
            right: 20px;
            width: 280px;
            background: rgba(0,0,20,0.95);
            border: 1px solid {color}30;
            border-radius: 12px;
            backdrop-filter: blur(10px);
            z-index: 50;
        }}
        
        .metric-row {{
            display: flex;
            justify-content: space-between;
            padding: 10px 15px;
            border-bottom: 1px solid {color}10;
        }}
        
        .metric-row:last-child {{ border-bottom: none; }}
        
        .metric-label {{ color: #666; font-size: 11px; }}
        .metric-value {{
            font-family: 'Orbitron', sans-serif;
            font-size: 16px;
            color: {color};
        }}
        
        .metric-value.green {{ color: #00ff88; }}
        .metric-value.orange {{ color: #ffaa00; }}
        .metric-value.magenta {{ color: #ff00ff; }}
        .metric-value.red {{ color: #ff3366; }}
        
        /* Mini Chart */
        .mini-chart {{
            padding: 10px 15px;
            height: 80px;
        }}
        
        .mini-chart canvas {{
            width: 100% !important;
            height: 60px !important;
        }}
        
        /* Control Bar */
        .control-bar {{
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 8px;
            padding: 12px 20px;
            background: rgba(0,0,20,0.95);
            border: 1px solid {color}30;
            border-radius: 30px;
            backdrop-filter: blur(10px);
            z-index: 50;
        }}
        
        .ctrl-btn {{
            padding: 10px 18px;
            background: {color}15;
            border: 1px solid {color}40;
            color: {color};
            font-family: 'Share Tech Mono', monospace;
            font-size: 12px;
            border-radius: 15px;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        
        .ctrl-btn:hover, .ctrl-btn.active {{
            background: {color};
            color: #000;
        }}
        
        .ctrl-btn.danger {{
            border-color: #ff336660;
            color: #ff3366;
        }}
        
        .ctrl-btn.danger:hover {{
            background: #ff3366;
            color: #fff;
        }}
        
        /* Progress Bar */
        .progress-container {{
            position: fixed;
            bottom: 80px;
            left: 350px;
            right: 310px;
            height: 6px;
            background: {color}15;
            border-radius: 3px;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, {color}, #ff00ff);
            border-radius: 3px;
            width: 0%;
            transition: width 0.3s;
            box-shadow: 0 0 15px {color}60;
        }}
        
        .progress-label {{
            position: absolute;
            right: 0;
            top: -20px;
            font-size: 11px;
            color: #888;
        }}
        
        /* Timeline */
        .timeline {{
            position: fixed;
            bottom: 100px;
            left: 350px;
            right: 310px;
            display: flex;
            justify-content: space-between;
        }}
        
        .timeline-point {{
            width: 8px;
            height: 8px;
            background: {color}30;
            border-radius: 50%;
            transition: all 0.3s;
        }}
        
        .timeline-point.active {{
            background: {color};
            box-shadow: 0 0 10px {color};
        }}
        
        .timeline-point.complete {{
            background: #00ff88;
        }}
        
        /* Data Panel */
        .data-panel {{
            position: fixed;
            bottom: 80px;
            right: 20px;
            width: 280px;
            background: rgba(0,0,20,0.95);
            border: 1px solid {color}30;
            border-radius: 12px;
            backdrop-filter: blur(10px);
            z-index: 50;
        }}
        
        .data-stream {{
            height: 120px;
            overflow: hidden;
            padding: 10px;
            font-size: 10px;
            font-family: 'Share Tech Mono', monospace;
        }}
        
        .data-line {{
            margin: 2px 0;
            white-space: nowrap;
            overflow: hidden;
            animation: slideIn 0.3s ease-out;
        }}
        
        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateX(-10px); }}
            to {{ opacity: 1; transform: translateX(0); }}
        }}
        
        /* Help Overlay */
        .help-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,20,0.95);
            z-index: 1000;
            display: none;
            justify-content: center;
            align-items: center;
        }}
        
        .help-overlay.visible {{ display: flex; }}
        
        .help-content {{
            max-width: 600px;
            padding: 40px;
            background: rgba(0,0,30,0.9);
            border: 1px solid {color}40;
            border-radius: 20px;
        }}
        
        .help-title {{
            font-family: 'Orbitron', sans-serif;
            font-size: 28px;
            margin-bottom: 20px;
            color: {color};
        }}
        
        .shortcut-list {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }}
        
        .shortcut {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .key {{
            padding: 5px 10px;
            background: {color}20;
            border: 1px solid {color}40;
            border-radius: 5px;
            font-size: 12px;
            min-width: 40px;
            text-align: center;
        }}
        
        .shortcut-desc {{
            font-size: 12px;
            color: #888;
        }}
        
        /* Fullscreen */
        .fullscreen #canvas-container {{
            top: 0;
        }}
        
        .fullscreen .header,
        .fullscreen .settings-panel,
        .fullscreen .metrics-panel,
        .fullscreen .data-panel,
        .fullscreen .control-bar,
        .fullscreen .progress-container,
        .fullscreen .timeline {{
            opacity: 0;
            pointer-events: none;
        }}
        
        /* Recording indicator */
        .recording-indicator {{
            position: fixed;
            top: 80px;
            left: 50%;
            transform: translateX(-50%);
            padding: 10px 20px;
            background: rgba(255,0,0,0.8);
            border-radius: 20px;
            display: none;
            align-items: center;
            gap: 10px;
            z-index: 200;
        }}
        
        .recording-indicator.visible {{ display: flex; }}
        
        .rec-dot {{
            width: 12px;
            height: 12px;
            background: #fff;
            border-radius: 50%;
            animation: blink 1s infinite;
        }}
        
        @keyframes blink {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.3; }}
        }}
        
        /* Toast notifications */
        .toast {{
            position: fixed;
            bottom: 130px;
            left: 50%;
            transform: translateX(-50%) translateY(100px);
            padding: 12px 25px;
            background: rgba(0,0,30,0.95);
            border: 1px solid {color}60;
            border-radius: 25px;
            font-size: 13px;
            opacity: 0;
            transition: all 0.3s;
            z-index: 200;
        }}
        
        .toast.visible {{
            opacity: 1;
            transform: translateX(-50%) translateY(0);
        }}
        
        /* Scrollbar */
        ::-webkit-scrollbar {{ width: 6px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{ background: {color}40; border-radius: 3px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: {color}60; }}
    </style>
</head>
<body>
    <div class="header">
        <div class="title">
            <span class="icon">{icon}</span>
            <span>{title}</span>
        </div>
        <div class="header-controls">
            <button class="header-btn" onclick="toggleHelp()">❓ Help</button>
            <button class="header-btn" onclick="exportData()">📥 Export</button>
            <button class="header-btn" onclick="toggleFullscreen()">⛶ Fullscreen</button>
            <a href="../dashboard/index.html" class="header-btn">← Dashboard</a>
        </div>
    </div>
    
    <div id="canvas-container"></div>
    
    <!-- Settings Panel -->
    <div class="settings-panel">
        <div class="panel-section">
            <div class="section-header">VISUALIZATION MODE</div>
            <div class="btn-group">
                <button class="mode-btn active" data-mode="solid" onclick="setRenderMode('solid')">Solid</button>
                <button class="mode-btn" data-mode="wireframe" onclick="setRenderMode('wireframe')">Wire</button>
                <button class="mode-btn" data-mode="points" onclick="setRenderMode('points')">Points</button>
                <button class="mode-btn" data-mode="hybrid" onclick="setRenderMode('hybrid')">Hybrid</button>
            </div>
        </div>
        
        <div class="panel-section">
            <div class="section-header">PARAMETERS</div>
            <div class="slider-group">
                <div class="slider-label">
                    <span>Rotation Speed</span>
                    <span class="slider-value" id="speedVal">1.0x</span>
                </div>
                <input type="range" class="slider" id="speedSlider" min="0" max="5" step="0.1" value="1" oninput="updateSpeed(this.value)">
            </div>
            <div class="slider-group">
                <div class="slider-label">
                    <span>Scale</span>
                    <span class="slider-value" id="scaleVal">1.0x</span>
                </div>
                <input type="range" class="slider" id="scaleSlider" min="0.1" max="3" step="0.1" value="1" oninput="updateScale(this.value)">
            </div>
            <div class="slider-group">
                <div class="slider-label">
                    <span>Complexity</span>
                    <span class="slider-value" id="complexVal">5</span>
                </div>
                <input type="range" class="slider" id="complexSlider" min="1" max="10" step="1" value="5" oninput="updateComplexity(this.value)">
            </div>
            <div class="slider-group">
                <div class="slider-label">
                    <span>Bloom Intensity</span>
                    <span class="slider-value" id="bloomVal">1.5</span>
                </div>
                <input type="range" class="slider" id="bloomSlider" min="0" max="3" step="0.1" value="1.5" oninput="updateBloom(this.value)">
            </div>
        </div>
        
        <div class="panel-section">
            <div class="section-header">COLOR SCHEME</div>
            <div class="color-picker-group">
                <div class="color-swatch active" style="background: {color}" onclick="setColorScheme('{color}')"></div>
                <div class="color-swatch" style="background: #ff00ff" onclick="setColorScheme('#ff00ff')"></div>
                <div class="color-swatch" style="background: #00ff88" onclick="setColorScheme('#00ff88')"></div>
                <div class="color-swatch" style="background: #ffaa00" onclick="setColorScheme('#ffaa00')"></div>
                <div class="color-swatch" style="background: #ff3366" onclick="setColorScheme('#ff3366')"></div>
                <div class="color-swatch" style="background: #9933ff" onclick="setColorScheme('#9933ff')"></div>
                <div class="color-swatch" style="background: #33ffcc" onclick="setColorScheme('#33ffcc')"></div>
                <div class="color-swatch" style="background: #ffffff" onclick="setColorScheme('#ffffff')"></div>
            </div>
        </div>
        
        <div class="panel-section">
            <div class="section-header">OPTIONS</div>
            <div class="toggle-group">
                <span class="toggle-label">Auto-Rotate</span>
                <div class="toggle active" id="autoRotateToggle" onclick="toggleOption('autoRotate')"></div>
            </div>
            <div class="toggle-group">
                <span class="toggle-label">Particles</span>
                <div class="toggle active" id="particlesToggle" onclick="toggleOption('particles')"></div>
            </div>
            <div class="toggle-group">
                <span class="toggle-label">Grid</span>
                <div class="toggle active" id="gridToggle" onclick="toggleOption('grid')"></div>
            </div>
            <div class="toggle-group">
                <span class="toggle-label">Bloom Effect</span>
                <div class="toggle active" id="bloomToggle" onclick="toggleOption('bloom')"></div>
            </div>
            <div class="toggle-group">
                <span class="toggle-label">Axes Helper</span>
                <div class="toggle" id="axesToggle" onclick="toggleOption('axes')"></div>
            </div>
            <div class="toggle-group">
                <span class="toggle-label">Sound</span>
                <div class="toggle" id="soundToggle" onclick="toggleOption('sound')"></div>
            </div>
        </div>
        
        <div class="panel-section">
            <div class="section-header">CAMERA PRESETS</div>
            <div class="btn-group">
                <button class="mode-btn" onclick="setCameraPreset('front')">Front</button>
                <button class="mode-btn" onclick="setCameraPreset('top')">Top</button>
                <button class="mode-btn" onclick="setCameraPreset('side')">Side</button>
                <button class="mode-btn" onclick="setCameraPreset('iso')">Iso</button>
            </div>
        </div>
    </div>
    
    <!-- Metrics Panel -->
    <div class="metrics-panel">
        <div class="panel-section">
            <div class="section-header">LIVE METRICS</div>
            <div class="metric-row">
                <span class="metric-label">Iterations</span>
                <span class="metric-value" id="iterations">0</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Entropy</span>
                <span class="metric-value orange" id="entropy">2.100</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Fidelity</span>
                <span class="metric-value green" id="fidelity">0.000</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Convergence</span>
                <span class="metric-value magenta" id="convergence">0.000</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">FPS</span>
                <span class="metric-value" id="fps">60</span>
            </div>
        </div>
        <div class="mini-chart">
            <canvas id="metricsChart"></canvas>
        </div>
    </div>
    
    <!-- Data Panel -->
    <div class="data-panel">
        <div class="panel-section">
            <div class="section-header">DATA STREAM</div>
        </div>
        <div class="data-stream" id="dataStream"></div>
    </div>
    
    <!-- Progress -->
    <div class="progress-container">
        <div class="progress-fill" id="progress"></div>
        <span class="progress-label" id="progressLabel">0%</span>
    </div>
    
    <!-- Timeline -->
    <div class="timeline">
        <div class="timeline-point active" id="tp1"></div>
        <div class="timeline-point" id="tp2"></div>
        <div class="timeline-point" id="tp3"></div>
        <div class="timeline-point" id="tp4"></div>
        <div class="timeline-point" id="tp5"></div>
        <div class="timeline-point" id="tp6"></div>
        <div class="timeline-point" id="tp7"></div>
        <div class="timeline-point" id="tp8"></div>
        <div class="timeline-point" id="tp9"></div>
        <div class="timeline-point" id="tp10"></div>
    </div>
    
    <!-- Control Bar -->
    <div class="control-bar">
        <button class="ctrl-btn" id="playBtn" onclick="toggleSimulation()">▶ Play</button>
        <button class="ctrl-btn" onclick="stepSimulation()">⏭ Step</button>
        <button class="ctrl-btn" onclick="resetSimulation()">↺ Reset</button>
        <button class="ctrl-btn" id="recordBtn" onclick="toggleRecording()">⏺ Record</button>
        <button class="ctrl-btn" onclick="captureScreenshot()">📷 Capture</button>
        <button class="ctrl-btn" onclick="randomize()">🎲 Randomize</button>
    </div>
    
    <!-- Recording Indicator -->
    <div class="recording-indicator" id="recordingIndicator">
        <div class="rec-dot"></div>
        <span>REC <span id="recTime">00:00</span></span>
    </div>
    
    <!-- Toast -->
    <div class="toast" id="toast"></div>
    
    <!-- Help Overlay -->
    <div class="help-overlay" id="helpOverlay">
        <div class="help-content">
            <div class="help-title">⌨️ Keyboard Shortcuts</div>
            <div class="shortcut-list">
                <div class="shortcut"><span class="key">Space</span><span class="shortcut-desc">Play/Pause</span></div>
                <div class="shortcut"><span class="key">R</span><span class="shortcut-desc">Reset</span></div>
                <div class="shortcut"><span class="key">W</span><span class="shortcut-desc">Wireframe</span></div>
                <div class="shortcut"><span class="key">G</span><span class="shortcut-desc">Toggle Grid</span></div>
                <div class="shortcut"><span class="key">P</span><span class="shortcut-desc">Particles</span></div>
                <div class="shortcut"><span class="key">B</span><span class="shortcut-desc">Bloom</span></div>
                <div class="shortcut"><span class="key">S</span><span class="shortcut-desc">Screenshot</span></div>
                <div class="shortcut"><span class="key">F</span><span class="shortcut-desc">Fullscreen</span></div>
                <div class="shortcut"><span class="key">1-4</span><span class="shortcut-desc">Camera Presets</span></div>
                <div class="shortcut"><span class="key">Esc</span><span class="shortcut-desc">Exit/Close</span></div>
                <div class="shortcut"><span class="key">+/-</span><span class="shortcut-desc">Zoom</span></div>
                <div class="shortcut"><span class="key">?</span><span class="shortcut-desc">Help</span></div>
            </div>
            <button class="mode-btn" style="margin-top: 25px; width: 100%;" onclick="toggleHelp()">Close</button>
        </div>
    </div>

    <script>
        // QRATUM Platform Integration
        const PLATFORM_API = 'http://localhost:9000';
        const MODULE_ID = '{module_id}';
        const MODULE_CATEGORY = '{category}';
        const COMPONENT = '{component}';
        let platformConnected = false;
        let platformData = {{}};
        
        // Connect to QRATUM Platform
        async function connectPlatform() {{
            try {{
                const response = await fetch(PLATFORM_API + '/api/status');
                if (response.ok) {{
                    platformData = await response.json();
                    platformConnected = true;
                    console.log('✅ Connected to QRATUM Platform');
                    addDataStreamLine('PLATFORM', 'Connected to QRATUM', '#00ff88');
                    startPlatformPolling();
                }}
            }} catch (e) {{
                console.log('⚠️ Platform offline, using local simulation');
                addDataStreamLine('PLATFORM', 'Local mode', '#ffaa00');
            }}
        }}
        
        // Poll platform for live metrics
        function startPlatformPolling() {{
            setInterval(async () => {{
                if (!platformConnected) return;
                try {{
                    const metrics = await fetch(PLATFORM_API + '/api/metrics').then(r => r.json());
                    // Update metrics from platform
                    if (metrics.iterations) {{
                        document.getElementById('iterations').textContent = metrics.iterations.toLocaleString();
                    }}
                    if (metrics.entropy) {{
                        document.getElementById('entropy').textContent = metrics.entropy.toFixed(3);
                    }}
                    if (metrics.convergence) {{
                        document.getElementById('convergence').textContent = metrics.convergence.toFixed(3);
                    }}
                    // Update progress from platform data
                    const progress = Math.min(100, metrics.convergence * 100);
                    document.getElementById('progress').style.width = progress + '%';
                    document.getElementById('progressLabel').textContent = progress.toFixed(1) + '%';
                }} catch (e) {{ /* ignore */ }}
            }}, 2000);
        }}
        
        // Add to data stream
        function addDataStreamLine(type, msg, color) {{
            const stream = document.getElementById('dataStream');
            if (stream) {{
                const line = document.createElement('div');
                line.className = 'stream-line';
                line.innerHTML = '<span style="color:' + color + '">[' + type + ']</span> ' + msg + 
                    ' <span style="color:#666">' + new Date().toLocaleTimeString() + '</span>';
                stream.insertBefore(line, stream.firstChild);
                if (stream.children.length > 20) stream.removeChild(stream.lastChild);
            }}
        }}
        
        // Three.js setup
        let scene, camera, renderer, mainObject;
        let isRunning = false;
        let iterations = 0;
        let wireframe = false;
        let autoRotate = true;
        let lastTime = performance.now();
        let frameCount = 0;
        
        const COLOR = '{color}';
        const MODULE_TYPE = '{module_type}';
        
        function init() {{
            const container = document.getElementById('canvas-container');
            
            scene = new THREE.Scene();
            scene.fog = new THREE.FogExp2(0x000010, 0.015);
            
            camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(0, 0, 50);
            
            renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
            renderer.setSize(window.innerWidth, window.innerHeight - 80);
            renderer.setClearColor(0x000010);
            container.appendChild(renderer.domElement);
            
            // Lights
            const light1 = new THREE.PointLight(parseInt(COLOR.replace('#', '0x')), 2, 200);
            light1.position.set(50, 50, 50);
            scene.add(light1);
            
            const light2 = new THREE.PointLight(0xff00ff, 1.5, 200);
            light2.position.set(-50, -50, 50);
            scene.add(light2);
            
            scene.add(new THREE.AmbientLight(0x404040));
            
            // Create main visualization based on module type
            createVisualization();
            
            // Particles
            createParticles();
            
            // Grid
            createGrid();
            
            // Mouse controls
            let isDragging = false;
            let prevX = 0, prevY = 0;
            
            container.addEventListener('mousedown', (e) => {{
                isDragging = true;
                prevX = e.clientX;
                prevY = e.clientY;
            }});
            
            container.addEventListener('mousemove', (e) => {{
                if (isDragging && mainObject) {{
                    const dx = e.clientX - prevX;
                    const dy = e.clientY - prevY;
                    mainObject.rotation.y += dx * 0.01;
                    mainObject.rotation.x += dy * 0.01;
                    prevX = e.clientX;
                    prevY = e.clientY;
                }}
            }});
            
            container.addEventListener('mouseup', () => isDragging = false);
            container.addEventListener('mouseleave', () => isDragging = false);
            
            container.addEventListener('wheel', (e) => {{
                camera.position.z += e.deltaY * 0.05;
                camera.position.z = Math.max(20, Math.min(100, camera.position.z));
            }});
            
            window.addEventListener('resize', onResize);
            
            animate();
        }}
        
        function createVisualization() {{
            mainObject = new THREE.Group();
            
            // Different visualizations based on module type seed
            const seed = MODULE_TYPE.split('').reduce((a, c) => a + c.charCodeAt(0), 0);
            const vizType = seed % 10;
            
            switch(vizType) {{
                case 0: createSphere(); break;
                case 1: createTorus(); break;
                case 2: createHelix(); break;
                case 3: createCube(); break;
                case 4: createIcosahedron(); break;
                case 5: createTube(); break;
                case 6: createKnot(); break;
                case 7: createOctahedron(); break;
                case 8: createTetrahedron(); break;
                case 9: createDodecahedron(); break;
            }}
            
            scene.add(mainObject);
        }}
        
        function createSphere() {{
            const geo = new THREE.SphereGeometry(15, 64, 64);
            const mat = new THREE.MeshPhongMaterial({{
                color: parseInt(COLOR.replace('#', '0x')),
                wireframe: wireframe,
                transparent: true,
                opacity: 0.8,
                emissive: parseInt(COLOR.replace('#', '0x')),
                emissiveIntensity: 0.2
            }});
            mainObject.add(new THREE.Mesh(geo, mat));
            
            // Inner sphere
            const innerGeo = new THREE.SphereGeometry(10, 32, 32);
            const innerMat = new THREE.MeshBasicMaterial({{
                color: 0xff00ff,
                wireframe: true,
                transparent: true,
                opacity: 0.3
            }});
            mainObject.add(new THREE.Mesh(innerGeo, innerMat));
        }}
        
        function createTorus() {{
            const geo = new THREE.TorusGeometry(12, 4, 32, 100);
            const mat = new THREE.MeshPhongMaterial({{
                color: parseInt(COLOR.replace('#', '0x')),
                wireframe: wireframe,
                transparent: true,
                opacity: 0.8,
                emissive: parseInt(COLOR.replace('#', '0x')),
                emissiveIntensity: 0.2
            }});
            mainObject.add(new THREE.Mesh(geo, mat));
            
            const geo2 = new THREE.TorusGeometry(12, 4, 32, 100);
            const mat2 = new THREE.MeshBasicMaterial({{
                color: 0xff00ff,
                wireframe: true,
                transparent: true,
                opacity: 0.3
            }});
            const torus2 = new THREE.Mesh(geo2, mat2);
            torus2.rotation.x = Math.PI / 2;
            mainObject.add(torus2);
        }}
        
        function createHelix() {{
            const points = [];
            for (let i = 0; i <= 200; i++) {{
                const t = i / 200;
                const angle = t * Math.PI * 8;
                points.push(new THREE.Vector3(
                    Math.cos(angle) * 8,
                    (t - 0.5) * 30,
                    Math.sin(angle) * 8
                ));
            }}
            const curve = new THREE.CatmullRomCurve3(points);
            const geo = new THREE.TubeGeometry(curve, 200, 1, 16, false);
            const mat = new THREE.MeshPhongMaterial({{
                color: parseInt(COLOR.replace('#', '0x')),
                wireframe: wireframe,
                emissive: parseInt(COLOR.replace('#', '0x')),
                emissiveIntensity: 0.3
            }});
            mainObject.add(new THREE.Mesh(geo, mat));
            
            // Second strand
            const points2 = [];
            for (let i = 0; i <= 200; i++) {{
                const t = i / 200;
                const angle = t * Math.PI * 8 + Math.PI;
                points2.push(new THREE.Vector3(
                    Math.cos(angle) * 8,
                    (t - 0.5) * 30,
                    Math.sin(angle) * 8
                ));
            }}
            const curve2 = new THREE.CatmullRomCurve3(points2);
            const geo2 = new THREE.TubeGeometry(curve2, 200, 1, 16, false);
            const mat2 = new THREE.MeshPhongMaterial({{
                color: 0xff00ff,
                wireframe: wireframe,
                emissive: 0xff00ff,
                emissiveIntensity: 0.3
            }});
            mainObject.add(new THREE.Mesh(geo2, mat2));
        }}
        
        function createCube() {{
            const geo = new THREE.BoxGeometry(15, 15, 15, 8, 8, 8);
            const mat = new THREE.MeshPhongMaterial({{
                color: parseInt(COLOR.replace('#', '0x')),
                wireframe: wireframe,
                transparent: true,
                opacity: 0.7
            }});
            mainObject.add(new THREE.Mesh(geo, mat));
            
            for (let i = 0; i < 8; i++) {{
                const x = (i & 1) ? 7.5 : -7.5;
                const y = (i & 2) ? 7.5 : -7.5;
                const z = (i & 4) ? 7.5 : -7.5;
                const sphereGeo = new THREE.SphereGeometry(1.5, 16, 16);
                const sphereMat = new THREE.MeshBasicMaterial({{ color: 0xff00ff }});
                const sphere = new THREE.Mesh(sphereGeo, sphereMat);
                sphere.position.set(x, y, z);
                mainObject.add(sphere);
            }}
        }}
        
        function createIcosahedron() {{
            const geo = new THREE.IcosahedronGeometry(15, 1);
            const mat = new THREE.MeshPhongMaterial({{
                color: parseInt(COLOR.replace('#', '0x')),
                wireframe: wireframe,
                transparent: true,
                opacity: 0.8,
                flatShading: true
            }});
            mainObject.add(new THREE.Mesh(geo, mat));
            
            const innerGeo = new THREE.IcosahedronGeometry(10, 0);
            const innerMat = new THREE.MeshBasicMaterial({{
                color: 0xff00ff,
                wireframe: true,
                transparent: true,
                opacity: 0.5
            }});
            mainObject.add(new THREE.Mesh(innerGeo, innerMat));
        }}
        
        function createTube() {{
            const points = [];
            for (let i = 0; i <= 100; i++) {{
                const t = i / 100;
                const angle = t * Math.PI * 4;
                const radius = 5 + Math.sin(t * Math.PI * 6) * 3;
                points.push(new THREE.Vector3(
                    Math.cos(angle) * radius,
                    (t - 0.5) * 25,
                    Math.sin(angle) * radius
                ));
            }}
            const curve = new THREE.CatmullRomCurve3(points);
            const geo = new THREE.TubeGeometry(curve, 100, 2, 16, false);
            const mat = new THREE.MeshPhongMaterial({{
                color: parseInt(COLOR.replace('#', '0x')),
                wireframe: wireframe,
                side: THREE.DoubleSide
            }});
            mainObject.add(new THREE.Mesh(geo, mat));
        }}
        
        function createKnot() {{
            const geo = new THREE.TorusKnotGeometry(10, 3, 128, 32);
            const mat = new THREE.MeshPhongMaterial({{
                color: parseInt(COLOR.replace('#', '0x')),
                wireframe: wireframe,
                emissive: parseInt(COLOR.replace('#', '0x')),
                emissiveIntensity: 0.2
            }});
            mainObject.add(new THREE.Mesh(geo, mat));
        }}
        
        function createOctahedron() {{
            const geo = new THREE.OctahedronGeometry(15, 0);
            const mat = new THREE.MeshPhongMaterial({{
                color: parseInt(COLOR.replace('#', '0x')),
                wireframe: wireframe,
                transparent: true,
                opacity: 0.8,
                flatShading: true
            }});
            mainObject.add(new THREE.Mesh(geo, mat));
            
            const edges = new THREE.EdgesGeometry(geo);
            const lineMat = new THREE.LineBasicMaterial({{ color: 0xffffff }});
            mainObject.add(new THREE.LineSegments(edges, lineMat));
        }}
        
        function createTetrahedron() {{
            const geo = new THREE.TetrahedronGeometry(15, 0);
            const mat = new THREE.MeshPhongMaterial({{
                color: parseInt(COLOR.replace('#', '0x')),
                wireframe: wireframe,
                transparent: true,
                opacity: 0.8,
                flatShading: true
            }});
            mainObject.add(new THREE.Mesh(geo, mat));
            
            const innerGeo = new THREE.TetrahedronGeometry(10, 0);
            const innerMat = new THREE.MeshBasicMaterial({{
                color: 0xff00ff,
                wireframe: true
            }});
            const inner = new THREE.Mesh(innerGeo, innerMat);
            inner.rotation.y = Math.PI;
            mainObject.add(inner);
        }}
        
        function createDodecahedron() {{
            const geo = new THREE.DodecahedronGeometry(15, 0);
            const mat = new THREE.MeshPhongMaterial({{
                color: parseInt(COLOR.replace('#', '0x')),
                wireframe: wireframe,
                transparent: true,
                opacity: 0.8,
                flatShading: true
            }});
            mainObject.add(new THREE.Mesh(geo, mat));
        }}
        
        let particles, grid, axesHelper;
        let options = {{
            autoRotate: true,
            particles: true,
            grid: true,
            bloom: true,
            axes: false,
            sound: false
        }};
        let rotationSpeed = 1;
        let objectScale = 1;
        let metricsChart;
        let metricsData = {{ entropy: [], fidelity: [], convergence: [] }};
        let isRecording = false;
        let recordStartTime = 0;
        let dataHistory = [];
        
        function createParticles() {{
            const geo = new THREE.BufferGeometry();
            const count = 3000;
            const positions = new Float32Array(count * 3);
            const colors = new Float32Array(count * 3);
            
            for (let i = 0; i < count * 3; i += 3) {{
                positions[i] = (Math.random() - 0.5) * 200;
                positions[i + 1] = (Math.random() - 0.5) * 200;
                positions[i + 2] = (Math.random() - 0.5) * 200;
                
                colors[i] = Math.random();
                colors[i + 1] = Math.random() * 0.5 + 0.5;
                colors[i + 2] = 1;
            }}
            
            geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
            
            const mat = new THREE.PointsMaterial({{
                size: 0.5,
                transparent: true,
                opacity: 0.7,
                vertexColors: true,
                blending: THREE.AdditiveBlending
            }});
            particles = new THREE.Points(geo, mat);
            scene.add(particles);
        }}
        
        function createGrid() {{
            grid = new THREE.GridHelper(100, 50, 0x333366, 0x222244);
            grid.position.y = -25;
            scene.add(grid);
            
            axesHelper = new THREE.AxesHelper(30);
            axesHelper.visible = false;
            scene.add(axesHelper);
        }}
        
        function initChart() {{
            const ctx = document.getElementById('metricsChart').getContext('2d');
            metricsChart = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: Array(30).fill(''),
                    datasets: [
                        {{ label: 'Entropy', data: [], borderColor: '#ffaa00', borderWidth: 1, fill: false, tension: 0.4, pointRadius: 0 }},
                        {{ label: 'Fidelity', data: [], borderColor: '#00ff88', borderWidth: 1, fill: false, tension: 0.4, pointRadius: 0 }},
                        {{ label: 'Convergence', data: [], borderColor: '#ff00ff', borderWidth: 1, fill: false, tension: 0.4, pointRadius: 0 }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{ legend: {{ display: false }} }},
                    scales: {{
                        x: {{ display: false }},
                        y: {{ display: false, min: 0, max: 2.5 }}
                    }},
                    animation: {{ duration: 0 }}
                }}
            }});
        }}
        
        function animate() {{
            requestAnimationFrame(animate);
            
            const now = performance.now();
            frameCount++;
            if (now - lastTime >= 1000) {{
                document.getElementById('fps').textContent = frameCount;
                frameCount = 0;
                lastTime = now;
            }}
            
            if (mainObject && options.autoRotate) {{
                mainObject.rotation.y += 0.005 * rotationSpeed;
                mainObject.rotation.x += 0.002 * rotationSpeed;
            }}
            
            if (particles && options.particles) {{
                particles.rotation.y += 0.0005 * rotationSpeed;
            }}
            
            if (isRunning) {{
                updateSimulation();
            }}
            
            if (isRecording) {{
                const elapsed = Math.floor((Date.now() - recordStartTime) / 1000);
                const mins = Math.floor(elapsed / 60).toString().padStart(2, '0');
                const secs = (elapsed % 60).toString().padStart(2, '0');
                document.getElementById('recTime').textContent = mins + ':' + secs;
            }}
            
            renderer.render(scene, camera);
        }}
        
        function updateSimulation() {{
            iterations++;
            const progress = (iterations % 1000) / 10;
            const entropy = 2.1 - progress / 50 + (Math.random() - 0.5) * 0.1;
            const fidelity = Math.min(0.999, progress / 100 + (Math.random() - 0.5) * 0.05);
            const convergence = Math.min(0.95, progress / 105 + Math.sin(iterations * 0.05) * 0.05);
            
            document.getElementById('iterations').textContent = iterations;
            document.getElementById('entropy').textContent = entropy.toFixed(3);
            document.getElementById('fidelity').textContent = fidelity.toFixed(3);
            document.getElementById('convergence').textContent = convergence.toFixed(3);
            document.getElementById('progress').style.width = progress + '%';
            document.getElementById('progressLabel').textContent = Math.floor(progress) + '%';
            
            // Update timeline
            const phase = Math.floor(progress / 10);
            for (let i = 1; i <= 10; i++) {{
                const tp = document.getElementById('tp' + i);
                tp.classList.remove('active', 'complete');
                if (i < phase) tp.classList.add('complete');
                else if (i === phase) tp.classList.add('active');
            }}
            
            // Update chart
            metricsData.entropy.push(entropy);
            metricsData.fidelity.push(fidelity);
            metricsData.convergence.push(convergence);
            if (metricsData.entropy.length > 30) {{
                metricsData.entropy.shift();
                metricsData.fidelity.shift();
                metricsData.convergence.shift();
            }}
            if (metricsChart) {{
                metricsChart.data.datasets[0].data = metricsData.entropy;
                metricsChart.data.datasets[1].data = metricsData.fidelity;
                metricsChart.data.datasets[2].data = metricsData.convergence;
                metricsChart.update();
            }}
            
            // Add data stream
            if (iterations % 5 === 0) {{
                addDataLine(entropy, fidelity, convergence);
            }}
            
            // Store history for export
            dataHistory.push({{ iteration: iterations, entropy, fidelity, convergence, timestamp: Date.now() }});
            if (dataHistory.length > 10000) dataHistory.shift();
        }}
        
        function addDataLine(entropy, fidelity, convergence) {{
            const stream = document.getElementById('dataStream');
            const line = document.createElement('div');
            line.className = 'data-line';
            line.innerHTML = '<span style="color:#ffaa00">E:' + entropy.toFixed(2) + '</span> ' +
                '<span style="color:#00ff88">F:' + fidelity.toFixed(2) + '</span> ' +
                '<span style="color:#ff00ff">C:' + convergence.toFixed(2) + '</span> ' +
                '<span style="color:#666">[' + iterations + ']</span>';
            stream.insertBefore(line, stream.firstChild);
            if (stream.children.length > 15) stream.removeChild(stream.lastChild);
        }}
        
        function toggleSimulation() {{
            isRunning = !isRunning;
            const btn = document.getElementById('playBtn');
            btn.textContent = isRunning ? '⏸ Pause' : '▶ Play';
            btn.classList.toggle('active', isRunning);
            showToast(isRunning ? 'Simulation started' : 'Simulation paused');
        }}
        
        function stepSimulation() {{
            if (!isRunning) {{
                updateSimulation();
                showToast('Step +1');
            }}
        }}
        
        function resetSimulation() {{
            iterations = 0;
            metricsData = {{ entropy: [], fidelity: [], convergence: [] }};
            dataHistory = [];
            document.getElementById('iterations').textContent = '0';
            document.getElementById('entropy').textContent = '2.100';
            document.getElementById('fidelity').textContent = '0.000';
            document.getElementById('convergence').textContent = '0.000';
            document.getElementById('progress').style.width = '0%';
            document.getElementById('progressLabel').textContent = '0%';
            document.getElementById('dataStream').innerHTML = '';
            for (let i = 1; i <= 10; i++) {{
                const tp = document.getElementById('tp' + i);
                tp.classList.remove('active', 'complete');
                if (i === 1) tp.classList.add('active');
            }}
            if (metricsChart) metricsChart.update();
            showToast('Simulation reset');
        }}
        
        function setRenderMode(mode) {{
            document.querySelectorAll('.mode-btn[data-mode]').forEach(btn => {{
                btn.classList.toggle('active', btn.dataset.mode === mode);
            }});
            
            mainObject.traverse((child) => {{
                if (child.isMesh && child.material) {{
                    switch(mode) {{
                        case 'solid':
                            child.material.wireframe = false;
                            child.visible = true;
                            break;
                        case 'wireframe':
                            child.material.wireframe = true;
                            child.visible = true;
                            break;
                        case 'points':
                            child.visible = false;
                            break;
                        case 'hybrid':
                            child.material.wireframe = false;
                            child.visible = true;
                            child.material.opacity = 0.5;
                            break;
                    }}
                }}
            }});
            showToast('Render mode: ' + mode);
        }}
        
        function updateSpeed(val) {{
            rotationSpeed = parseFloat(val);
            document.getElementById('speedVal').textContent = val + 'x';
        }}
        
        function updateScale(val) {{
            objectScale = parseFloat(val);
            if (mainObject) {{
                mainObject.scale.set(objectScale, objectScale, objectScale);
            }}
            document.getElementById('scaleVal').textContent = val + 'x';
        }}
        
        function updateComplexity(val) {{
            document.getElementById('complexVal').textContent = val;
            // Complexity would regenerate geometry - simplified here
            showToast('Complexity: ' + val);
        }}
        
        function updateBloom(val) {{
            document.getElementById('bloomVal').textContent = val;
            // Would update bloom pass intensity
        }}
        
        function setColorScheme(color) {{
            document.querySelectorAll('.color-swatch').forEach(s => s.classList.remove('active'));
            event.target.classList.add('active');
            
            const colorInt = parseInt(color.replace('#', '0x'));
            mainObject.traverse((child) => {{
                if (child.isMesh && child.material && !child.material.wireframe) {{
                    child.material.color.setHex(colorInt);
                    if (child.material.emissive) child.material.emissive.setHex(colorInt);
                }}
            }});
            if (particles) {{
                particles.material.color.setHex(colorInt);
            }}
            showToast('Color changed');
        }}
        
        function toggleOption(opt) {{
            options[opt] = !options[opt];
            document.getElementById(opt + 'Toggle').classList.toggle('active', options[opt]);
            
            switch(opt) {{
                case 'particles':
                    if (particles) particles.visible = options.particles;
                    break;
                case 'grid':
                    if (grid) grid.visible = options.grid;
                    break;
                case 'axes':
                    if (axesHelper) axesHelper.visible = options.axes;
                    break;
                case 'sound':
                    // Sound toggle
                    break;
            }}
            showToast(opt + ': ' + (options[opt] ? 'ON' : 'OFF'));
        }}
        
        function setCameraPreset(preset) {{
            const duration = 500;
            const start = {{ x: camera.position.x, y: camera.position.y, z: camera.position.z }};
            let end;
            
            switch(preset) {{
                case 'front': end = {{ x: 0, y: 0, z: 50 }}; break;
                case 'top': end = {{ x: 0, y: 50, z: 0.1 }}; break;
                case 'side': end = {{ x: 50, y: 0, z: 0 }}; break;
                case 'iso': end = {{ x: 30, y: 30, z: 30 }}; break;
            }}
            
            const startTime = Date.now();
            function animateCamera() {{
                const elapsed = Date.now() - startTime;
                const t = Math.min(1, elapsed / duration);
                const ease = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
                
                camera.position.x = start.x + (end.x - start.x) * ease;
                camera.position.y = start.y + (end.y - start.y) * ease;
                camera.position.z = start.z + (end.z - start.z) * ease;
                camera.lookAt(0, 0, 0);
                
                if (t < 1) requestAnimationFrame(animateCamera);
            }}
            animateCamera();
            showToast('Camera: ' + preset);
        }}
        
        function toggleRecording() {{
            isRecording = !isRecording;
            document.getElementById('recordBtn').classList.toggle('active', isRecording);
            document.getElementById('recordingIndicator').classList.toggle('visible', isRecording);
            
            if (isRecording) {{
                recordStartTime = Date.now();
                showToast('Recording started');
            }} else {{
                showToast('Recording stopped');
            }}
        }}
        
        function captureScreenshot() {{
            const link = document.createElement('a');
            link.download = '{module_id}_' + Date.now() + '.png';
            link.href = renderer.domElement.toDataURL('image/png');
            link.click();
            showToast('Screenshot captured');
        }}
        
        function randomize() {{
            // Randomize parameters
            updateSpeed((Math.random() * 4 + 0.5).toFixed(1));
            document.getElementById('speedSlider').value = rotationSpeed;
            
            updateScale((Math.random() * 2 + 0.5).toFixed(1));
            document.getElementById('scaleSlider').value = objectScale;
            
            // Random rotation
            if (mainObject) {{
                mainObject.rotation.x = Math.random() * Math.PI * 2;
                mainObject.rotation.y = Math.random() * Math.PI * 2;
                mainObject.rotation.z = Math.random() * Math.PI * 2;
            }}
            
            showToast('Randomized!');
        }}
        
        function exportData() {{
            const data = {{
                module: '{module_id}',
                category: '{category}',
                exportTime: new Date().toISOString(),
                iterations: iterations,
                history: dataHistory.slice(-1000),
                settings: {{
                    rotationSpeed,
                    objectScale,
                    options
                }}
            }};
            
            const blob = new Blob([JSON.stringify(data, null, 2)], {{ type: 'application/json' }});
            const link = document.createElement('a');
            link.download = '{module_id}_data_' + Date.now() + '.json';
            link.href = URL.createObjectURL(blob);
            link.click();
            showToast('Data exported');
        }}
        
        function toggleFullscreen() {{
            document.body.classList.toggle('fullscreen');
            if (document.body.classList.contains('fullscreen')) {{
                renderer.setSize(window.innerWidth, window.innerHeight);
            }} else {{
                renderer.setSize(window.innerWidth, window.innerHeight - 65);
            }}
            camera.aspect = renderer.domElement.width / renderer.domElement.height;
            camera.updateProjectionMatrix();
        }}
        
        function toggleHelp() {{
            document.getElementById('helpOverlay').classList.toggle('visible');
        }}
        
        function showToast(msg) {{
            const toast = document.getElementById('toast');
            toast.textContent = msg;
            toast.classList.add('visible');
            setTimeout(() => toast.classList.remove('visible'), 2000);
        }}
        
        function onResize() {{
            const h = document.body.classList.contains('fullscreen') ? window.innerHeight : window.innerHeight - 65;
            camera.aspect = window.innerWidth / h;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, h);
        }}
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {{
            if (e.target.tagName === 'INPUT') return;
            
            switch(e.key.toLowerCase()) {{
                case ' ': e.preventDefault(); toggleSimulation(); break;
                case 'r': resetSimulation(); break;
                case 'w': setRenderMode('wireframe'); break;
                case 'g': toggleOption('grid'); break;
                case 'p': toggleOption('particles'); break;
                case 'b': toggleOption('bloom'); break;
                case 's': captureScreenshot(); break;
                case 'f': toggleFullscreen(); break;
                case '1': setCameraPreset('front'); break;
                case '2': setCameraPreset('top'); break;
                case '3': setCameraPreset('side'); break;
                case '4': setCameraPreset('iso'); break;
                case 'escape': 
                    if (document.getElementById('helpOverlay').classList.contains('visible')) {{
                        toggleHelp();
                    }} else if (document.body.classList.contains('fullscreen')) {{
                        toggleFullscreen();
                    }}
                    break;
                case '?': toggleHelp(); break;
                case '+': case '=': camera.position.z = Math.max(20, camera.position.z - 5); break;
                case '-': camera.position.z = Math.min(100, camera.position.z + 5); break;
            }}
        }});
        
        // Initialize
        init();
        initChart();
        connectPlatform();
    </script>
</body>
</html>"""


# Central dashboard template - using string concatenation to avoid format issues
def get_dashboard_template(category_buttons, module_cards):
    return (
        """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QUBIC Module Dashboard | 100 Modules</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Share Tech Mono', monospace;
            background: #000010;
            color: #00f5ff;
            min-height: 100vh;
        }
        
        #bg-canvas {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 0;
        }
        
        .container {
            position: relative;
            z-index: 1;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            padding: 40px 20px;
            background: linear-gradient(180deg, rgba(0,0,30,0.9) 0%, transparent 100%);
        }
        
        .logo {
            font-family: 'Orbitron', sans-serif;
            font-size: 72px;
            font-weight: 900;
            background: linear-gradient(90deg, #00f5ff, #ff00ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: 12px;
            margin-bottom: 10px;
        }
        
        .subtitle {
            font-size: 18px;
            color: #888;
            letter-spacing: 4px;
        }
        
        .stats-bar {
            display: flex;
            justify-content: center;
            gap: 60px;
            margin: 30px 0;
            padding: 20px;
            background: rgba(0,0,30,0.8);
            border-top: 1px solid #00f5ff40;
            border-bottom: 1px solid #00f5ff40;
        }
        
        .stat {
            text-align: center;
        }
        
        .stat-value {
            font-family: 'Orbitron', sans-serif;
            font-size: 48px;
            color: #00f5ff;
        }
        
        .stat-label {
            font-size: 12px;
            color: #666;
            letter-spacing: 2px;
            margin-top: 5px;
        }
        
        .search-bar {
            display: flex;
            justify-content: center;
            margin: 30px 0;
        }
        
        .search-input {
            width: 500px;
            padding: 15px 25px;
            background: rgba(0,0,30,0.9);
            border: 1px solid #00f5ff40;
            border-radius: 30px;
            color: #00f5ff;
            font-family: 'Share Tech Mono', monospace;
            font-size: 16px;
            outline: none;
            transition: all 0.3s;
        }
        
        .search-input:focus {
            border-color: #00f5ff;
            box-shadow: 0 0 30px #00f5ff30;
        }
        
        .search-input::placeholder {
            color: #555;
        }
        
        .category-nav {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 10px;
            margin: 30px 0;
            padding: 0 20px;
        }
        
        .cat-btn {
            padding: 10px 20px;
            background: rgba(0,0,30,0.8);
            border: 1px solid #00f5ff30;
            color: #00f5ff;
            font-family: 'Share Tech Mono', monospace;
            font-size: 13px;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .cat-btn:hover, .cat-btn.active {
            background: #00f5ff;
            color: #000;
        }
        
        .modules-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 20px;
            padding: 20px 40px;
            max-width: 1800px;
            margin: 0 auto;
        }
        
        .module-card {
            background: rgba(0,0,30,0.9);
            border: 1px solid #00f5ff20;
            border-radius: 15px;
            padding: 25px;
            cursor: pointer;
            transition: all 0.3s;
            position: relative;
            overflow: hidden;
        }
        
        .module-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: var(--color);
            transform: scaleX(0);
            transition: transform 0.3s;
        }
        
        .module-card:hover {
            transform: translateY(-5px);
            border-color: var(--color);
            box-shadow: 0 10px 40px rgba(0,245,255,0.2);
        }
        
        .module-card:hover::before {
            transform: scaleX(1);
        }
        
        .module-icon {
            font-size: 36px;
            margin-bottom: 15px;
        }
        
        .module-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 16px;
            color: var(--color);
            margin-bottom: 8px;
        }
        
        .module-desc {
            font-size: 12px;
            color: #666;
            line-height: 1.5;
        }
        
        .module-category {
            position: absolute;
            top: 15px;
            right: 15px;
            font-size: 10px;
            padding: 4px 10px;
            background: var(--color)20;
            color: var(--color);
            border-radius: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .footer {
            text-align: center;
            padding: 40px;
            color: #444;
            font-size: 12px;
        }
        
        @keyframes glow {
            0%, 100% { box-shadow: 0 0 20px #00f5ff40; }
            50% { box-shadow: 0 0 40px #00f5ff60; }
        }
        
        .module-card:hover {
            animation: glow 2s infinite;
        }
    </style>
</head>
<body>
    <canvas id="bg-canvas"></canvas>
    
    <div class="container">
        <div class="header">
            <div class="logo">QUBIC</div>
            <div class="subtitle">QUANTUM UNIFIED BIOINFORMATICS INTERACTIVE CONSOLE</div>
        </div>
        
        <div class="stats-bar">
            <div class="stat">
                <div class="stat-value">100</div>
                <div class="stat-label">TOTAL MODULES</div>
            </div>
            <div class="stat">
                <div class="stat-value">10</div>
                <div class="stat-label">CATEGORIES</div>
            </div>
            <div class="stat">
                <div class="stat-value">∞</div>
                <div class="stat-label">POSSIBILITIES</div>
            </div>
        </div>
        
        <div class="search-bar">
            <input type="text" class="search-input" id="searchInput" placeholder="🔍 Search modules..." oninput="filterModules()">
        </div>
        
        <div class="category-nav">
            <button class="cat-btn active" onclick="filterCategory('all')">ALL (100)</button>
            """
        + category_buttons
        + """
        </div>
        
        <div class="modules-grid" id="modulesGrid">
            """
        + module_cards
        + """
        </div>
        
        <div class="footer">
            QUBIC v2.0 | 100 Interactive Visualization Modules | Powered by Three.js & WebGL
        </div>
    </div>

    <script>
        // Background animation
        const canvas = document.getElementById('bg-canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        
        const particles = [];
        for (let i = 0; i < 100; i++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5,
                size: Math.random() * 2
            });
        }
        
        function animateBg() {
            ctx.fillStyle = 'rgba(0, 0, 16, 0.1)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            ctx.fillStyle = '#00f5ff';
            particles.forEach(p => {
                p.x += p.vx;
                p.y += p.vy;
                
                if (p.x < 0) p.x = canvas.width;
                if (p.x > canvas.width) p.x = 0;
                if (p.y < 0) p.y = canvas.height;
                if (p.y > canvas.height) p.y = 0;
                
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                ctx.fill();
            });
            
            requestAnimationFrame(animateBg);
        }
        
        animateBg();
        
        window.addEventListener('resize', () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        });
        
        // Filtering
        let currentCategory = 'all';
        
        function filterCategory(category) {
            currentCategory = category;
            
            document.querySelectorAll('.cat-btn').forEach(btn => {
                btn.classList.remove('active');
                if (btn.textContent.toLowerCase().includes(category)) {
                    btn.classList.add('active');
                }
            });
            
            filterModules();
        }
        
        function filterModules() {
            const search = document.getElementById('searchInput').value.toLowerCase();
            const cards = document.querySelectorAll('.module-card');
            
            cards.forEach(card => {
                const title = card.querySelector('.module-title').textContent.toLowerCase();
                const desc = card.querySelector('.module-desc').textContent.toLowerCase();
                const category = card.dataset.category;
                
                const matchesSearch = title.includes(search) || desc.includes(search);
                const matchesCategory = currentCategory === 'all' || category === currentCategory;
                
                card.style.display = (matchesSearch && matchesCategory) ? 'block' : 'none';
            });
        }
        
        // Module launch
        function launchModule(moduleId) {
            window.location.href = '../' + moduleId + '/index.html';
        }
    </script>
</body>
</html>"""
    )


def generate_modules():
    """Generate all 100 module HTML files and the central dashboard."""

    # Component mapping for platform integration
    COMPONENT_MAP = {
        "quantum": "quasim",
        "bioinformatics": "xenon",
        "neural": "autonomous",
        "physics": "quasim",
        "chemistry": "quasim",
        "crypto": "quasim",
        "network": "autonomous",
        "space": "quasim",
        "financial": "autonomous",
        "data": "qubic",
    }

    # Create base directories
    os.makedirs(f"{BASE_DIR}/dashboard", exist_ok=True)

    all_modules = []

    # Generate each module
    for category, modules in MODULES.items():
        component = COMPONENT_MAP.get(category, "qubic")

        for module_id, title, description, color, icon in modules:
            # Create module directory
            module_dir = f"{BASE_DIR}/{module_id}"
            os.makedirs(module_dir, exist_ok=True)

            # Generate module HTML with platform integration
            html = MODULE_TEMPLATE.format(
                title=title,
                description=description,
                color=color,
                icon=icon,
                module_id=module_id,
                module_type=module_id,
                category=category.upper(),
                component=component,
            )

            with open(f"{module_dir}/index.html", "w") as f:
                f.write(html)

            all_modules.append(
                {
                    "id": module_id,
                    "title": title,
                    "description": description,
                    "color": color,
                    "icon": icon,
                    "category": category,
                    "component": component,
                }
            )

            print(f"✓ Generated: {module_id} [{component}]")

    # Generate category buttons
    category_buttons = []
    for cat in MODULES:
        count = len(MODULES[cat])
        category_buttons.append(
            f'<button class="cat-btn" onclick="filterCategory(\'{cat}\')">{cat.upper()} ({count})</button>'
        )

    # Generate module cards
    module_cards = []
    for m in all_modules:
        card = f"""
            <div class="module-card" style="--color: {m['color']}" data-category="{m['category']}" onclick="launchModule('{m['id']}')">
                <div class="module-category">{m['category']}</div>
                <div class="module-icon">{m['icon']}</div>
                <div class="module-title">{m['title']}</div>
                <div class="module-desc">{m['description']}</div>
            </div>
        """
        module_cards.append(card)

    # Generate dashboard
    dashboard_html = get_dashboard_template(
        "\n            ".join(category_buttons), "\n            ".join(module_cards)
    )

    with open(f"{BASE_DIR}/dashboard/index.html", "w") as f:
        f.write(dashboard_html)

    print("\n✓ Generated central dashboard")

    # Save module manifest
    with open(f"{BASE_DIR}/modules.json", "w") as f:
        json.dump(all_modules, f, indent=2)

    print("✓ Saved module manifest")

    return len(all_modules)


if __name__ == "__main__":
    print("=" * 60)
    print("QUBIC Module Generator")
    print("=" * 60)
    print()

    count = generate_modules()

    print()
    print("=" * 60)
    print(f"✅ Successfully generated {count} modules!")
    print(f"📁 Location: {BASE_DIR}")
    print(f"🚀 Dashboard: {BASE_DIR}/dashboard/index.html")
    print("=" * 60)
