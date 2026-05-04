"""Web server for the Molecular Dynamics Lab.

Provides HTTP and WebSocket server for real-time molecular visualization,
docking, and dynamics simulation.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Server configuration."""

    host: str = "localhost"
    http_port: int = 8080
    ws_port: int = 8081
    static_dir: Optional[Path] = None
    cors_enabled: bool = True
    debug: bool = False


class MolecularLabHandler(SimpleHTTPRequestHandler):
    """HTTP request handler for Molecular Lab."""

    def __init__(self, *args, lab_server: MolecularLabServer = None, **kwargs):
        self.lab_server = lab_server
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            html = self.lab_server.generate_full_app()
            self.wfile.write(html.encode())
        elif self.path == "/api/status":
            self._send_json({"status": "ok", "version": "1.0.0"})
        elif self.path.startswith("/api/structure/"):
            pdb_id = self.path.split("/")[-1]
            self._handle_structure_request(pdb_id)
        else:
            super().do_GET()

    def do_POST(self):
        """Handle POST requests."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode()

        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON")
            return

        if self.path == "/api/dock":
            self._handle_dock_request(data)
        elif self.path == "/api/simulate":
            self._handle_simulate_request(data)
        elif self.path == "/api/interaction":
            self._handle_interaction_request(data)
        else:
            self._send_error(404, "Not found")

    def _handle_structure_request(self, pdb_id: str):
        """Handle structure fetch request."""
        try:
            structure = self.lab_server.load_structure(pdb_id)
            self._send_json(structure.to_dict())
        except Exception as e:
            self._send_error(500, str(e))

    def _handle_dock_request(self, data: dict):
        """Handle docking request."""
        try:
            result = self.lab_server.run_docking(data)
            self._send_json(result)
        except Exception as e:
            self._send_error(500, str(e))

    def _handle_simulate_request(self, data: dict):
        """Handle simulation request."""
        try:
            result = self.lab_server.run_simulation(data)
            self._send_json(result)
        except Exception as e:
            self._send_error(500, str(e))

    def _handle_interaction_request(self, data: dict):
        """Handle VR interaction request."""
        try:
            result = self.lab_server.process_interaction(data)
            self._send_json(result)
        except Exception as e:
            self._send_error(500, str(e))

    def _send_json(self, data: Any):
        """Send JSON response."""
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        if self.lab_server and self.lab_server.config.cors_enabled:
            self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _send_error(self, code: int, message: str):
        """Send error response."""
        self.send_response(code)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode())

    def log_message(self, format: str, *args):
        """Log HTTP requests."""
        if self.lab_server and self.lab_server.config.debug:
            logger.debug(f"{self.address_string()} - {format % args}")


class MolecularLabServer:
    """Web server for the Advanced Molecular Dynamics Lab.

    Serves the interactive 3D molecular viewer with VR support,
    docking simulation, and real-time MD simulation.
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        """Initialize the server.

        Args:
            config: Server configuration
        """
        self.config = config or ServerConfig()
        self._http_server: Optional[HTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._running = False

        # Import components
        from ..core.docking_engine import DockingEngine
        from ..core.md_simulator import MDSimulator
        from ..core.molecular_viewer import MolecularViewer, ViewerConfig
        from ..core.pdb_loader import PDBLoader
        from ..webxr.haptic_engine import HapticEngine
        from ..webxr.vr_controller import VRController

        self.pdb_loader = PDBLoader()
        self.viewer = MolecularViewer(ViewerConfig(webxr_enabled=True))
        self.docking_engine = DockingEngine()
        self.md_simulator = MDSimulator()
        self.vr_controller = VRController()
        self.haptic_engine = HapticEngine()

        self._structures: dict[str, Any] = {}
        self._ws_clients: list[Any] = []

    def load_structure(self, pdb_id: str):
        """Load a PDB structure.

        Args:
            pdb_id: PDB identifier

        Returns:
            Loaded PDBStructure
        """
        if pdb_id not in self._structures:
            self._structures[pdb_id] = self.pdb_loader.load_from_rcsb(pdb_id)
        return self._structures[pdb_id]

    def run_docking(self, request_data: dict) -> dict:
        """Run docking simulation.

        Args:
            request_data: Docking request parameters

        Returns:
            Docking results
        """
        receptor_id = request_data.get("receptor_pdb")
        ligand_id = request_data.get("ligand_pdb")

        receptor = self.load_structure(receptor_id)
        ligand = self.load_structure(ligand_id)

        self.docking_engine.set_receptor(receptor)
        self.docking_engine.set_ligand(ligand)

        if "binding_site" in request_data:
            site = request_data["binding_site"]
            self.docking_engine.define_binding_site(
                center=tuple(site["center"]),
                size=tuple(site["size"]),
            )
        else:
            self.docking_engine.auto_detect_binding_site()

        result = self.docking_engine.dock()
        return result.to_dict()

    def run_simulation(self, request_data: dict) -> dict:
        """Run MD simulation.

        Args:
            request_data: Simulation request parameters

        Returns:
            Simulation results
        """
        pdb_id = request_data.get("pdb_id")
        num_steps = request_data.get("num_steps", 1000)
        temperature = request_data.get("temperature", 300.0)

        structure = self.load_structure(pdb_id)
        self.md_simulator.load_structure(structure)
        self.md_simulator.config.temperature = temperature
        self.md_simulator.initialize_velocities()

        trajectory = self.md_simulator.run(num_steps, save_frequency=10)

        return {
            "frames": [f.to_dict() for f in trajectory],
            "final_state": self.md_simulator.get_state().to_dict(),
        }

    def process_interaction(self, interaction_data: dict) -> dict:
        """Process VR interaction.

        Args:
            interaction_data: Interaction event data

        Returns:
            Response data
        """
        result = self.vr_controller.process_interaction(interaction_data)

        # Generate haptic feedback
        if interaction_data.get("type") in ["select", "grab", "hover"]:
            self.haptic_engine.feedback_from_molecular_event(
                f"atom_{interaction_data['type']}ed",
                interaction_data,
                interaction_data.get("hand", "right"),
            )

        return result

    def generate_full_app(self) -> str:
        """Generate the complete HTML application.

        Returns:
            Complete HTML page with all features
        """
        vr_js = self.vr_controller.generate_session_init_js()
        hand_tracking_js = self.vr_controller.generate_hand_tracking_js()
        haptic_js = self.haptic_engine.generate_haptic_js()
        vr_controls = self.vr_controller.generate_vr_controls_ui()

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Molecular Dynamics Lab</title>
    
    <!-- 3Dmol.js for molecular visualization -->
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    
    <!-- Three.js for WebXR -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #ffffff;
            overflow-x: hidden;
        }}
        
        /* Header */
        .header {{
            background: rgba(0, 0, 0, 0.3);
            padding: 15px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .logo {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        
        .logo-icon {{
            font-size: 32px;
        }}
        
        .logo-text {{
            font-size: 24px;
            font-weight: 600;
            background: linear-gradient(90deg, #00d4ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        /* Main Layout */
        .main-container {{
            display: grid;
            grid-template-columns: 300px 1fr 300px;
            height: calc(100vh - 70px);
            gap: 0;
        }}
        
        /* Sidebar */
        .sidebar {{
            background: rgba(0, 0, 0, 0.4);
            padding: 20px;
            overflow-y: auto;
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .sidebar-right {{
            border-left: 1px solid rgba(255, 255, 255, 0.1);
            border-right: none;
        }}
        
        .panel {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 15px;
        }}
        
        .panel-title {{
            font-size: 14px;
            font-weight: 600;
            color: #00d4ff;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        /* Form Controls */
        .form-group {{
            margin-bottom: 12px;
        }}
        
        .form-group label {{
            display: block;
            font-size: 12px;
            color: #888;
            margin-bottom: 5px;
        }}
        
        .form-group input,
        .form-group select {{
            width: 100%;
            padding: 10px;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            color: #fff;
            font-size: 14px;
        }}
        
        .form-group input:focus,
        .form-group select:focus {{
            outline: none;
            border-color: #00d4ff;
        }}
        
        /* Buttons */
        .btn {{
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }}
        
        .btn-primary {{
            background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%);
            color: #000;
        }}
        
        .btn-primary:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(0, 212, 255, 0.4);
        }}
        
        .btn-secondary {{
            background: rgba(255, 255, 255, 0.1);
            color: #fff;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}
        
        .btn-secondary:hover {{
            background: rgba(255, 255, 255, 0.2);
        }}
        
        .btn-full {{
            width: 100%;
        }}
        
        /* Viewer Container */
        .viewer-container {{
            position: relative;
            background: #000;
        }}
        
        #mol-viewer {{
            width: 100%;
            height: 100%;
        }}
        
        /* Loading Overlay */
        .loading-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 100;
        }}
        
        .spinner {{
            width: 60px;
            height: 60px;
            border: 4px solid rgba(0, 212, 255, 0.2);
            border-top-color: #00d4ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}
        
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
        
        .loading-text {{
            margin-top: 20px;
            font-size: 16px;
            color: #00d4ff;
        }}
        
        /* Tabs */
        .tabs {{
            display: flex;
            gap: 5px;
            margin-bottom: 15px;
        }}
        
        .tab {{
            flex: 1;
            padding: 10px;
            background: rgba(255, 255, 255, 0.05);
            border: none;
            border-radius: 8px;
            color: #888;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 13px;
        }}
        
        .tab.active {{
            background: rgba(0, 212, 255, 0.2);
            color: #00d4ff;
        }}
        
        /* Status Bar */
        .status-bar {{
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 10px 20px;
            background: rgba(0, 0, 0, 0.7);
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 12px;
            color: #888;
        }}
        
        .status-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #00ff88;
        }}
        
        .status-dot.warning {{
            background: #ffaa00;
        }}
        
        .status-dot.error {{
            background: #ff4444;
        }}
        
        /* Energy Chart */
        .energy-chart {{
            height: 150px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            margin-top: 10px;
        }}
        
        /* Atom Info Card */
        .atom-info {{
            padding: 10px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            font-size: 12px;
        }}
        
        .atom-info-row {{
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .atom-info-row:last-child {{
            border-bottom: none;
        }}
        
        /* Toolbar */
        .toolbar {{
            position: absolute;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 10px;
            padding: 10px;
            background: rgba(0, 0, 0, 0.7);
            border-radius: 12px;
            z-index: 50;
        }}
        
        .tool-btn {{
            width: 44px;
            height: 44px;
            background: rgba(255, 255, 255, 0.1);
            border: none;
            border-radius: 8px;
            color: #fff;
            cursor: pointer;
            font-size: 18px;
            transition: all 0.3s ease;
        }}
        
        .tool-btn:hover {{
            background: rgba(0, 212, 255, 0.3);
        }}
        
        .tool-btn.active {{
            background: rgba(0, 212, 255, 0.5);
            color: #00d4ff;
        }}
        
        /* Slider */
        input[type="range"] {{
            -webkit-appearance: none;
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 3px;
            outline: none;
        }}
        
        input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            background: #00d4ff;
            border-radius: 50%;
            cursor: pointer;
        }}
        
        /* Animations */
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        
        .fade-in {{
            animation: fadeIn 0.3s ease;
        }}
        
        /* Responsive */
        @media (max-width: 1200px) {{
            .main-container {{
                grid-template-columns: 1fr;
            }}
            
            .sidebar {{
                display: none;
            }}
        }}
    </style>
</head>
<body>
    <header class="header">
        <div class="logo">
            <span class="logo-icon">🧬</span>
            <span class="logo-text">Molecular Dynamics Lab</span>
        </div>
        <div class="header-actions">
            <button class="btn btn-secondary" onclick="showHelp()">
                <span>❓</span> Help
            </button>
        </div>
    </header>
    
    <main class="main-container">
        <!-- Left Sidebar - Structure Loading -->
        <aside class="sidebar">
            <div class="panel">
                <div class="panel-title">📂 Load Structure</div>
                <div class="form-group">
                    <label>PDB ID</label>
                    <input type="text" id="pdb-input" placeholder="e.g., 1CRN, 3HTB" maxlength="4">
                </div>
                <button class="btn btn-primary btn-full" onclick="loadStructure()">
                    <span>⬇️</span> Load from RCSB
                </button>
                <div style="margin: 10px 0; text-align: center; color: #666;">or</div>
                <button class="btn btn-secondary btn-full" onclick="uploadPDB()">
                    <span>📤</span> Upload PDB File
                </button>
                <input type="file" id="pdb-upload" accept=".pdb" style="display: none;" onchange="handlePDBUpload(event)">
            </div>
            
            <div class="panel">
                <div class="panel-title">🎨 Visualization</div>
                <div class="form-group">
                    <label>Style</label>
                    <select id="style-select" onchange="changeStyle()">
                        <option value="cartoon">Cartoon</option>
                        <option value="stick">Stick</option>
                        <option value="sphere">Sphere</option>
                        <option value="line">Line</option>
                        <option value="surface">Surface</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Color Scheme</label>
                    <select id="color-select" onchange="changeColor()">
                        <option value="chainHetatm">Chain</option>
                        <option value="ssPyMOL">Secondary Structure</option>
                        <option value="spectrum">Spectrum</option>
                        <option value="greenCarbon">Element</option>
                        <option value="Jmol">Jmol</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Background</label>
                    <input type="color" id="bg-color" value="#000000" onchange="changeBgColor()">
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-title">🔬 Docking</div>
                <div class="form-group">
                    <label>Receptor PDB</label>
                    <input type="text" id="receptor-pdb" placeholder="Receptor PDB ID">
                </div>
                <div class="form-group">
                    <label>Ligand PDB</label>
                    <input type="text" id="ligand-pdb" placeholder="Ligand PDB ID">
                </div>
                <button class="btn btn-primary btn-full" onclick="runDocking()">
                    <span>⚗️</span> Run Docking
                </button>
            </div>
        </aside>
        
        <!-- Main Viewer -->
        <div class="viewer-container">
            <div id="loading-overlay" class="loading-overlay">
                <div class="spinner"></div>
                <div class="loading-text">Initializing Molecular Viewer...</div>
            </div>
            
            <div id="mol-viewer"></div>
            
            <!-- Toolbar -->
            <div class="toolbar">
                <button class="tool-btn" onclick="resetView()" title="Reset View">🏠</button>
                <button class="tool-btn" onclick="toggleSpin()" title="Spin">🔄</button>
                <button class="tool-btn" onclick="toggleSurface()" title="Surface">🫧</button>
                <button class="tool-btn" onclick="toggleLabels()" title="Labels">🏷️</button>
                <button class="tool-btn" onclick="measureDistance()" title="Measure">📏</button>
                <button class="tool-btn" onclick="exportImage()" title="Screenshot">📷</button>
                <button class="tool-btn" id="vr-enter-btn" onclick="enterVR()" title="Enter VR">🥽</button>
            </div>
            
            <!-- Status Bar -->
            <div class="status-bar">
                <div class="status-item">
                    <span class="status-dot"></span>
                    <span id="structure-info">No structure loaded</span>
                </div>
                <div class="status-item">
                    <span id="fps-counter">60 FPS</span>
                </div>
                <div class="status-item">
                    <span id="vr-status">VR: Not Available</span>
                </div>
            </div>
        </div>
        
        <!-- Right Sidebar - Simulation & Info -->
        <aside class="sidebar sidebar-right">
            <div class="panel">
                <div class="panel-title">⚡ Molecular Dynamics</div>
                <div class="form-group">
                    <label>Temperature (K)</label>
                    <input type="number" id="md-temperature" value="300" min="0" max="1000">
                </div>
                <div class="form-group">
                    <label>Steps</label>
                    <input type="number" id="md-steps" value="1000" min="100" max="100000">
                </div>
                <div class="form-group">
                    <label>Timestep (ps)</label>
                    <input type="number" id="md-timestep" value="0.002" step="0.001" min="0.001">
                </div>
                <button class="btn btn-primary btn-full" onclick="runSimulation()">
                    <span>▶️</span> Run Simulation
                </button>
                <button class="btn btn-secondary btn-full" onclick="stopSimulation()" style="margin-top: 10px;">
                    <span>⏹️</span> Stop
                </button>
            </div>
            
            <div class="panel">
                <div class="panel-title">📊 Energy</div>
                <div class="energy-chart" id="energy-chart"></div>
                <div class="atom-info" style="margin-top: 10px;">
                    <div class="atom-info-row">
                        <span>Kinetic</span>
                        <span id="ke-value">0.00 kcal/mol</span>
                    </div>
                    <div class="atom-info-row">
                        <span>Potential</span>
                        <span id="pe-value">0.00 kcal/mol</span>
                    </div>
                    <div class="atom-info-row">
                        <span>Total</span>
                        <span id="te-value">0.00 kcal/mol</span>
                    </div>
                    <div class="atom-info-row">
                        <span>Temperature</span>
                        <span id="temp-value">0.00 K</span>
                    </div>
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-title">🎯 Selected Atom</div>
                <div class="atom-info" id="selected-atom-info">
                    <div style="text-align: center; padding: 20px; color: #666;">
                        Click an atom to see details
                    </div>
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-title">🎮 VR Controls</div>
                <div class="form-group">
                    <label>Molecule Scale</label>
                    <input type="range" id="vr-scale" min="0.01" max="0.5" step="0.01" value="0.1" onchange="updateVRScale()">
                </div>
                <div class="form-group">
                    <label>Haptic Intensity</label>
                    <input type="range" id="haptic-intensity" min="0" max="1" step="0.1" value="1" onchange="updateHapticIntensity()">
                </div>
                <div class="form-group">
                    <label>
                        <input type="checkbox" id="hand-tracking" checked onchange="toggleHandTracking()">
                        Enable Hand Tracking
                    </label>
                </div>
            </div>
        </aside>
    </main>
    
    {vr_controls}
    
    <script>
        // Global state
        let viewer = null;
        let currentStructure = null;
        let isSpinning = false;
        let surfaceVisible = false;
        let labelsVisible = false;
        let simulationRunning = false;
        let selectedAtoms = [];
        let grabbedAtom = null;
        
        // WebSocket for real-time updates
        let ws = null;
        
        // Initialize application
        document.addEventListener('DOMContentLoaded', async function() {{
            // Initialize 3Dmol viewer
            viewer = $3Dmol.createViewer('mol-viewer', {{
                backgroundColor: '#000000',
                antialias: true
            }});
            
            // Set up click handler
            viewer.setClickable({{}}, true, function(atom) {{
                if (atom) {{
                    showAtomInfo(atom);
                    highlightAtom(atom);
                }}
            }});
            
            // Set up hover handler
            viewer.setHoverable({{}}, true, function(atom) {{
                if (atom) {{
                    showAtomTooltip(atom);
                }}
            }}, function() {{
                hideAtomTooltip();
            }});
            
            // Check VR support
            checkVRSupport();
            
            // Connect WebSocket
            connectWebSocket();
            
            // Hide loading overlay
            setTimeout(() => {{
                document.getElementById('loading-overlay').style.display = 'none';
            }}, 1000);
            
            // Start FPS counter
            startFPSCounter();
        }});
        
        // Load structure from RCSB
        async function loadStructure() {{
            const pdbId = document.getElementById('pdb-input').value.trim().toUpperCase();
            if (!pdbId || pdbId.length !== 4) {{
                alert('Please enter a valid 4-character PDB ID');
                return;
            }}
            
            showLoading('Loading structure from RCSB...');
            
            try {{
                const response = await fetch(`/api/structure/${{pdbId}}`);
                if (!response.ok) throw new Error('Structure not found');
                
                const data = await response.json();
                displayStructure(data);
                
            }} catch (error) {{
                alert('Failed to load structure: ' + error.message);
            }} finally {{
                hideLoading();
            }}
        }}
        
        // Display structure in viewer
        function displayStructure(structureData) {{
            viewer.removeAllModels();
            viewer.removeAllSurfaces();
            viewer.removeAllLabels();
            
            // Add model from raw PDB content if available
            if (structureData.raw_content) {{
                viewer.addModel(structureData.raw_content, 'pdb');
            }}
            
            // Apply default style
            const style = document.getElementById('style-select').value;
            const color = document.getElementById('color-select').value;
            applyStyle(style, color);
            
            viewer.zoomTo();
            viewer.render();
            
            currentStructure = structureData;
            updateStructureInfo();
        }}
        
        // Apply rendering style
        function applyStyle(style, colorScheme) {{
            const styleSpec = {{}};
            styleSpec[style] = {{ colorscheme: colorScheme }};
            viewer.setStyle({{}}, styleSpec);
            viewer.render();
        }}
        
        function changeStyle() {{
            const style = document.getElementById('style-select').value;
            const color = document.getElementById('color-select').value;
            applyStyle(style, color);
        }}
        
        function changeColor() {{
            changeStyle();
        }}
        
        function changeBgColor() {{
            const color = document.getElementById('bg-color').value;
            viewer.setBackgroundColor(color);
        }}
        
        // Toggle functions
        function toggleSpin() {{
            isSpinning = !isSpinning;
            if (isSpinning) {{
                viewer.spin('y');
            }} else {{
                viewer.spin(false);
            }}
        }}
        
        function toggleSurface() {{
            if (surfaceVisible) {{
                viewer.removeAllSurfaces();
            }} else {{
                viewer.addSurface($3Dmol.SurfaceType.VDW, {{
                    opacity: 0.7,
                    colorscheme: 'greenCarbon'
                }});
            }}
            surfaceVisible = !surfaceVisible;
            viewer.render();
        }}
        
        function toggleLabels() {{
            labelsVisible = !labelsVisible;
            // Toggle residue labels
            viewer.render();
        }}
        
        function resetView() {{
            viewer.zoomTo();
            viewer.render();
        }}
        
        function exportImage() {{
            const uri = viewer.pngURI();
            const link = document.createElement('a');
            link.href = uri;
            link.download = 'molecule.png';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }}
        
        // Show atom info on click
        function showAtomInfo(atom) {{
            const infoDiv = document.getElementById('selected-atom-info');
            infoDiv.innerHTML = `
                <div class="atom-info-row">
                    <span>Atom</span>
                    <span>${{atom.atom}}</span>
                </div>
                <div class="atom-info-row">
                    <span>Element</span>
                    <span>${{atom.elem}}</span>
                </div>
                <div class="atom-info-row">
                    <span>Residue</span>
                    <span>${{atom.resn}} ${{atom.resi}}</span>
                </div>
                <div class="atom-info-row">
                    <span>Chain</span>
                    <span>${{atom.chain}}</span>
                </div>
                <div class="atom-info-row">
                    <span>Position</span>
                    <span>(${{atom.x.toFixed(2)}}, ${{atom.y.toFixed(2)}}, ${{atom.z.toFixed(2)}})</span>
                </div>
                <div class="atom-info-row">
                    <span>B-factor</span>
                    <span>${{atom.b ? atom.b.toFixed(2) : 'N/A'}}</span>
                </div>
            `;
        }}
        
        function highlightAtom(atom) {{
            // Add highlight sphere
            viewer.addSphere({{
                center: {{x: atom.x, y: atom.y, z: atom.z}},
                radius: 1.0,
                color: '#ff0000',
                opacity: 0.5
            }});
            viewer.render();
            
            // Remove after delay
            setTimeout(() => {{
                viewer.removeAllShapes();
                viewer.render();
            }}, 2000);
        }}
        
        // Run docking
        async function runDocking() {{
            const receptorPdb = document.getElementById('receptor-pdb').value.trim();
            const ligandPdb = document.getElementById('ligand-pdb').value.trim();
            
            if (!receptorPdb || !ligandPdb) {{
                alert('Please enter both receptor and ligand PDB IDs');
                return;
            }}
            
            showLoading('Running docking simulation...');
            
            try {{
                const response = await fetch('/api/dock', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        receptor_pdb: receptorPdb,
                        ligand_pdb: ligandPdb
                    }})
                }});
                
                const result = await response.json();
                displayDockingResult(result);
                
            }} catch (error) {{
                alert('Docking failed: ' + error.message);
            }} finally {{
                hideLoading();
            }}
        }}
        
        function displayDockingResult(result) {{
            console.log('Docking result:', result);
            alert(`Docking complete! Best score: ${{result.best_score?.toFixed(2) || 'N/A'}} kcal/mol`);
        }}
        
        // Run MD simulation
        async function runSimulation() {{
            if (!currentStructure) {{
                alert('Please load a structure first');
                return;
            }}
            
            const temperature = parseFloat(document.getElementById('md-temperature').value);
            const steps = parseInt(document.getElementById('md-steps').value);
            
            showLoading('Running molecular dynamics...');
            simulationRunning = true;
            
            try {{
                const response = await fetch('/api/simulate', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        pdb_id: currentStructure.pdb_id,
                        num_steps: steps,
                        temperature: temperature
                    }})
                }});
                
                const result = await response.json();
                displaySimulationResult(result);
                
            }} catch (error) {{
                alert('Simulation failed: ' + error.message);
            }} finally {{
                hideLoading();
                simulationRunning = false;
            }}
        }}
        
        function stopSimulation() {{
            simulationRunning = false;
            // Send stop signal via WebSocket
            if (ws && ws.readyState === WebSocket.OPEN) {{
                ws.send(JSON.stringify({{ type: 'stop_simulation' }}));
            }}
        }}
        
        function displaySimulationResult(result) {{
            const state = result.final_state;
            document.getElementById('ke-value').textContent = `${{state.kinetic_energy.toFixed(2)}} kcal/mol`;
            document.getElementById('pe-value').textContent = `${{state.potential_energy.toFixed(2)}} kcal/mol`;
            document.getElementById('te-value').textContent = `${{state.total_energy.toFixed(2)}} kcal/mol`;
            document.getElementById('temp-value').textContent = `${{state.temperature.toFixed(2)}} K`;
        }}
        
        // WebSocket connection
        function connectWebSocket() {{
            const wsUrl = `ws://${{window.location.hostname}}:8081`;
            try {{
                ws = new WebSocket(wsUrl);
                ws.onmessage = handleWSMessage;
                ws.onerror = (e) => console.log('WebSocket error:', e);
            }} catch (e) {{
                console.log('WebSocket not available');
            }}
        }}
        
        function handleWSMessage(event) {{
            const data = JSON.parse(event.data);
            
            if (data.type === 'simulation_frame') {{
                updateVisualization(data.frame);
            }} else if (data.type === 'haptic') {{
                playHapticWaveform(data.waveform, data.channel);
            }}
        }}
        
        // VR Support
        async function checkVRSupport() {{
            if (navigator.xr) {{
                const supported = await navigator.xr.isSessionSupported('immersive-vr');
                document.getElementById('vr-status').textContent = 
                    supported ? 'VR: Ready' : 'VR: Not Supported';
                document.getElementById('vr-enter-btn').style.opacity = supported ? 1 : 0.5;
            }} else {{
                document.getElementById('vr-status').textContent = 'VR: Not Available';
            }}
        }}
        
        async function enterVR() {{
            if (!navigator.xr) {{
                alert('WebXR is not available in this browser');
                return;
            }}
            
            try {{
                const session = await navigator.xr.requestSession('immersive-vr', {{
                    requiredFeatures: ['local-floor'],
                    optionalFeatures: ['hand-tracking']
                }});
                
                // Initialize VR session
                initVRSession(session);
                
            }} catch (error) {{
                alert('Failed to enter VR: ' + error.message);
            }}
        }}
        
        function initVRSession(session) {{
            console.log('VR Session started');
            // Full VR initialization would go here
        }}
        
        // Utility functions
        function showLoading(message) {{
            const overlay = document.getElementById('loading-overlay');
            overlay.querySelector('.loading-text').textContent = message;
            overlay.style.display = 'flex';
        }}
        
        function hideLoading() {{
            document.getElementById('loading-overlay').style.display = 'none';
        }}
        
        function updateStructureInfo() {{
            if (currentStructure) {{
                document.getElementById('structure-info').textContent = 
                    `${{currentStructure.pdb_id}} | ${{currentStructure.num_atoms}} atoms | ${{currentStructure.num_residues}} residues`;
            }}
        }}
        
        function uploadPDB() {{
            document.getElementById('pdb-upload').click();
        }}
        
        function handlePDBUpload(event) {{
            const file = event.target.files[0];
            if (file) {{
                const reader = new FileReader();
                reader.onload = function(e) {{
                    viewer.removeAllModels();
                    viewer.addModel(e.target.result, 'pdb');
                    changeStyle();
                    viewer.zoomTo();
                    viewer.render();
                }};
                reader.readAsText(file);
            }}
        }}
        
        // FPS Counter
        let lastFrameTime = performance.now();
        let frameCount = 0;
        
        function startFPSCounter() {{
            setInterval(() => {{
                const fps = Math.round(frameCount * 1000 / (performance.now() - lastFrameTime));
                document.getElementById('fps-counter').textContent = fps + ' FPS';
                frameCount = 0;
                lastFrameTime = performance.now();
            }}, 1000);
            
            function countFrame() {{
                frameCount++;
                requestAnimationFrame(countFrame);
            }}
            countFrame();
        }}
        
        function measureDistance() {{
            alert('Click two atoms to measure the distance between them');
            // Implementation would track two clicks and calculate distance
        }}
        
        function showHelp() {{
            alert(`Molecular Dynamics Lab Help

🖱️ Mouse Controls:
- Left drag: Rotate
- Right drag: Pan
- Scroll: Zoom
- Click: Select atom

🎮 VR Controls:
- Trigger: Select atom
- Grip: Grab molecule
- Thumbstick: Navigate

📂 Loading:
- Enter a 4-character PDB ID (e.g., 1CRN)
- Or upload a local PDB file

🔬 Docking:
- Enter receptor and ligand PDB IDs
- Click Run Docking

⚡ Simulation:
- Set temperature and steps
- Click Run Simulation
            `);
        }}
        
        function showAtomTooltip(atom) {{
            // Show hover tooltip
        }}
        
        function hideAtomTooltip() {{
            // Hide hover tooltip
        }}
        
        function updateVRScale() {{
            const scale = document.getElementById('vr-scale').value;
            // Update VR molecule scale
        }}
        
        function updateHapticIntensity() {{
            const intensity = document.getElementById('haptic-intensity').value;
            // Update haptic intensity
        }}
        
        function toggleHandTracking() {{
            const enabled = document.getElementById('hand-tracking').checked;
            // Toggle hand tracking
        }}
        
        // WebXR and Haptic JavaScript
        {vr_js}
        
        {hand_tracking_js}
        
        {haptic_js}
    </script>
</body>
</html>"""

    def start(self) -> None:
        """Start the server."""
        handler = lambda *args, **kwargs: MolecularLabHandler(*args, lab_server=self, **kwargs)

        self._http_server = HTTPServer((self.config.host, self.config.http_port), handler)

        self._server_thread = threading.Thread(target=self._http_server.serve_forever)
        self._server_thread.daemon = True
        self._server_thread.start()

        self._running = True
        logger.info(
            f"Molecular Dynamics Lab server started at "
            f"http://{self.config.host}:{self.config.http_port}"
        )

    def stop(self) -> None:
        """Stop the server."""
        if self._http_server:
            self._http_server.shutdown()
            self._running = False
            logger.info("Molecular Dynamics Lab server stopped")

    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running
