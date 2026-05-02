#!/usr/bin/env python3
"""XENON Live Simulation Server with WebSocket streaming."""

import random
import threading
import time

from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config["SECRET_KEY"] = "xenon-live-sim"
socketio = SocketIO(app, cors_allowed_origins="*")

# Simulation state
sim_state = {
    "running": False,
    "iteration": 0,
    "max_iterations": 100,
    "entropy": 2.1,
    "posterior": 0.0,
    "mechanisms": 0,
    "phase": "IDLE",
    "sequences": [],
    "energy_history": [],
    "waveform": [],
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>XENON Live Simulation</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Share Tech Mono', monospace;
            background: #000010;
            color: #00f5ff;
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        .header {
            background: linear-gradient(90deg, rgba(0,245,255,0.1), transparent);
            padding: 20px 40px;
            border-bottom: 1px solid rgba(0,245,255,0.3);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            font-family: 'Orbitron', sans-serif;
            font-size: 36px;
            font-weight: 900;
            background: linear-gradient(90deg, #00f5ff, #ff00ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: 6px;
        }
        
        .status {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .live-badge {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 20px;
            background: rgba(0,255,136,0.1);
            border: 1px solid #00ff88;
            border-radius: 20px;
            font-size: 14px;
            color: #00ff88;
        }
        
        .live-dot {
            width: 10px;
            height: 10px;
            background: #00ff88;
            border-radius: 50%;
            animation: pulse 1s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(0,255,136,0.7); }
            50% { box-shadow: 0 0 0 15px rgba(0,255,136,0); }
        }
        
        .main-container {
            display: grid;
            grid-template-columns: 350px 1fr 350px;
            gap: 20px;
            padding: 20px;
            height: calc(100vh - 100px);
        }
        
        .panel {
            background: rgba(0,0,30,0.8);
            border: 1px solid rgba(0,245,255,0.2);
            border-radius: 15px;
            overflow: hidden;
        }
        
        .panel-header {
            padding: 15px 20px;
            background: rgba(0,245,255,0.1);
            border-bottom: 1px solid rgba(0,245,255,0.2);
            font-family: 'Orbitron', sans-serif;
            font-size: 12px;
            letter-spacing: 2px;
        }
        
        .panel-body { padding: 20px; }
        
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 15px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        
        .metric-label { color: #888; font-size: 13px; }
        .metric-value {
            font-family: 'Orbitron', sans-serif;
            font-size: 24px;
        }
        
        .metric-value.cyan { color: #00f5ff; }
        .metric-value.green { color: #00ff88; }
        .metric-value.magenta { color: #ff00ff; }
        .metric-value.orange { color: #ff8800; }
        
        #canvas-container {
            width: 100%;
            height: 100%;
            border-radius: 15px;
            overflow: hidden;
        }
        
        .progress-container {
            margin: 20px 0;
        }
        
        .progress-bar {
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00f5ff, #ff00ff);
            border-radius: 4px;
            transition: width 0.3s;
            box-shadow: 0 0 20px rgba(0,245,255,0.5);
        }
        
        .phase-indicator {
            display: flex;
            justify-content: space-between;
            margin-top: 15px;
            font-size: 11px;
        }
        
        .phase { color: #444; transition: color 0.3s; }
        .phase.active { color: #00f5ff; text-shadow: 0 0 10px #00f5ff; }
        .phase.complete { color: #00ff88; }
        
        .sequence-stream {
            font-size: 12px;
            height: 200px;
            overflow: hidden;
            background: rgba(0,0,0,0.5);
            border-radius: 8px;
            padding: 10px;
        }
        
        .seq-line { margin: 3px 0; white-space: nowrap; }
        .base-A { color: #00ff88; }
        .base-T { color: #ff3366; }
        .base-G { color: #ffaa00; }
        .base-C { color: #00aaff; }
        
        .controls {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }
        
        .btn {
            flex: 1;
            padding: 15px;
            background: rgba(0,245,255,0.1);
            border: 1px solid rgba(0,245,255,0.3);
            color: #00f5ff;
            font-family: 'Share Tech Mono', monospace;
            font-size: 14px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .btn:hover {
            background: #00f5ff;
            color: #000;
        }
        
        .btn.active {
            background: #00f5ff;
            color: #000;
        }
        
        .btn.danger {
            border-color: #ff3366;
            color: #ff3366;
        }
        
        .btn.danger:hover {
            background: #ff3366;
            color: #fff;
        }
        
        .graph-container {
            height: 120px;
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            margin: 15px 0;
            position: relative;
        }
        
        #waveform, #energy-graph {
            width: 100%;
            height: 100%;
        }
        
        .console {
            font-size: 11px;
            height: 150px;
            overflow-y: auto;
            background: rgba(0,0,0,0.5);
            border-radius: 8px;
            padding: 10px;
        }
        
        .console-line { margin: 2px 0; }
        .console-line.info { color: #00f5ff; }
        .console-line.success { color: #00ff88; }
        .console-line.warning { color: #ffaa00; }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">XENON</div>
        <div class="status">
            <div class="live-badge" id="live-badge">
                <div class="live-dot"></div>
                <span id="status-text">CONNECTING...</span>
            </div>
        </div>
    </div>
    
    <div class="main-container">
        <!-- Left Panel -->
        <div style="display: flex; flex-direction: column; gap: 20px;">
            <div class="panel">
                <div class="panel-header">⚡ LIVE METRICS</div>
                <div class="panel-body">
                    <div class="metric">
                        <span class="metric-label">Iteration</span>
                        <span class="metric-value cyan" id="iteration">0 / 100</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Entropy</span>
                        <span class="metric-value orange" id="entropy">2.100</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Posterior</span>
                        <span class="metric-value magenta" id="posterior">0.000</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Mechanisms</span>
                        <span class="metric-value green" id="mechanisms">0</span>
                    </div>
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-header">📊 WAVEFORM</div>
                <div class="panel-body">
                    <div class="graph-container">
                        <canvas id="waveform"></canvas>
                    </div>
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-header">🎛️ CONTROLS</div>
                <div class="panel-body">
                    <div class="controls">
                        <button class="btn" id="startBtn" onclick="startSim()">▶ START</button>
                        <button class="btn danger" onclick="stopSim()">⏹ STOP</button>
                    </div>
                    <div class="controls">
                        <button class="btn" onclick="resetSim()">↺ RESET</button>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Center - 3D View -->
        <div class="panel">
            <div id="canvas-container"></div>
        </div>
        
        <!-- Right Panel -->
        <div style="display: flex; flex-direction: column; gap: 20px;">
            <div class="panel">
                <div class="panel-header">⏱️ PROGRESS</div>
                <div class="panel-body">
                    <div class="progress-container">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <span>Phase: <span id="phase">IDLE</span></span>
                            <span id="percent">0%</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="progress" style="width: 0%"></div>
                        </div>
                        <div class="phase-indicator">
                            <span class="phase" id="p1">ALIGN</span>
                            <span class="phase" id="p2">FUSE</span>
                            <span class="phase" id="p3">ENTROPY</span>
                            <span class="phase" id="p4">INFER</span>
                            <span class="phase" id="p5">DONE</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-header">🧬 SEQUENCE STREAM</div>
                <div class="panel-body">
                    <div class="sequence-stream" id="sequences"></div>
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-header">📈 ENERGY</div>
                <div class="panel-body">
                    <div class="graph-container">
                        <canvas id="energy-graph"></canvas>
                    </div>
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-header">📟 CONSOLE</div>
                <div class="panel-body">
                    <div class="console" id="console"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // WebSocket connection
        const socket = io();
        
        // 3D Scene
        let scene, camera, renderer, helix;
        
        // Data arrays
        let waveformData = new Array(100).fill(0);
        let energyData = new Array(100).fill(-20);
        
        // Initialize
        socket.on('connect', () => {
            document.getElementById('status-text').textContent = 'CONNECTED';
            log('Connected to XENON server', 'success');
        });
        
        socket.on('disconnect', () => {
            document.getElementById('status-text').textContent = 'DISCONNECTED';
            log('Disconnected from server', 'warning');
        });
        
        // Live data updates
        socket.on('update', (data) => {
            document.getElementById('iteration').textContent = `${data.iteration} / ${data.max_iterations}`;
            document.getElementById('entropy').textContent = data.entropy.toFixed(3);
            document.getElementById('posterior').textContent = data.posterior.toFixed(3);
            document.getElementById('mechanisms').textContent = data.mechanisms;
            document.getElementById('phase').textContent = data.phase;
            
            const percent = Math.floor((data.iteration / data.max_iterations) * 100);
            document.getElementById('percent').textContent = `${percent}%`;
            document.getElementById('progress').style.width = `${percent}%`;
            
            // Update phases
            const phases = ['p1', 'p2', 'p3', 'p4', 'p5'];
            const phaseIdx = Math.floor(percent / 20);
            phases.forEach((p, i) => {
                document.getElementById(p).className = 'phase';
                if (i < phaseIdx) document.getElementById(p).classList.add('complete');
                else if (i === phaseIdx) document.getElementById(p).classList.add('active');
            });
            
            // Update waveform
            waveformData.shift();
            waveformData.push(data.waveform || Math.sin(Date.now() * 0.01) * 0.5 + Math.random() * 0.3);
            drawWaveform();
            
            // Update energy
            energyData.shift();
            energyData.push(data.energy || -20 - Math.random() * 20);
            drawEnergy();
            
            // Rotate helix
            if (helix) helix.rotation.y += 0.05;
        });
        
        socket.on('sequence', (seq) => {
            const container = document.getElementById('sequences');
            const line = document.createElement('div');
            line.className = 'seq-line';
            let html = '';
            for (const base of seq) {
                html += `<span class="base-${base}">${base}</span>`;
            }
            line.innerHTML = html;
            container.insertBefore(line, container.firstChild);
            if (container.children.length > 15) container.removeChild(container.lastChild);
        });
        
        socket.on('log', (msg) => {
            log(msg.text, msg.type);
        });
        
        socket.on('complete', (data) => {
            document.getElementById('status-text').textContent = 'COMPLETE';
            document.getElementById('startBtn').classList.remove('active');
            log(`Simulation complete! Top mechanism: ${data.top_mechanism}`, 'success');
        });
        
        // Controls
        function startSim() {
            socket.emit('start', { target: 'EGFR', iterations: 100 });
            document.getElementById('startBtn').classList.add('active');
            document.getElementById('status-text').textContent = 'RUNNING';
            log('Starting XENON simulation...', 'info');
        }
        
        function stopSim() {
            socket.emit('stop');
            document.getElementById('startBtn').classList.remove('active');
            document.getElementById('status-text').textContent = 'STOPPED';
            log('Simulation stopped', 'warning');
        }
        
        function resetSim() {
            socket.emit('reset');
            document.getElementById('iteration').textContent = '0 / 100';
            document.getElementById('entropy').textContent = '2.100';
            document.getElementById('posterior').textContent = '0.000';
            document.getElementById('mechanisms').textContent = '0';
            document.getElementById('progress').style.width = '0%';
            document.getElementById('percent').textContent = '0%';
            document.getElementById('status-text').textContent = 'READY';
            log('Simulation reset', 'info');
        }
        
        function log(text, type = 'info') {
            const console = document.getElementById('console');
            const line = document.createElement('div');
            line.className = `console-line ${type}`;
            const time = new Date().toLocaleTimeString();
            line.textContent = `[${time}] ${text}`;
            console.appendChild(line);
            console.scrollTop = console.scrollHeight;
        }
        
        // Draw waveform
        function drawWaveform() {
            const canvas = document.getElementById('waveform');
            const ctx = canvas.getContext('2d');
            canvas.width = canvas.offsetWidth * 2;
            canvas.height = canvas.offsetHeight * 2;
            ctx.scale(2, 2);
            
            const w = canvas.offsetWidth;
            const h = canvas.offsetHeight;
            
            ctx.fillStyle = 'rgba(0,0,0,0.3)';
            ctx.fillRect(0, 0, w, h);
            
            ctx.beginPath();
            ctx.strokeStyle = '#00f5ff';
            ctx.lineWidth = 2;
            ctx.shadowBlur = 10;
            ctx.shadowColor = '#00f5ff';
            
            waveformData.forEach((v, i) => {
                const x = (i / waveformData.length) * w;
                const y = h / 2 + v * h * 0.4;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();
        }
        
        // Draw energy
        function drawEnergy() {
            const canvas = document.getElementById('energy-graph');
            const ctx = canvas.getContext('2d');
            canvas.width = canvas.offsetWidth * 2;
            canvas.height = canvas.offsetHeight * 2;
            ctx.scale(2, 2);
            
            const w = canvas.offsetWidth;
            const h = canvas.offsetHeight;
            
            ctx.fillStyle = 'rgba(0,0,0,0.3)';
            ctx.fillRect(0, 0, w, h);
            
            const gradient = ctx.createLinearGradient(0, 0, 0, h);
            gradient.addColorStop(0, 'rgba(255,0,255,0.6)');
            gradient.addColorStop(1, 'rgba(0,245,255,0.1)');
            
            ctx.beginPath();
            ctx.fillStyle = gradient;
            ctx.moveTo(0, h);
            
            energyData.forEach((v, i) => {
                const x = (i / energyData.length) * w;
                const y = h - ((v + 50) / 50) * h;
                ctx.lineTo(x, y);
            });
            
            ctx.lineTo(w, h);
            ctx.closePath();
            ctx.fill();
        }
        
        // 3D DNA Helix
        function init3D() {
            const container = document.getElementById('canvas-container');
            
            scene = new THREE.Scene();
            scene.fog = new THREE.FogExp2(0x000010, 0.02);
            
            camera = new THREE.PerspectiveCamera(60, container.offsetWidth / container.offsetHeight, 0.1, 1000);
            camera.position.set(0, 0, 30);
            
            renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
            renderer.setSize(container.offsetWidth, container.offsetHeight);
            renderer.setClearColor(0x000010);
            container.appendChild(renderer.domElement);
            
            // Lights
            const light1 = new THREE.PointLight(0x00f5ff, 2, 100);
            light1.position.set(20, 20, 20);
            scene.add(light1);
            
            const light2 = new THREE.PointLight(0xff00ff, 2, 100);
            light2.position.set(-20, -20, 20);
            scene.add(light2);
            
            scene.add(new THREE.AmbientLight(0x404040));
            
            // DNA Helix
            helix = new THREE.Group();
            
            const helixRadius = 3;
            const helixHeight = 25;
            const turns = 3;
            const points = 80;
            
            for (let strand = 0; strand < 2; strand++) {
                const pts = [];
                const offset = strand * Math.PI;
                
                for (let i = 0; i <= points; i++) {
                    const t = i / points;
                    const angle = t * turns * Math.PI * 2 + offset;
                    const x = Math.cos(angle) * helixRadius;
                    const y = (t - 0.5) * helixHeight;
                    const z = Math.sin(angle) * helixRadius;
                    pts.push(new THREE.Vector3(x, y, z));
                }
                
                const curve = new THREE.CatmullRomCurve3(pts);
                const geo = new THREE.TubeGeometry(curve, points * 2, 0.2, 8, false);
                const mat = new THREE.MeshPhongMaterial({
                    color: strand === 0 ? 0x00f5ff : 0xff00ff,
                    emissive: strand === 0 ? 0x003344 : 0x330033,
                    shininess: 100
                });
                helix.add(new THREE.Mesh(geo, mat));
            }
            
            // Base pairs
            for (let i = 0; i < points; i += 4) {
                const t = i / points;
                const angle1 = t * turns * Math.PI * 2;
                const y = (t - 0.5) * helixHeight;
                
                const geo = new THREE.CylinderGeometry(0.08, 0.08, helixRadius * 2, 8);
                const colors = [0x00ff88, 0xff3366, 0xffaa00, 0x00aaff];
                const mat = new THREE.MeshBasicMaterial({
                    color: colors[i % 4],
                    transparent: true,
                    opacity: 0.6
                });
                const base = new THREE.Mesh(geo, mat);
                base.position.y = y;
                base.rotation.z = Math.PI / 2;
                base.rotation.y = angle1;
                helix.add(base);
            }
            
            scene.add(helix);
            
            // Particles
            const particleGeo = new THREE.BufferGeometry();
            const particleCount = 1000;
            const positions = new Float32Array(particleCount * 3);
            
            for (let i = 0; i < particleCount * 3; i += 3) {
                positions[i] = (Math.random() - 0.5) * 100;
                positions[i + 1] = (Math.random() - 0.5) * 100;
                positions[i + 2] = (Math.random() - 0.5) * 100;
            }
            
            particleGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            const particleMat = new THREE.PointsMaterial({
                color: 0x00f5ff,
                size: 0.3,
                transparent: true,
                opacity: 0.5
            });
            scene.add(new THREE.Points(particleGeo, particleMat));
            
            animate3D();
            
            window.addEventListener('resize', () => {
                camera.aspect = container.offsetWidth / container.offsetHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.offsetWidth, container.offsetHeight);
            });
        }
        
        function animate3D() {
            requestAnimationFrame(animate3D);
            if (helix) helix.rotation.y += 0.005;
            renderer.render(scene, camera);
        }
        
        // Initialize
        init3D();
        drawWaveform();
        drawEnergy();
        log('XENON Live Dashboard initialized', 'success');
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@socketio.on("connect")
def handle_connect():
    print("Client connected")
    emit("log", {"text": "Connected to XENON server", "type": "success"})


@socketio.on("start")
def handle_start(data):
    global sim_state
    sim_state["running"] = True
    sim_state["iteration"] = 0
    sim_state["max_iterations"] = data.get("iterations", 100)
    sim_state["entropy"] = 2.1
    sim_state["posterior"] = 0.0
    sim_state["mechanisms"] = 0

    # Start simulation in background thread
    thread = threading.Thread(target=run_simulation)
    thread.daemon = True
    thread.start()


@socketio.on("stop")
def handle_stop():
    global sim_state
    sim_state["running"] = False


@socketio.on("reset")
def handle_reset():
    global sim_state
    sim_state = {
        "running": False,
        "iteration": 0,
        "max_iterations": 100,
        "entropy": 2.1,
        "posterior": 0.0,
        "mechanisms": 0,
        "phase": "IDLE",
        "sequences": [],
        "energy_history": [],
        "waveform": [],
    }


def run_simulation():
    global sim_state
    import numpy as np

    np.random.seed(42)

    phases = ["ALIGNMENT", "FUSION", "ENTROPY", "INFERENCE", "COMPLETE"]
    bases = "ATGC"

    socketio.emit(
        "log", {"text": "Starting XENON quantum bioinformatics simulation...", "type": "info"}
    )

    while sim_state["running"] and sim_state["iteration"] < sim_state["max_iterations"]:
        sim_state["iteration"] += 1
        i = sim_state["iteration"]
        max_i = sim_state["max_iterations"]

        # Update metrics with some randomness
        progress = i / max_i
        sim_state["entropy"] = max(0.1, 2.1 - progress * 1.8 + np.random.randn() * 0.05)
        sim_state["posterior"] = min(0.55, progress * 0.55 + np.random.randn() * 0.02)
        sim_state["mechanisms"] = min(14, 1 + i // 7)
        sim_state["phase"] = phases[min(4, int(progress * 5))]

        # Generate sequence
        seq = "".join(random.choice(bases) for _ in range(50))

        # Waveform value
        waveform = np.sin(i * 0.2) * 0.5 + np.random.randn() * 0.2

        # Energy value
        energy = -20 - np.random.random() * 20 + np.sin(i * 0.1) * 10

        # Emit update
        socketio.emit(
            "update",
            {
                "iteration": sim_state["iteration"],
                "max_iterations": sim_state["max_iterations"],
                "entropy": sim_state["entropy"],
                "posterior": sim_state["posterior"],
                "mechanisms": sim_state["mechanisms"],
                "phase": sim_state["phase"],
                "waveform": waveform,
                "energy": energy,
            },
        )

        # Emit sequence
        socketio.emit("sequence", seq)

        # Log phase changes
        if i == 1:
            socketio.emit("log", {"text": "Phase 1: Quantum Alignment started", "type": "info"})
        elif i == max_i // 5:
            socketio.emit("log", {"text": "Phase 2: Multi-omics Fusion started", "type": "info"})
        elif i == 2 * max_i // 5:
            socketio.emit("log", {"text": "Phase 3: Transfer Entropy analysis", "type": "info"})
        elif i == 3 * max_i // 5:
            socketio.emit("log", {"text": "Phase 4: Neural-Symbolic Inference", "type": "info"})
        elif i == 4 * max_i // 5:
            socketio.emit(
                "log",
                {"text": f'Discovered {sim_state["mechanisms"]} mechanisms', "type": "success"},
            )

        time.sleep(0.1)  # 100ms per iteration

    if sim_state["iteration"] >= sim_state["max_iterations"]:
        sim_state["running"] = False
        sim_state["phase"] = "COMPLETE"
        socketio.emit(
            "complete",
            {
                "top_mechanism": f"EGFR_target_mutant_5_{random.randint(1,9)}",
                "posterior": sim_state["posterior"],
                "mechanisms": sim_state["mechanisms"],
            },
        )
        socketio.emit(
            "log",
            {
                "text": f'✅ Simulation complete! Final posterior: {sim_state["posterior"]:.4f}',
                "type": "success",
            },
        )


if __name__ == "__main__":
    print("=" * 60)
    print("🧬 XENON Live Simulation Server")
    print("=" * 60)
    print()
    socketio.run(app, host="0.0.0.0", port=8099, debug=False, allow_unsafe_werkzeug=True)
