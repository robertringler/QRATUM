/**
 * QRATUM Dashboard - 3D Visualization Module
 * Uses Three.js for tensor field and simulation visualization
 * @version 2.0.0
 */

const QratumVisualization = (function() {
    'use strict';

    // Three.js components
    let scene, camera, renderer, controls;
    let tensorField = null;
    let deformationMesh = null;
    let circuitVisualization = null;
    let energyLandscape = null;
    let animationId = null;
    let clock = null;

    // Visualization state
    let currentMode = 'tensor';
    let colorMap = 'viridis';
    let isAnimating = true;

    // Color maps
    const colorMaps = {
        viridis: [
            { pos: 0, color: new THREE.Color(0x440154) },
            { pos: 0.25, color: new THREE.Color(0x31688e) },
            { pos: 0.5, color: new THREE.Color(0x35b779) },
            { pos: 0.75, color: new THREE.Color(0xfde725) },
            { pos: 1, color: new THREE.Color(0xfde725) }
        ],
        plasma: [
            { pos: 0, color: new THREE.Color(0x0d0887) },
            { pos: 0.25, color: new THREE.Color(0x7e03a8) },
            { pos: 0.5, color: new THREE.Color(0xcc4778) },
            { pos: 0.75, color: new THREE.Color(0xf89540) },
            { pos: 1, color: new THREE.Color(0xf0f921) }
        ],
        thermal: [
            { pos: 0, color: new THREE.Color(0x000080) },
            { pos: 0.25, color: new THREE.Color(0x0000ff) },
            { pos: 0.5, color: new THREE.Color(0x00ffff) },
            { pos: 0.75, color: new THREE.Color(0xffff00) },
            { pos: 1, color: new THREE.Color(0xff0000) }
        ],
        rainbow: [
            { pos: 0, color: new THREE.Color(0xff0000) },
            { pos: 0.2, color: new THREE.Color(0xff8000) },
            { pos: 0.4, color: new THREE.Color(0xffff00) },
            { pos: 0.6, color: new THREE.Color(0x00ff00) },
            { pos: 0.8, color: new THREE.Color(0x0000ff) },
            { pos: 1, color: new THREE.Color(0x8000ff) }
        ]
    };

    /**
     * Initialize visualization
     * @param {string} containerId - Container element ID
     */
    function init(containerId) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error('[Viz] Container not found:', containerId);
            return;
        }

        // Clean up existing
        cleanup();

        // Create scene
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0a0f);
        scene.fog = new THREE.FogExp2(0x0a0a0f, 0.02);

        // Create camera
        camera = new THREE.PerspectiveCamera(
            60,
            container.clientWidth / container.clientHeight,
            0.1,
            1000
        );
        camera.position.set(0, 15, 30);

        // Create renderer - Crisp Standard: devicePixelRatio + antialias + sRGB
        renderer = new THREE.WebGLRenderer({ 
            antialias: true,
            alpha: false,
            powerPreference: 'high-performance'
        });
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio); // Use full device pixel ratio for crisp rendering
        renderer.setClearColor(0x0a0a0f, 1);
        
        // sRGB color space for accurate colors
        if (renderer.outputColorSpace !== undefined) {
            renderer.outputColorSpace = THREE.SRGBColorSpace;
        } else if (renderer.outputEncoding !== undefined) {
            renderer.outputEncoding = THREE.sRGBEncoding; // Legacy fallback
        }
        
        container.appendChild(renderer.domElement);

        // Create controls (with fallback if OrbitControls not available)
        if (typeof THREE.OrbitControls !== 'undefined') {
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.maxDistance = 100;
            controls.minDistance = 5;
        } else {
            console.warn('[Viz] OrbitControls not available, using basic camera controls');
            // Basic camera controls will be handled manually
            controls = {
                update: function() {},
                reset: function() {
                    camera.position.set(0, 15, 30);
                    camera.lookAt(0, 0, 0);
                }
            };
        }

        // Add lights
        const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
        scene.add(ambientLight);

        const pointLight1 = new THREE.PointLight(0x00f5ff, 2, 100);
        pointLight1.position.set(20, 20, 20);
        scene.add(pointLight1);

        const pointLight2 = new THREE.PointLight(0x7b2cbf, 2, 100);
        pointLight2.position.set(-20, -20, 20);
        scene.add(pointLight2);

        // Add grid helper
        const gridHelper = new THREE.GridHelper(40, 20, 0x00f5ff, 0x1a1a2e);
        gridHelper.material.transparent = true;
        gridHelper.material.opacity = 0.3;
        scene.add(gridHelper);

        // Initialize clock
        clock = new THREE.Clock();

        // Create default visualization
        createTensorField();

        // Start animation loop
        animate();

        // Handle resize
        window.addEventListener('resize', () => handleResize(container));

        console.log('[Viz] Initialized');
    }

    /**
     * Create tensor field visualization
     * @param {Object} data - Optional tensor data
     */
    function createTensorField(data = null) {
        // Remove existing
        if (tensorField) {
            scene.remove(tensorField);
            tensorField = null;
        }

        tensorField = new THREE.Group();

        // Create 3D grid of tensor glyphs
        const gridSize = 5;
        const spacing = 3;

        for (let x = -gridSize; x <= gridSize; x++) {
            for (let y = -gridSize; y <= gridSize; y++) {
                for (let z = -gridSize; z <= gridSize; z++) {
                    // Skip some points for performance
                    if (Math.abs(x) + Math.abs(y) + Math.abs(z) > gridSize * 1.5) continue;

                    // Calculate tensor magnitude (simulated)
                    const distance = Math.sqrt(x*x + y*y + z*z);
                    const magnitude = Math.exp(-distance * 0.3) * (0.5 + Math.random() * 0.5);

                    // Get color from colormap
                    const color = getColorFromMap(magnitude, colorMap);

                    // Create glyph (icosahedron for tensor representation)
                    const geometry = new THREE.IcosahedronGeometry(magnitude * 0.5, 1);
                    const material = new THREE.MeshPhongMaterial({
                        color: color,
                        emissive: color,
                        emissiveIntensity: 0.3,
                        transparent: true,
                        opacity: 0.8,
                        shininess: 100
                    });

                    const glyph = new THREE.Mesh(geometry, material);
                    glyph.position.set(x * spacing, y * spacing, z * spacing);
                    glyph.userData = { x, y, z, magnitude };
                    tensorField.add(glyph);
                }
            }
        }

        scene.add(tensorField);
    }

    /**
     * Create deformation visualization
     * @param {Object} data - Deformation data
     */
    function createDeformationVisualization(data = null) {
        // Remove existing
        if (deformationMesh) {
            scene.remove(deformationMesh);
            deformationMesh = null;
        }

        deformationMesh = new THREE.Group();

        // Create deformable surface
        const width = 20;
        const height = 20;
        const segments = 30;

        const geometry = new THREE.PlaneGeometry(width, height, segments, segments);
        
        // Apply deformation to vertices
        const positions = geometry.attributes.position;
        for (let i = 0; i < positions.count; i++) {
            const x = positions.getX(i);
            const y = positions.getY(i);
            
            // Calculate deformation (wave pattern)
            const distance = Math.sqrt(x*x + y*y);
            const deformation = Math.sin(distance * 0.5) * Math.exp(-distance * 0.1) * 3;
            
            positions.setZ(i, deformation);
        }
        
        geometry.computeVertexNormals();

        // Color vertices based on deformation
        const colors = new Float32Array(positions.count * 3);
        for (let i = 0; i < positions.count; i++) {
            const z = positions.getZ(i);
            const normalized = (z + 3) / 6; // Normalize to 0-1
            const color = getColorFromMap(normalized, colorMap);
            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
        }
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.MeshPhongMaterial({
            vertexColors: true,
            side: THREE.DoubleSide,
            shininess: 80
        });

        const mesh = new THREE.Mesh(geometry, material);
        mesh.rotation.x = -Math.PI / 2;
        deformationMesh.add(mesh);

        // Add wireframe overlay
        const wireframe = new THREE.WireframeGeometry(geometry);
        const wireMaterial = new THREE.LineBasicMaterial({
            color: 0x00f5ff,
            transparent: true,
            opacity: 0.2
        });
        const wireframeMesh = new THREE.LineSegments(wireframe, wireMaterial);
        wireframeMesh.rotation.x = -Math.PI / 2;
        deformationMesh.add(wireframeMesh);

        scene.add(deformationMesh);
    }

    /**
     * Create quantum circuit visualization
     */
    function createCircuitVisualization() {
        // Remove existing
        if (circuitVisualization) {
            scene.remove(circuitVisualization);
            circuitVisualization = null;
        }

        circuitVisualization = new THREE.Group();

        // Create qubit lines
        const numQubits = 5;
        const numGates = 8;
        const lineSpacing = 3;
        const gateSpacing = 3;

        // Qubit lines
        for (let q = 0; q < numQubits; q++) {
            const points = [
                new THREE.Vector3(-12, q * lineSpacing - 6, 0),
                new THREE.Vector3(12, q * lineSpacing - 6, 0)
            ];
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const material = new THREE.LineBasicMaterial({
                color: 0x00f5ff,
                transparent: true,
                opacity: 0.5
            });
            const line = new THREE.Line(geometry, material);
            circuitVisualization.add(line);

            // Qubit label
            const labelGeometry = new THREE.SphereGeometry(0.3, 16, 16);
            const labelMaterial = new THREE.MeshPhongMaterial({
                color: 0x00f5ff,
                emissive: 0x004455
            });
            const label = new THREE.Mesh(labelGeometry, labelMaterial);
            label.position.set(-14, q * lineSpacing - 6, 0);
            circuitVisualization.add(label);
        }

        // Gates
        const gateTypes = ['H', 'X', 'Y', 'Z', 'CNOT', 'RZ', 'CZ', 'T'];
        const gateColors = {
            H: 0x00f5ff,
            X: 0xff006e,
            Y: 0x00ff88,
            Z: 0x7b2cbf,
            CNOT: 0xffaa00,
            RZ: 0x33ccff,
            CZ: 0xff9933,
            T: 0x66ff33
        };

        // Random gate placement
        for (let g = 0; g < numGates; g++) {
            const gateType = gateTypes[g % gateTypes.length];
            const qubit = Math.floor(Math.random() * numQubits);
            const xPos = (g - numGates/2) * gateSpacing;

            // Gate box
            const gateGeometry = new THREE.BoxGeometry(1.5, 1.5, 1.5);
            const gateMaterial = new THREE.MeshPhongMaterial({
                color: gateColors[gateType],
                emissive: gateColors[gateType],
                emissiveIntensity: 0.3,
                transparent: true,
                opacity: 0.9
            });
            const gate = new THREE.Mesh(gateGeometry, gateMaterial);
            gate.position.set(xPos, qubit * lineSpacing - 6, 0);
            circuitVisualization.add(gate);

            // For CNOT/CZ, add control line
            if (gateType === 'CNOT' || gateType === 'CZ') {
                const targetQubit = (qubit + 1) % numQubits;
                const controlPoints = [
                    new THREE.Vector3(xPos, qubit * lineSpacing - 6, 0),
                    new THREE.Vector3(xPos, targetQubit * lineSpacing - 6, 0)
                ];
                const controlGeometry = new THREE.BufferGeometry().setFromPoints(controlPoints);
                const controlMaterial = new THREE.LineBasicMaterial({
                    color: gateColors[gateType],
                    linewidth: 2
                });
                const controlLine = new THREE.Line(controlGeometry, controlMaterial);
                circuitVisualization.add(controlLine);

                // Control dot
                const dotGeometry = new THREE.SphereGeometry(0.4, 16, 16);
                const dotMaterial = new THREE.MeshPhongMaterial({
                    color: gateColors[gateType],
                    emissive: gateColors[gateType],
                    emissiveIntensity: 0.5
                });
                const dot = new THREE.Mesh(dotGeometry, dotMaterial);
                dot.position.set(xPos, targetQubit * lineSpacing - 6, 0);
                circuitVisualization.add(dot);
            }
        }

        scene.add(circuitVisualization);
    }

    /**
     * Create energy landscape visualization
     */
    function createEnergyLandscape() {
        // Remove existing
        if (energyLandscape) {
            scene.remove(energyLandscape);
            energyLandscape = null;
        }

        energyLandscape = new THREE.Group();

        // Create 3D surface representing energy landscape
        const size = 20;
        const segments = 50;
        const geometry = new THREE.PlaneGeometry(size, size, segments, segments);

        // Generate energy landscape (with local minima)
        const positions = geometry.attributes.position;
        for (let i = 0; i < positions.count; i++) {
            const x = positions.getX(i);
            const y = positions.getY(i);

            // Create complex energy landscape with multiple minima
            let energy = 0;
            
            // Global bowl
            energy += (x*x + y*y) * 0.05;
            
            // Local minima
            energy -= 2 * Math.exp(-((x-3)*(x-3) + (y-3)*(y-3)) * 0.5);
            energy -= 1.5 * Math.exp(-((x+4)*(x+4) + (y-2)*(y-2)) * 0.3);
            energy -= 1.8 * Math.exp(-((x-2)*(x-2) + (y+4)*(y+4)) * 0.4);
            
            // Add noise
            energy += Math.sin(x * 2) * Math.cos(y * 2) * 0.3;

            positions.setZ(i, energy * 2);
        }

        geometry.computeVertexNormals();

        // Color based on energy
        const colors = new Float32Array(positions.count * 3);
        let minZ = Infinity, maxZ = -Infinity;
        
        for (let i = 0; i < positions.count; i++) {
            const z = positions.getZ(i);
            minZ = Math.min(minZ, z);
            maxZ = Math.max(maxZ, z);
        }

        for (let i = 0; i < positions.count; i++) {
            const z = positions.getZ(i);
            const normalized = (z - minZ) / (maxZ - minZ);
            const color = getColorFromMap(normalized, colorMap);
            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
        }
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.MeshPhongMaterial({
            vertexColors: true,
            side: THREE.DoubleSide,
            shininess: 50
        });

        const mesh = new THREE.Mesh(geometry, material);
        mesh.rotation.x = -Math.PI / 2;
        energyLandscape.add(mesh);

        // Add optimization path marker
        const markerGeometry = new THREE.SphereGeometry(0.5, 16, 16);
        const markerMaterial = new THREE.MeshPhongMaterial({
            color: 0xff006e,
            emissive: 0xff006e,
            emissiveIntensity: 0.5
        });
        const marker = new THREE.Mesh(markerGeometry, markerMaterial);
        marker.position.set(0, 5, 0);
        marker.userData.isMarker = true;
        energyLandscape.add(marker);

        scene.add(energyLandscape);
    }

    /**
     * Get color from colormap
     * @param {number} value - Value between 0 and 1
     * @param {string} mapName - Colormap name
     * @returns {THREE.Color}
     */
    function getColorFromMap(value, mapName) {
        const map = colorMaps[mapName] || colorMaps.viridis;
        value = Math.max(0, Math.min(1, value));

        // Find surrounding colors
        let lower = map[0];
        let upper = map[map.length - 1];

        for (let i = 0; i < map.length - 1; i++) {
            if (value >= map[i].pos && value <= map[i + 1].pos) {
                lower = map[i];
                upper = map[i + 1];
                break;
            }
        }

        // Interpolate
        const t = (value - lower.pos) / (upper.pos - lower.pos);
        const color = new THREE.Color();
        color.r = lower.color.r + (upper.color.r - lower.color.r) * t;
        color.g = lower.color.g + (upper.color.g - lower.color.g) * t;
        color.b = lower.color.b + (upper.color.b - lower.color.b) * t;

        return color;
    }

    /**
     * Animation loop
     */
    function animate() {
        animationId = requestAnimationFrame(animate);

        if (!isAnimating) {
            controls.update();
            renderer.render(scene, camera);
            return;
        }

        const delta = clock.getDelta();
        const elapsed = clock.getElapsedTime();

        // Animate tensor field
        if (tensorField && currentMode === 'tensor') {
            tensorField.rotation.y += 0.003;
            tensorField.children.forEach((glyph, i) => {
                glyph.rotation.x += 0.01;
                glyph.rotation.y += 0.01;
                const scale = 1 + Math.sin(elapsed * 2 + i * 0.1) * 0.1;
                glyph.scale.setScalar(scale);
            });
        }

        // Animate deformation
        if (deformationMesh && currentMode === 'deformation') {
            const mesh = deformationMesh.children[0];
            if (mesh && mesh.geometry) {
                const positions = mesh.geometry.attributes.position;
                for (let i = 0; i < positions.count; i++) {
                    const x = positions.getX(i);
                    const y = positions.getY(i);
                    const distance = Math.sqrt(x*x + y*y);
                    const z = Math.sin(distance * 0.5 - elapsed * 2) * Math.exp(-distance * 0.1) * 3;
                    positions.setZ(i, z);
                }
                positions.needsUpdate = true;
                mesh.geometry.computeVertexNormals();
            }
        }

        // Animate quantum circuit
        if (circuitVisualization && currentMode === 'circuit') {
            circuitVisualization.children.forEach((child, i) => {
                if (child.geometry && child.geometry.type === 'BoxGeometry') {
                    child.rotation.y += 0.02;
                    child.rotation.z = Math.sin(elapsed * 3 + i) * 0.2;
                }
            });
        }

        // Animate energy landscape marker
        if (energyLandscape && currentMode === 'energy') {
            energyLandscape.children.forEach(child => {
                if (child.userData.isMarker) {
                    // Simulate optimization path
                    const t = elapsed * 0.5;
                    child.position.x = Math.cos(t) * 5;
                    child.position.z = Math.sin(t * 1.5) * 5;
                    child.position.y = 3 + Math.sin(t * 2);
                }
            });
        }

        controls.update();
        renderer.render(scene, camera);
    }

    /**
     * Handle container resize
     * @param {HTMLElement} container
     */
    function handleResize(container) {
        if (!camera || !renderer) return;

        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    }

    /**
     * Set visualization mode
     * @param {string} mode - Mode name
     */
    function setMode(mode) {
        currentMode = mode;

        // Hide all
        if (tensorField) tensorField.visible = false;
        if (deformationMesh) deformationMesh.visible = false;
        if (circuitVisualization) circuitVisualization.visible = false;
        if (energyLandscape) energyLandscape.visible = false;

        // Show selected
        switch (mode) {
            case 'tensor':
                if (!tensorField) createTensorField();
                tensorField.visible = true;
                break;
            case 'deformation':
                if (!deformationMesh) createDeformationVisualization();
                deformationMesh.visible = true;
                break;
            case 'circuit':
                if (!circuitVisualization) createCircuitVisualization();
                circuitVisualization.visible = true;
                break;
            case 'energy':
                if (!energyLandscape) createEnergyLandscape();
                energyLandscape.visible = true;
                break;
        }

        console.log('[Viz] Mode changed to:', mode);
    }

    /**
     * Set color map
     * @param {string} mapName - Color map name
     */
    function setColorMap(mapName) {
        colorMap = mapName;
        
        // Recreate visualization with new colors
        switch (currentMode) {
            case 'tensor':
                createTensorField();
                break;
            case 'deformation':
                createDeformationVisualization();
                break;
            case 'energy':
                createEnergyLandscape();
                break;
        }

        console.log('[Viz] Color map changed to:', mapName);
    }

    /**
     * Reset camera position
     */
    function resetCamera() {
        if (camera && controls) {
            camera.position.set(0, 15, 30);
            controls.reset();
        }
    }

    /**
     * Toggle fullscreen
     * @param {string} containerId - Container element ID
     */
    function toggleFullscreen(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        if (!document.fullscreenElement) {
            container.requestFullscreen().catch(err => {
                console.error('[Viz] Fullscreen error:', err);
            });
        } else {
            document.exitFullscreen();
        }
    }

    /**
     * Toggle animation
     * @param {boolean} enabled
     */
    function setAnimating(enabled) {
        isAnimating = enabled;
    }

    /**
     * Cleanup visualization
     */
    function cleanup() {
        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }

        if (renderer) {
            renderer.dispose();
            if (renderer.domElement && renderer.domElement.parentNode) {
                renderer.domElement.parentNode.removeChild(renderer.domElement);
            }
        }

        scene = null;
        camera = null;
        renderer = null;
        controls = null;
        tensorField = null;
        deformationMesh = null;
        circuitVisualization = null;
        energyLandscape = null;
    }

    /**
     * Update visualization with new data
     * @param {Object} data - Visualization data
     */
    function updateData(data) {
        // Update based on current mode and data type
        if (data.tensorField) {
            createTensorField(data.tensorField);
        }
        if (data.deformation) {
            createDeformationVisualization(data.deformation);
        }
    }

    // Public API
    return {
        init,
        setMode,
        setColorMap,
        resetCamera,
        toggleFullscreen,
        setAnimating,
        updateData,
        cleanup,
        get currentMode() { return currentMode; },
        get colorMap() { return colorMap; }
    };
})();

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = QratumVisualization;
}
