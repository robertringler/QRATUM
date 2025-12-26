/**
 * QRATUM SOI - Planetary Node Map
 * 
 * Holographic Earth visualization with:
 * - Live validator nodes
 * - Z-zones glowing by classification
 * - Air-gapped Z3 vaults as black monoliths
 * - BFT quorum flows animated in real time
 * 
 * @version 1.0.0
 */

class PlanetaryMapRenderer {
    constructor(scene, camera) {
        this.scene = scene;
        this.camera = camera;
        this.isActive = false;
        
        // Groups for organization
        this.earthGroup = null;
        this.nodesGroup = null;
        this.connectionsGroup = null;
        this.zonesGroup = null;
        
        // Earth mesh
        this.earth = null;
        this.atmosphere = null;
        
        // Node data
        this.nodes = [];
        this.connections = [];
        
        // Animation state
        this.earthRotation = 0;
    }
    
    /**
     * Initialize the planetary map
     */
    async initialize() {
        console.log('[PlanetaryMap] Initializing...');
        
        // Create groups
        this.earthGroup = new THREE.Group();
        this.nodesGroup = new THREE.Group();
        this.connectionsGroup = new THREE.Group();
        this.zonesGroup = new THREE.Group();
        
        // Create Earth
        this.createEarth();
        
        // Create atmosphere
        this.createAtmosphere();
        
        // Create zone rings
        this.createZoneRings();
        
        // Generate demo nodes
        this.generateDemoNodes();
        
        // Add groups to scene
        this.scene.add(this.earthGroup);
        
        console.log('[PlanetaryMap] Initialization complete');
    }
    
    /**
     * Create holographic Earth mesh
     */
    createEarth() {
        // Earth geometry
        const earthGeometry = new THREE.SphereGeometry(80, 64, 64);
        
        // Holographic wireframe material
        const earthMaterial = new THREE.MeshBasicMaterial({
            color: 0x003566,
            wireframe: true,
            transparent: true,
            opacity: 0.3
        });
        
        this.earth = new THREE.Mesh(earthGeometry, earthMaterial);
        this.earthGroup.add(this.earth);
        
        // Add solid core
        const coreGeometry = new THREE.SphereGeometry(78, 32, 32);
        const coreMaterial = new THREE.MeshPhongMaterial({
            color: 0x001d3d,
            transparent: true,
            opacity: 0.8,
            emissive: 0x001d3d,
            emissiveIntensity: 0.3
        });
        const core = new THREE.Mesh(coreGeometry, coreMaterial);
        this.earthGroup.add(core);
        
        // Add latitude/longitude grid lines
        this.createGridLines();
    }
    
    /**
     * Create latitude/longitude grid lines
     */
    createGridLines() {
        const gridMaterial = new THREE.LineBasicMaterial({
            color: 0x00b4d8,
            transparent: true,
            opacity: 0.15
        });
        
        // Latitude lines
        for (let lat = -60; lat <= 60; lat += 30) {
            const radius = 81 * Math.cos(lat * Math.PI / 180);
            const y = 81 * Math.sin(lat * Math.PI / 180);
            
            const curve = new THREE.EllipseCurve(0, 0, radius, radius, 0, 2 * Math.PI, false, 0);
            const points = curve.getPoints(64);
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const line = new THREE.Line(geometry, gridMaterial);
            line.rotation.x = Math.PI / 2;
            line.position.y = y;
            this.earthGroup.add(line);
        }
        
        // Longitude lines
        for (let lon = 0; lon < 360; lon += 30) {
            const points = [];
            for (let i = 0; i <= 64; i++) {
                const lat = (i / 64) * Math.PI - Math.PI / 2;
                const x = 81 * Math.cos(lat) * Math.cos(lon * Math.PI / 180);
                const y = 81 * Math.sin(lat);
                const z = 81 * Math.cos(lat) * Math.sin(lon * Math.PI / 180);
                points.push(new THREE.Vector3(x, y, z));
            }
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const line = new THREE.Line(geometry, gridMaterial);
            this.earthGroup.add(line);
        }
    }
    
    /**
     * Create atmosphere glow
     */
    createAtmosphere() {
        const atmosphereGeometry = new THREE.SphereGeometry(85, 64, 64);
        const atmosphereMaterial = new THREE.ShaderMaterial({
            vertexShader: `
                varying vec3 vNormal;
                void main() {
                    vNormal = normalize(normalMatrix * normal);
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                varying vec3 vNormal;
                void main() {
                    float intensity = pow(0.7 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.0);
                    gl_FragColor = vec4(0.0, 0.71, 0.85, 1.0) * intensity;
                }
            `,
            blending: THREE.AdditiveBlending,
            side: THREE.BackSide,
            transparent: true
        });
        
        this.atmosphere = new THREE.Mesh(atmosphereGeometry, atmosphereMaterial);
        this.earthGroup.add(this.atmosphere);
    }
    
    /**
     * Create zone classification rings
     */
    createZoneRings() {
        const zoneColors = {
            Z0: 0x10b981,  // Green
            Z1: 0x3b82f6,  // Blue
            Z2: 0xf59e0b,  // Amber
            Z3: 0xef4444   // Red
        };
        
        const zoneRadii = {
            Z0: 100,
            Z1: 115,
            Z2: 130,
            Z3: 145
        };
        
        Object.entries(zoneColors).forEach(([zone, color]) => {
            const radius = zoneRadii[zone];
            
            // Create ring
            const ringGeometry = new THREE.RingGeometry(radius - 0.5, radius + 0.5, 128);
            const ringMaterial = new THREE.MeshBasicMaterial({
                color: color,
                transparent: true,
                opacity: 0.3,
                side: THREE.DoubleSide
            });
            const ring = new THREE.Mesh(ringGeometry, ringMaterial);
            ring.rotation.x = Math.PI / 2;
            ring.userData.zone = zone;
            
            this.zonesGroup.add(ring);
        });
        
        this.earthGroup.add(this.zonesGroup);
    }
    
    /**
     * Generate demo validator nodes
     */
    generateDemoNodes() {
        const nodeCount = 256;
        const zoneDistribution = { Z0: 0.6, Z1: 0.25, Z2: 0.1, Z3: 0.05 };
        const zoneRadii = { Z0: 100, Z1: 115, Z2: 130, Z3: 145 };
        const zoneColors = {
            Z0: 0x10b981,
            Z1: 0x3b82f6,
            Z2: 0xf59e0b,
            Z3: 0xef4444
        };
        
        // Create node geometry template
        const nodeGeometry = new THREE.SphereGeometry(1.5, 16, 16);
        
        let currentIndex = 0;
        Object.entries(zoneDistribution).forEach(([zone, ratio]) => {
            const count = Math.floor(nodeCount * ratio);
            const radius = zoneRadii[zone];
            const color = zoneColors[zone];
            
            const nodeMaterial = new THREE.MeshPhongMaterial({
                color: color,
                emissive: color,
                emissiveIntensity: 0.5,
                transparent: true,
                opacity: 0.8
            });
            
            for (let i = 0; i < count; i++) {
                // Random position on sphere at zone radius
                const theta = Math.random() * Math.PI * 2;
                const phi = Math.acos(2 * Math.random() - 1);
                
                const x = radius * Math.sin(phi) * Math.cos(theta);
                const y = radius * Math.cos(phi);
                const z = radius * Math.sin(phi) * Math.sin(theta);
                
                const node = new THREE.Mesh(nodeGeometry, nodeMaterial.clone());
                node.position.set(x, y, z);
                node.userData = {
                    id: `val_${currentIndex.toString().padStart(4, '0')}`,
                    zone: zone,
                    status: Math.random() > 0.05 ? 'active' : 'jailed',
                    stake: 400000 + Math.floor(Math.random() * 100000)
                };
                
                this.nodes.push(node);
                this.nodesGroup.add(node);
                currentIndex++;
            }
        });
        
        this.earthGroup.add(this.nodesGroup);
        
        // Create connections between some nodes (BFT quorum visualization)
        this.createQuorumConnections();
    }
    
    /**
     * Create quorum connection lines
     */
    createQuorumConnections() {
        const connectionMaterial = new THREE.LineBasicMaterial({
            color: 0x00b4d8,
            transparent: true,
            opacity: 0.2
        });
        
        // Connect nearby nodes
        const activeNodes = this.nodes.filter(n => n.userData.status === 'active');
        const connectionCount = Math.min(100, activeNodes.length * 2);
        
        for (let i = 0; i < connectionCount; i++) {
            const nodeA = activeNodes[Math.floor(Math.random() * activeNodes.length)];
            const nodeB = activeNodes[Math.floor(Math.random() * activeNodes.length)];
            
            if (nodeA === nodeB) continue;
            
            const points = [
                nodeA.position.clone(),
                nodeB.position.clone()
            ];
            
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const line = new THREE.Line(geometry, connectionMaterial.clone());
            line.userData.nodeA = nodeA.userData.id;
            line.userData.nodeB = nodeB.userData.id;
            
            this.connections.push(line);
            this.connectionsGroup.add(line);
        }
        
        this.earthGroup.add(this.connectionsGroup);
    }
    
    /**
     * Activate this domain renderer
     */
    activate() {
        this.isActive = true;
        this.earthGroup.visible = true;
        console.log('[PlanetaryMap] Activated');
    }
    
    /**
     * Deactivate this domain renderer
     */
    deactivate() {
        this.isActive = false;
        this.earthGroup.visible = false;
        console.log('[PlanetaryMap] Deactivated');
    }
    
    /**
     * Update animation
     */
    update(delta, elapsed) {
        if (!this.isActive) return;
        
        // Rotate Earth slowly
        this.earthRotation += delta * 0.05;
        this.earthGroup.rotation.y = this.earthRotation;
        
        // Pulse nodes
        this.nodes.forEach((node, i) => {
            const pulse = Math.sin(elapsed * 2 + i * 0.1) * 0.5 + 0.5;
            node.scale.setScalar(0.8 + pulse * 0.4);
            
            if (node.userData.status === 'active') {
                node.material.opacity = 0.6 + pulse * 0.4;
            }
        });
        
        // Animate connections (data flow)
        this.connections.forEach((line, i) => {
            const pulse = Math.sin(elapsed * 3 + i * 0.5) * 0.5 + 0.5;
            line.material.opacity = 0.1 + pulse * 0.2;
        });
        
        // Pulse zone rings
        this.zonesGroup.children.forEach((ring, i) => {
            const pulse = Math.sin(elapsed + i * 0.5) * 0.1;
            ring.material.opacity = 0.2 + pulse;
        });
    }
    
    /**
     * Update node status from telemetry
     */
    updateNodeStatus(validatorId, status) {
        const node = this.nodes.find(n => n.userData.id === validatorId);
        if (node) {
            node.userData.status = status;
            
            // Update visual
            if (status === 'slashed' || status === 'jailed') {
                node.material.color.setHex(0xef4444);
                node.material.emissive.setHex(0xef4444);
            }
        }
    }
    
    /**
     * Highlight zone
     */
    highlightZone(zone) {
        this.zonesGroup.children.forEach(ring => {
            if (ring.userData.zone === zone) {
                ring.material.opacity = 0.6;
            } else {
                ring.material.opacity = 0.1;
            }
        });
    }
    
    /**
     * Get node at screen position
     */
    getNodeAtPosition(x, y, camera) {
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2(x, y);
        raycaster.setFromCamera(mouse, camera);
        
        const intersects = raycaster.intersectObjects(this.nodes);
        if (intersects.length > 0) {
            return intersects[0].object.userData;
        }
        return null;
    }
    
    /**
     * Cleanup
     */
    dispose() {
        this.nodes.forEach(node => {
            node.geometry.dispose();
            node.material.dispose();
        });
        
        this.connections.forEach(conn => {
            conn.geometry.dispose();
            conn.material.dispose();
        });
        
        this.scene.remove(this.earthGroup);
    }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PlanetaryMapRenderer;
}
