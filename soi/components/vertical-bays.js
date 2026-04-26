/**
 * QRATUM SOI - Vertical Operations Bays
 * 
 * Each vertical is a cinematic chamber:
 * - VITRA-E0: DNA helix cathedral with provenance rays
 * - CAPRA: Financial lattice towers
 * - JURIS: Court-grade ledger halls
 * - ECORA: Planetary energy mesh
 * - FLUXA: Logistics hypergraphs
 * 
 * @version 1.0.0
 */

class VerticalBaysRenderer {
    constructor(scene, camera) {
        this.scene = scene;
        this.camera = camera;
        this.isActive = false;
        
        // Groups
        this.baysGroup = null;
        this.currentVertical = 'VITRA';
        
        // Vertical chambers
        this.chambers = {};
    }
    
    /**
     * Initialize vertical bays
     */
    async initialize() {
        console.log('[VerticalBays] Initializing...');
        
        this.baysGroup = new THREE.Group();
        
        // Create all vertical chambers
        this.createVITRAChamber();
        this.createCAPRAChamber();
        this.createJURISChamber();
        this.createECORAChamber();
        this.createFLUXAChamber();
        
        // Show VITRA by default
        this.switchVertical('VITRA');
        
        // Initially hidden
        this.baysGroup.visible = false;
        this.scene.add(this.baysGroup);
        
        console.log('[VerticalBays] Initialization complete');
    }
    
    /**
     * Create VITRA-E0 chamber - DNA helix cathedral
     */
    createVITRAChamber() {
        const chamber = new THREE.Group();
        
        // DNA double helix
        const helixMaterial = new THREE.MeshPhongMaterial({
            color: 0x10b981,
            emissive: 0x10b981,
            emissiveIntensity: 0.3,
            transparent: true,
            opacity: 0.8
        });
        
        const sphereGeom = new THREE.SphereGeometry(2, 8, 8);
        const nucleotideCount = 50;
        
        for (let i = 0; i < nucleotideCount; i++) {
            const angle = (i / 10) * Math.PI * 2;
            const y = (i - nucleotideCount / 2) * 5;
            const radius = 30;
            
            // Strand 1
            const nucleotide1 = new THREE.Mesh(sphereGeom, helixMaterial.clone());
            nucleotide1.position.set(
                radius * Math.cos(angle),
                y,
                radius * Math.sin(angle)
            );
            chamber.add(nucleotide1);
            
            // Strand 2 (offset by 180 degrees)
            const nucleotide2 = new THREE.Mesh(sphereGeom, helixMaterial.clone());
            nucleotide2.position.set(
                radius * Math.cos(angle + Math.PI),
                y,
                radius * Math.sin(angle + Math.PI)
            );
            chamber.add(nucleotide2);
            
            // Base pair connection
            const connectionMaterial = new THREE.LineBasicMaterial({
                color: 0x00b4d8,
                transparent: true,
                opacity: 0.4
            });
            const points = [nucleotide1.position.clone(), nucleotide2.position.clone()];
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const connection = new THREE.Line(geometry, connectionMaterial);
            chamber.add(connection);
        }
        
        // Provenance rays
        const rayMaterial = new THREE.LineBasicMaterial({
            color: 0x8b5cf6,
            transparent: true,
            opacity: 0.3
        });
        
        for (let i = 0; i < 12; i++) {
            const angle = (i / 12) * Math.PI * 2;
            const points = [
                new THREE.Vector3(0, 0, 0),
                new THREE.Vector3(
                    Math.cos(angle) * 150,
                    Math.sin(angle * 0.5) * 50,
                    Math.sin(angle) * 150
                )
            ];
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const ray = new THREE.Line(geometry, rayMaterial);
            chamber.add(ray);
        }
        
        this.chambers.VITRA = chamber;
        this.baysGroup.add(chamber);
    }
    
    /**
     * Create CAPRA chamber - Financial lattice towers
     */
    createCAPRAChamber() {
        const chamber = new THREE.Group();
        
        const towerMaterial = new THREE.MeshPhongMaterial({
            color: 0x3b82f6,
            emissive: 0x1e40af,
            emissiveIntensity: 0.3,
            transparent: true,
            opacity: 0.7
        });
        
        // Create lattice towers
        const towerCount = 7;
        for (let i = 0; i < towerCount; i++) {
            const angle = (i / towerCount) * Math.PI * 2;
            const radius = 80;
            const height = 50 + Math.random() * 100;
            
            // Tower base
            const towerGeom = new THREE.CylinderGeometry(8, 12, height, 6);
            const tower = new THREE.Mesh(towerGeom, towerMaterial.clone());
            tower.position.set(
                radius * Math.cos(angle),
                height / 2 - 50,
                radius * Math.sin(angle)
            );
            chamber.add(tower);
            
            // Tower wireframe overlay
            const wireGeom = new THREE.CylinderGeometry(10, 14, height, 6);
            const wireMat = new THREE.MeshBasicMaterial({
                color: 0x60a5fa,
                wireframe: true,
                transparent: true,
                opacity: 0.3
            });
            const wire = new THREE.Mesh(wireGeom, wireMat);
            wire.position.copy(tower.position);
            chamber.add(wire);
        }
        
        // Central financial core
        const coreGeom = new THREE.IcosahedronGeometry(25, 1);
        const coreMat = new THREE.MeshPhongMaterial({
            color: 0x60a5fa,
            emissive: 0x3b82f6,
            emissiveIntensity: 0.5,
            transparent: true,
            opacity: 0.6
        });
        const core = new THREE.Mesh(coreGeom, coreMat);
        chamber.add(core);
        
        chamber.visible = false;
        this.chambers.CAPRA = chamber;
        this.baysGroup.add(chamber);
    }
    
    /**
     * Create JURIS chamber - Court-grade ledger halls
     */
    createJURISChamber() {
        const chamber = new THREE.Group();
        
        const columnMaterial = new THREE.MeshPhongMaterial({
            color: 0xfbbf24,
            emissive: 0xf59e0b,
            emissiveIntensity: 0.2,
            transparent: true,
            opacity: 0.8
        });
        
        // Create columned hall
        const columnCount = 6;
        const columnRadius = 100;
        
        for (let side = 0; side < 2; side++) {
            for (let i = 0; i < columnCount; i++) {
                const columnGeom = new THREE.CylinderGeometry(4, 4, 120, 16);
                const column = new THREE.Mesh(columnGeom, columnMaterial);
                column.position.set(
                    (side === 0 ? -1 : 1) * 60,
                    0,
                    (i - columnCount / 2) * 40
                );
                chamber.add(column);
            }
        }
        
        // Central ledger pedestal
        const pedestalGeom = new THREE.BoxGeometry(40, 10, 40);
        const pedestalMat = new THREE.MeshPhongMaterial({
            color: 0xfbbf24,
            emissive: 0xf59e0b,
            emissiveIntensity: 0.3
        });
        const pedestal = new THREE.Mesh(pedestalGeom, pedestalMat);
        pedestal.position.y = -50;
        chamber.add(pedestal);
        
        // Floating ledger pages
        const pageMaterial = new THREE.MeshBasicMaterial({
            color: 0xfef3c7,
            transparent: true,
            opacity: 0.6,
            side: THREE.DoubleSide
        });
        
        for (let i = 0; i < 5; i++) {
            const pageGeom = new THREE.PlaneGeometry(20, 30);
            const page = new THREE.Mesh(pageGeom, pageMaterial.clone());
            page.position.set(
                (Math.random() - 0.5) * 40,
                -30 + i * 15,
                (Math.random() - 0.5) * 40
            );
            page.rotation.y = Math.random() * Math.PI;
            page.userData.floatOffset = Math.random() * Math.PI * 2;
            chamber.add(page);
        }
        
        chamber.visible = false;
        this.chambers.JURIS = chamber;
        this.baysGroup.add(chamber);
    }
    
    /**
     * Create ECORA chamber - Planetary energy mesh
     */
    createECORAChamber() {
        const chamber = new THREE.Group();
        
        // Central energy sphere
        const sphereGeom = new THREE.SphereGeometry(30, 32, 32);
        const sphereMat = new THREE.MeshPhongMaterial({
            color: 0x22c55e,
            emissive: 0x15803d,
            emissiveIntensity: 0.5,
            transparent: true,
            opacity: 0.6
        });
        const sphere = new THREE.Mesh(sphereGeom, sphereMat);
        chamber.add(sphere);
        
        // Energy mesh lines
        const meshMaterial = new THREE.LineBasicMaterial({
            color: 0x4ade80,
            transparent: true,
            opacity: 0.4
        });
        
        const meshPoints = 20;
        for (let i = 0; i < meshPoints; i++) {
            for (let j = 0; j < meshPoints; j++) {
                const phi = (i / meshPoints) * Math.PI * 2;
                const theta = (j / meshPoints) * Math.PI;
                const radius = 80;
                
                const x = radius * Math.sin(theta) * Math.cos(phi);
                const y = radius * Math.cos(theta);
                const z = radius * Math.sin(theta) * Math.sin(phi);
                
                // Connect to center
                if (Math.random() > 0.7) {
                    const points = [
                        new THREE.Vector3(x, y, z),
                        new THREE.Vector3(x * 0.4, y * 0.4, z * 0.4)
                    ];
                    const geometry = new THREE.BufferGeometry().setFromPoints(points);
                    const line = new THREE.Line(geometry, meshMaterial);
                    chamber.add(line);
                }
            }
        }
        
        // Orbital energy rings
        for (let i = 0; i < 3; i++) {
            const ringGeom = new THREE.RingGeometry(60 + i * 20, 62 + i * 20, 64);
            const ringMat = new THREE.MeshBasicMaterial({
                color: 0x4ade80,
                transparent: true,
                opacity: 0.3,
                side: THREE.DoubleSide
            });
            const ring = new THREE.Mesh(ringGeom, ringMat);
            ring.rotation.x = Math.PI / 2 + (i * 0.3);
            ring.rotation.z = i * 0.5;
            chamber.add(ring);
        }
        
        chamber.visible = false;
        this.chambers.ECORA = chamber;
        this.baysGroup.add(chamber);
    }
    
    /**
     * Create FLUXA chamber - Logistics hypergraphs
     */
    createFLUXAChamber() {
        const chamber = new THREE.Group();
        
        // Hypergraph nodes
        const nodeGeom = new THREE.OctahedronGeometry(4, 0);
        const nodeMat = new THREE.MeshPhongMaterial({
            color: 0xf97316,
            emissive: 0xea580c,
            emissiveIntensity: 0.4
        });
        
        const nodes = [];
        const nodeCount = 30;
        
        for (let i = 0; i < nodeCount; i++) {
            const node = new THREE.Mesh(nodeGeom, nodeMat.clone());
            node.position.set(
                (Math.random() - 0.5) * 200,
                (Math.random() - 0.5) * 100,
                (Math.random() - 0.5) * 200
            );
            nodes.push(node);
            chamber.add(node);
        }
        
        // Hyperedges (connecting multiple nodes)
        const edgeMaterial = new THREE.LineBasicMaterial({
            color: 0xfb923c,
            transparent: true,
            opacity: 0.3
        });
        
        for (let i = 0; i < nodeCount; i++) {
            const node = nodes[i];
            // Connect to 2-3 random other nodes
            const connections = 2 + Math.floor(Math.random() * 2);
            for (let j = 0; j < connections; j++) {
                const targetIdx = Math.floor(Math.random() * nodeCount);
                if (targetIdx !== i) {
                    const target = nodes[targetIdx];
                    const points = [node.position.clone(), target.position.clone()];
                    const geometry = new THREE.BufferGeometry().setFromPoints(points);
                    const edge = new THREE.Line(geometry, edgeMaterial);
                    chamber.add(edge);
                }
            }
        }
        
        // Central routing hub
        const hubGeom = new THREE.TorusKnotGeometry(20, 5, 100, 16);
        const hubMat = new THREE.MeshPhongMaterial({
            color: 0xf97316,
            emissive: 0xea580c,
            emissiveIntensity: 0.3,
            transparent: true,
            opacity: 0.6
        });
        const hub = new THREE.Mesh(hubGeom, hubMat);
        chamber.add(hub);
        
        chamber.visible = false;
        this.chambers.FLUXA = chamber;
        this.baysGroup.add(chamber);
    }
    
    /**
     * Switch to different vertical
     */
    switchVertical(vertical) {
        Object.keys(this.chambers).forEach(key => {
            this.chambers[key].visible = (key === vertical);
        });
        this.currentVertical = vertical;
        console.log('[VerticalBays] Switched to', vertical);
    }
    
    /**
     * Activate
     */
    activate() {
        this.isActive = true;
        this.baysGroup.visible = true;
        console.log('[VerticalBays] Activated');
    }
    
    /**
     * Deactivate
     */
    deactivate() {
        this.isActive = false;
        this.baysGroup.visible = false;
        console.log('[VerticalBays] Deactivated');
    }
    
    /**
     * Update animation
     */
    update(delta, elapsed) {
        if (!this.isActive) return;
        
        const chamber = this.chambers[this.currentVertical];
        if (!chamber) return;
        
        // Rotate current chamber slowly
        chamber.rotation.y = elapsed * 0.1;
        
        // Specific animations per vertical
        switch (this.currentVertical) {
            case 'VITRA':
                // DNA helix rotation
                chamber.rotation.y = elapsed * 0.2;
                break;
            case 'CAPRA':
                // Pulse core
                chamber.children.forEach(child => {
                    if (child.geometry.type === 'IcosahedronGeometry') {
                        child.rotation.y = elapsed * 0.5;
                        child.rotation.x = elapsed * 0.3;
                    }
                });
                break;
            case 'JURIS':
                // Float ledger pages
                chamber.children.forEach(child => {
                    if (child.userData.floatOffset !== undefined) {
                        child.position.y += Math.sin(elapsed + child.userData.floatOffset) * 0.02;
                    }
                });
                break;
            case 'ECORA':
                // Rotate energy rings
                let ringIdx = 0;
                chamber.children.forEach(child => {
                    if (child.geometry && child.geometry.type === 'RingGeometry') {
                        child.rotation.z = elapsed * (0.1 + ringIdx * 0.05);
                        ringIdx++;
                    }
                });
                break;
            case 'FLUXA':
                // Rotate torus knot hub
                chamber.children.forEach(child => {
                    if (child.geometry && child.geometry.type === 'TorusKnotGeometry') {
                        child.rotation.x = elapsed * 0.3;
                        child.rotation.y = elapsed * 0.2;
                    }
                });
                break;
        }
    }
    
    /**
     * Cleanup
     */
    dispose() {
        this.scene.remove(this.baysGroup);
    }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VerticalBaysRenderer;
}
