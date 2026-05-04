/**
 * QRATUM SOI - QRADLE Execution Theater
 * 
 * Visualization of QRADLE deterministic execution:
 * - Deterministic state machine lattice
 * - Rollback vectors as reversible time branches
 * - Fatal invariant violations as red horizon fractures
 * - ZK proofs streaming as quantum-noise overlays
 * 
 * @version 1.0.0
 */

class ExecutionTheaterRenderer {
    constructor(scene, camera) {
        this.scene = scene;
        this.camera = camera;
        this.isActive = false;
        
        // Groups
        this.theaterGroup = null;
        this.latticeGroup = null;
        this.branchesGroup = null;
        this.proofGroup = null;
        
        // State
        this.executions = [];
        this.checkpoints = [];
    }
    
    /**
     * Initialize the execution theater
     */
    async initialize() {
        console.log('[ExecutionTheater] Initializing...');
        
        this.theaterGroup = new THREE.Group();
        this.latticeGroup = new THREE.Group();
        this.branchesGroup = new THREE.Group();
        this.proofGroup = new THREE.Group();
        
        // Create state machine lattice
        this.createStateLattice();
        
        // Create time branches
        this.createTimeBranches();
        
        // Create proof particle system
        this.createProofOverlay();
        
        // Add to theater group
        this.theaterGroup.add(this.latticeGroup);
        this.theaterGroup.add(this.branchesGroup);
        this.theaterGroup.add(this.proofGroup);
        
        // Initially hidden
        this.theaterGroup.visible = false;
        this.scene.add(this.theaterGroup);
        
        console.log('[ExecutionTheater] Initialization complete');
    }
    
    /**
     * Create state machine lattice visualization
     */
    createStateLattice() {
        const nodeCount = 50;
        const nodeGeometry = new THREE.OctahedronGeometry(3, 0);
        const nodeMaterial = new THREE.MeshPhongMaterial({
            color: 0x00b4d8,
            emissive: 0x003566,
            emissiveIntensity: 0.5,
            transparent: true,
            opacity: 0.8
        });
        
        // Create execution nodes in a grid pattern
        for (let x = 0; x < 10; x++) {
            for (let y = 0; y < 5; y++) {
                const node = new THREE.Mesh(nodeGeometry, nodeMaterial.clone());
                node.position.set(
                    (x - 4.5) * 30,
                    (y - 2) * 25,
                    0
                );
                node.userData = {
                    type: 'execution',
                    index: x + y * 10,
                    status: 'completed'
                };
                
                this.executions.push(node);
                this.latticeGroup.add(node);
            }
        }
        
        // Connect nodes with edges
        const edgeMaterial = new THREE.LineBasicMaterial({
            color: 0x0077b6,
            transparent: true,
            opacity: 0.4
        });
        
        this.executions.forEach((node, i) => {
            // Connect to next in row
            if ((i + 1) % 10 !== 0 && i + 1 < this.executions.length) {
                const nextNode = this.executions[i + 1];
                const points = [node.position.clone(), nextNode.position.clone()];
                const geometry = new THREE.BufferGeometry().setFromPoints(points);
                const edge = new THREE.Line(geometry, edgeMaterial);
                this.latticeGroup.add(edge);
            }
            
            // Connect to node below
            if (i + 10 < this.executions.length) {
                const belowNode = this.executions[i + 10];
                const points = [node.position.clone(), belowNode.position.clone()];
                const geometry = new THREE.BufferGeometry().setFromPoints(points);
                const edge = new THREE.Line(geometry, edgeMaterial);
                this.latticeGroup.add(edge);
            }
        });
    }
    
    /**
     * Create rollback time branches
     */
    createTimeBranches() {
        const checkpointCount = 5;
        const checkpointGeometry = new THREE.CylinderGeometry(5, 5, 2, 32);
        const checkpointMaterial = new THREE.MeshPhongMaterial({
            color: 0x8b5cf6,
            emissive: 0x5b21b6,
            emissiveIntensity: 0.5,
            transparent: true,
            opacity: 0.8
        });
        
        // Create checkpoints along timeline
        for (let i = 0; i < checkpointCount; i++) {
            const checkpoint = new THREE.Mesh(checkpointGeometry, checkpointMaterial.clone());
            checkpoint.position.set((i - 2) * 60, -80, 0);
            checkpoint.rotation.x = Math.PI / 2;
            checkpoint.userData = {
                type: 'checkpoint',
                index: i,
                epoch: 127838 + i
            };
            
            this.checkpoints.push(checkpoint);
            this.branchesGroup.add(checkpoint);
            
            // Create branch lines
            if (i > 0) {
                const prevCheckpoint = this.checkpoints[i - 1];
                const branchMaterial = new THREE.LineBasicMaterial({
                    color: 0x8b5cf6,
                    transparent: true,
                    opacity: 0.6
                });
                
                const points = [prevCheckpoint.position.clone(), checkpoint.position.clone()];
                const geometry = new THREE.BufferGeometry().setFromPoints(points);
                const branch = new THREE.Line(geometry, branchMaterial);
                this.branchesGroup.add(branch);
            }
            
            // Create rollback vector (upward arrow)
            const arrowGeometry = new THREE.ConeGeometry(3, 10, 16);
            const arrowMaterial = new THREE.MeshPhongMaterial({
                color: 0x10b981,
                emissive: 0x10b981,
                emissiveIntensity: 0.3,
                transparent: true,
                opacity: 0.6
            });
            const arrow = new THREE.Mesh(arrowGeometry, arrowMaterial);
            arrow.position.set(checkpoint.position.x, checkpoint.position.y + 15, 0);
            this.branchesGroup.add(arrow);
        }
    }
    
    /**
     * Create ZK proof particle overlay
     */
    createProofOverlay() {
        const particleCount = 500;
        const positions = new Float32Array(particleCount * 3);
        const colors = new Float32Array(particleCount * 3);
        
        for (let i = 0; i < particleCount * 3; i += 3) {
            positions[i] = (Math.random() - 0.5) * 400;
            positions[i + 1] = (Math.random() - 0.5) * 200;
            positions[i + 2] = (Math.random() - 0.5) * 100;
            
            colors[i] = 0.0;     // R
            colors[i + 1] = 0.7; // G
            colors[i + 2] = 0.85; // B
        }
        
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        
        const material = new THREE.PointsMaterial({
            size: 2,
            vertexColors: true,
            transparent: true,
            opacity: 0.4,
            blending: THREE.AdditiveBlending
        });
        
        const particles = new THREE.Points(geometry, material);
        particles.userData.type = 'proof_particles';
        this.proofGroup.add(particles);
    }
    
    /**
     * Activate
     */
    activate() {
        this.isActive = true;
        this.theaterGroup.visible = true;
        console.log('[ExecutionTheater] Activated');
    }
    
    /**
     * Deactivate
     */
    deactivate() {
        this.isActive = false;
        this.theaterGroup.visible = false;
        console.log('[ExecutionTheater] Deactivated');
    }
    
    /**
     * Update animation
     */
    update(delta, elapsed) {
        if (!this.isActive) return;
        
        // Pulse execution nodes
        this.executions.forEach((node, i) => {
            const pulse = Math.sin(elapsed * 2 + i * 0.2) * 0.5 + 0.5;
            node.scale.setScalar(0.8 + pulse * 0.4);
            node.rotation.y = elapsed * 0.5;
        });
        
        // Animate checkpoints
        this.checkpoints.forEach((checkpoint, i) => {
            const pulse = Math.sin(elapsed * 1.5 + i * 0.5) * 0.3 + 0.7;
            checkpoint.material.opacity = pulse;
        });
        
        // Animate proof particles
        const particles = this.proofGroup.children[0];
        if (particles) {
            const positions = particles.geometry.attributes.position.array;
            for (let i = 0; i < positions.length; i += 3) {
                positions[i + 1] += Math.sin(elapsed + i) * 0.1;
            }
            particles.geometry.attributes.position.needsUpdate = true;
        }
    }
    
    /**
     * Show invariant violation
     */
    showInvariantViolation(index) {
        if (index < this.executions.length) {
            const node = this.executions[index];
            node.material.color.setHex(0xef4444);
            node.material.emissive.setHex(0xef4444);
            node.userData.status = 'violated';
        }
    }
    
    /**
     * Cleanup
     */
    dispose() {
        this.scene.remove(this.theaterGroup);
    }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ExecutionTheaterRenderer;
}
