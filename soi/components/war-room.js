/**
 * QRATUM SOI - Aethernet Consensus War Room
 * 
 * Visualization of BFT consensus:
 * - Validator lifecycle rings
 * - Slashing heat maps
 * - Trajectory-aware collapse precursors (amber → crimson)
 * - Self-suspension triggers as gravitational wells
 * 
 * @version 1.0.0
 */

class WarRoomRenderer {
    constructor(scene, camera) {
        this.scene = scene;
        this.camera = camera;
        this.isActive = false;
        
        // Groups
        this.warRoomGroup = null;
        this.validatorRingsGroup = null;
        this.heatMapGroup = null;
        this.trajectoryGroup = null;
        
        // State
        this.validators = [];
        this.slashingEvents = [];
    }
    
    /**
     * Initialize the war room
     */
    async initialize() {
        console.log('[WarRoom] Initializing...');
        
        this.warRoomGroup = new THREE.Group();
        this.validatorRingsGroup = new THREE.Group();
        this.heatMapGroup = new THREE.Group();
        this.trajectoryGroup = new THREE.Group();
        
        // Create validator lifecycle rings
        this.createValidatorRings();
        
        // Create slashing heat map
        this.createSlashingHeatMap();
        
        // Create trajectory visualization
        this.createTrajectoryVisualization();
        
        // Add to war room group
        this.warRoomGroup.add(this.validatorRingsGroup);
        this.warRoomGroup.add(this.heatMapGroup);
        this.warRoomGroup.add(this.trajectoryGroup);
        
        // Initially hidden
        this.warRoomGroup.visible = false;
        this.scene.add(this.warRoomGroup);
        
        console.log('[WarRoom] Initialization complete');
    }
    
    /**
     * Create validator lifecycle rings
     */
    createValidatorRings() {
        const statusColors = {
            active: 0x10b981,
            pending: 0xf59e0b,
            jailed: 0xef4444,
            slashed: 0x7c3aed
        };
        
        const statuses = ['active', 'pending', 'jailed', 'slashed'];
        const counts = [243, 8, 3, 2];
        
        statuses.forEach((status, index) => {
            const radius = 60 + index * 30;
            const count = counts[index];
            const color = statusColors[status];
            
            // Create ring
            const ringGeometry = new THREE.RingGeometry(radius - 1, radius + 1, 128);
            const ringMaterial = new THREE.MeshBasicMaterial({
                color: color,
                transparent: true,
                opacity: 0.3,
                side: THREE.DoubleSide
            });
            const ring = new THREE.Mesh(ringGeometry, ringMaterial);
            ring.rotation.x = Math.PI / 2;
            ring.userData = { status, count };
            this.validatorRingsGroup.add(ring);
            
            // Add validator indicators on ring
            const indicatorGeometry = new THREE.SphereGeometry(2, 8, 8);
            const indicatorMaterial = new THREE.MeshPhongMaterial({
                color: color,
                emissive: color,
                emissiveIntensity: 0.5
            });
            
            const indicatorCount = Math.min(count, 50); // Limit for performance
            for (let i = 0; i < indicatorCount; i++) {
                const angle = (i / indicatorCount) * Math.PI * 2;
                const indicator = new THREE.Mesh(indicatorGeometry, indicatorMaterial.clone());
                indicator.position.set(
                    radius * Math.cos(angle),
                    0,
                    radius * Math.sin(angle)
                );
                indicator.userData = { status, index: i };
                this.validators.push(indicator);
                this.validatorRingsGroup.add(indicator);
            }
        });
    }
    
    /**
     * Create slashing heat map
     */
    createSlashingHeatMap() {
        // Create ground plane for heat map
        const planeGeometry = new THREE.PlaneGeometry(300, 200, 50, 30);
        const planeMaterial = new THREE.MeshBasicMaterial({
            color: 0x001d3d,
            transparent: true,
            opacity: 0.5,
            wireframe: true
        });
        const plane = new THREE.Mesh(planeGeometry, planeMaterial);
        plane.rotation.x = -Math.PI / 2;
        plane.position.y = -40;
        this.heatMapGroup.add(plane);
        
        // Add heat spots for slashing events
        const heatSpotGeometry = new THREE.CircleGeometry(15, 32);
        for (let i = 0; i < 5; i++) {
            const heatMaterial = new THREE.MeshBasicMaterial({
                color: 0xef4444,
                transparent: true,
                opacity: 0.3 + Math.random() * 0.3
            });
            const heatSpot = new THREE.Mesh(heatSpotGeometry, heatMaterial);
            heatSpot.rotation.x = -Math.PI / 2;
            heatSpot.position.set(
                (Math.random() - 0.5) * 200,
                -39,
                (Math.random() - 0.5) * 150
            );
            heatSpot.userData = {
                type: 'slashing_event',
                intensity: Math.random()
            };
            this.slashingEvents.push(heatSpot);
            this.heatMapGroup.add(heatSpot);
        }
    }
    
    /**
     * Create trajectory visualization
     */
    createTrajectoryVisualization() {
        // Create central monitoring sphere
        const sphereGeometry = new THREE.SphereGeometry(20, 32, 32);
        const sphereMaterial = new THREE.MeshPhongMaterial({
            color: 0x10b981,
            emissive: 0x10b981,
            emissiveIntensity: 0.3,
            transparent: true,
            opacity: 0.6
        });
        const healthSphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
        healthSphere.position.y = 50;
        healthSphere.userData = { type: 'health_indicator' };
        this.trajectoryGroup.add(healthSphere);
        
        // Create orbital precursor indicators
        for (let i = 0; i < 3; i++) {
            const orbitRadius = 40 + i * 15;
            
            // Orbit ring
            const orbitGeometry = new THREE.RingGeometry(orbitRadius - 0.3, orbitRadius + 0.3, 64);
            const orbitMaterial = new THREE.MeshBasicMaterial({
                color: 0x00b4d8,
                transparent: true,
                opacity: 0.2,
                side: THREE.DoubleSide
            });
            const orbit = new THREE.Mesh(orbitGeometry, orbitMaterial);
            orbit.position.y = 50;
            this.trajectoryGroup.add(orbit);
            
            // Precursor indicator
            const indicatorGeometry = new THREE.OctahedronGeometry(3, 0);
            const indicatorMaterial = new THREE.MeshPhongMaterial({
                color: 0x10b981, // Will change to amber/crimson on precursor detection
                emissive: 0x10b981,
                emissiveIntensity: 0.5
            });
            const indicator = new THREE.Mesh(indicatorGeometry, indicatorMaterial);
            indicator.position.set(orbitRadius, 50, 0);
            indicator.userData = {
                type: 'precursor',
                orbitRadius,
                orbitSpeed: 0.5 - i * 0.1,
                angle: Math.random() * Math.PI * 2
            };
            this.trajectoryGroup.add(indicator);
        }
        
        // Create gravitational well (self-suspension trigger)
        const wellGeometry = new THREE.TorusGeometry(30, 3, 16, 100);
        const wellMaterial = new THREE.MeshPhongMaterial({
            color: 0x7c3aed,
            emissive: 0x5b21b6,
            emissiveIntensity: 0.3,
            transparent: true,
            opacity: 0.4
        });
        const well = new THREE.Mesh(wellGeometry, wellMaterial);
        well.position.set(100, 50, 0);
        well.userData = { type: 'suspension_trigger' };
        this.trajectoryGroup.add(well);
    }
    
    /**
     * Activate
     */
    activate() {
        this.isActive = true;
        this.warRoomGroup.visible = true;
        console.log('[WarRoom] Activated');
    }
    
    /**
     * Deactivate
     */
    deactivate() {
        this.isActive = false;
        this.warRoomGroup.visible = false;
        console.log('[WarRoom] Deactivated');
    }
    
    /**
     * Update animation
     */
    update(delta, elapsed) {
        if (!this.isActive) return;
        
        // Rotate validator rings
        this.validatorRingsGroup.rotation.y = elapsed * 0.1;
        
        // Pulse validators
        this.validators.forEach((validator, i) => {
            const pulse = Math.sin(elapsed * 2 + i * 0.3) * 0.5 + 0.5;
            validator.scale.setScalar(0.8 + pulse * 0.4);
        });
        
        // Pulse slashing heat spots
        this.slashingEvents.forEach((spot, i) => {
            const pulse = Math.sin(elapsed * 1.5 + i * 0.5) * 0.2 + 0.3;
            spot.material.opacity = pulse;
        });
        
        // Animate precursor orbits
        this.trajectoryGroup.children.forEach(child => {
            if (child.userData.type === 'precursor') {
                child.userData.angle += delta * child.userData.orbitSpeed;
                const angle = child.userData.angle;
                const radius = child.userData.orbitRadius;
                child.position.x = radius * Math.cos(angle);
                child.position.z = radius * Math.sin(angle);
                child.rotation.y = elapsed;
            }
        });
        
        // Rotate gravitational well
        this.trajectoryGroup.children.forEach(child => {
            if (child.userData.type === 'suspension_trigger') {
                child.rotation.x = elapsed * 0.5;
                child.rotation.z = elapsed * 0.3;
            }
        });
    }
    
    /**
     * Update health status
     */
    updateHealthStatus(healthScore, collapseProbability) {
        const healthSphere = this.trajectoryGroup.children.find(
            c => c.userData.type === 'health_indicator'
        );
        
        if (healthSphere) {
            // Color based on health
            if (healthScore > 0.9) {
                healthSphere.material.color.setHex(0x10b981);
                healthSphere.material.emissive.setHex(0x10b981);
            } else if (healthScore > 0.7) {
                healthSphere.material.color.setHex(0xf59e0b);
                healthSphere.material.emissive.setHex(0xf59e0b);
            } else {
                healthSphere.material.color.setHex(0xef4444);
                healthSphere.material.emissive.setHex(0xef4444);
            }
        }
    }
    
    /**
     * Show precursor alert
     */
    showPrecursorAlert(severity) {
        const precursors = this.trajectoryGroup.children.filter(
            c => c.userData.type === 'precursor'
        );
        
        const color = severity === 'crimson' ? 0xef4444 : 0xf59e0b;
        precursors.forEach(p => {
            p.material.color.setHex(color);
            p.material.emissive.setHex(color);
        });
    }
    
    /**
     * Cleanup
     */
    dispose() {
        this.scene.remove(this.warRoomGroup);
    }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WarRoomRenderer;
}
