/**
 * QRATUM SOI Renderer
 * Main rendering engine for the Sovereign Operations Interface
 * 
 * Uses Three.js for 3D holographic visualization.
 * Purely reflective - renders from read-only state streams.
 * 
 * @version 1.0.0
 */

/**
 * Main SOI Application Class
 */
class SovereignOperationsInterface {
    constructor() {
        this.initialized = false;
        this.currentDomain = 'planetary';
        this.renderer = null;
        this.scene = null;
        this.camera = null;
        this.controls = null;
        this.animationId = null;
        
        // Domain renderers
        this.domains = {
            planetary: null,
            qradle: null,
            consensus: null,
            verticals: null
        };
        
        // UI elements
        this.elements = {};
        
        // Animation state
        this.clock = null;
    }
    
    /**
     * Initialize the SOI
     */
    async initialize() {
        console.log('[SOI] Initializing Sovereign Operations Interface...');
        
        try {
            // Show loader progress
            this.updateLoaderProgress(10, 'Initializing renderer...');
            
            // Cache DOM elements
            this.cacheElements();
            
            // Initialize Three.js
            this.updateLoaderProgress(30, 'Creating 3D environment...');
            await this.initializeRenderer();
            
            // Initialize telemetry bus
            this.updateLoaderProgress(50, 'Connecting to Aethernet...');
            this.initializeTelemetry();
            
            // Initialize domain renderers
            this.updateLoaderProgress(70, 'Loading domain modules...');
            await this.initializeDomains();
            
            // Setup event listeners
            this.updateLoaderProgress(85, 'Binding event handlers...');
            this.setupEventListeners();
            
            // Start animation loop
            this.updateLoaderProgress(95, 'Starting render loop...');
            this.startAnimation();
            
            // Hide loader and show interface
            this.updateLoaderProgress(100, 'QRATUM SOI READY');
            await this.showInterface();
            
            this.initialized = true;
            console.log('[SOI] Initialization complete');
            
        } catch (error) {
            console.error('[SOI] Initialization failed:', error);
            this.showError('Failed to initialize SOI: ' + error.message);
        }
    }
    
    /**
     * Cache DOM elements
     */
    cacheElements() {
        this.elements = {
            loader: document.getElementById('soi-loader'),
            loaderStatus: document.getElementById('loader-status'),
            loaderSubstatus: document.getElementById('loader-substatus'),
            loaderProgress: document.getElementById('loader-progress'),
            container: document.getElementById('soi-container'),
            viewport: document.getElementById('soi-viewport'),
            canvas: document.getElementById('soi-canvas'),
            panel: document.getElementById('soi-panel'),
            
            // HUD elements
            hudEpoch: document.getElementById('hud-epoch'),
            hudTime: document.getElementById('hud-time'),
            hudValidators: document.getElementById('hud-validators'),
            hudQuorum: document.getElementById('hud-quorum'),
            hudDomain: document.getElementById('hud-domain'),
            hudMerkle: document.getElementById('hud-merkle'),
            hudChain: document.getElementById('hud-chain'),
            
            // Telemetry panel
            telemetryFeed: document.getElementById('telemetry-feed'),
            telemetryStatus: document.getElementById('telemetry-status'),
            currentProof: document.getElementById('current-proof'),
            
            // Health displays
            healthActive: document.getElementById('health-active'),
            healthPending: document.getElementById('health-pending'),
            healthJailed: document.getElementById('health-jailed'),
            healthSlashed: document.getElementById('health-slashed'),
            
            // Zone displays
            zoneZ0: document.getElementById('zone-z0'),
            zoneZ1: document.getElementById('zone-z1'),
            zoneZ2: document.getElementById('zone-z2'),
            zoneZ3: document.getElementById('zone-z3'),
            
            // Trajectory
            collapseProbability: document.getElementById('collapse-probability'),
            trajectoryArc: document.getElementById('trajectory-arc'),
            precursorList: document.getElementById('precursor-list'),
            
            // Navigation
            navDomains: document.querySelectorAll('.nav-domain'),
            
            // Modal
            modal: document.getElementById('soi-modal'),
            modalTitle: document.getElementById('modal-title'),
            modalBody: document.getElementById('modal-body')
        };
    }
    
    /**
     * Update loader progress
     */
    updateLoaderProgress(percent, status) {
        if (this.elements.loaderProgress) {
            this.elements.loaderProgress.style.width = `${percent}%`;
        }
        if (this.elements.loaderSubstatus) {
            this.elements.loaderSubstatus.textContent = status;
        }
    }
    
    /**
     * Initialize Three.js renderer
     */
    async initializeRenderer() {
        const container = this.elements.viewport;
        const canvas = this.elements.canvas;
        
        // Create scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000814);
        this.scene.fog = new THREE.FogExp2(0x000814, 0.0008);
        
        // Create camera
        const aspect = container.clientWidth / container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 10000);
        this.camera.position.set(0, 150, 300);
        this.camera.lookAt(0, 0, 0);
        
        // Create renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: canvas,
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.2;
        
        // Add ambient light
        const ambientLight = new THREE.AmbientLight(0x001d3d, 0.5);
        this.scene.add(ambientLight);
        
        // Add directional light
        const dirLight = new THREE.DirectionalLight(0x00b4d8, 0.8);
        dirLight.position.set(100, 200, 100);
        this.scene.add(dirLight);
        
        // Add point lights for holographic effect
        const pointLight1 = new THREE.PointLight(0x00b4d8, 1, 500);
        pointLight1.position.set(0, 100, 0);
        this.scene.add(pointLight1);
        
        const pointLight2 = new THREE.PointLight(0x0077b6, 0.5, 400);
        pointLight2.position.set(-200, 50, 100);
        this.scene.add(pointLight2);
        
        // Create clock for animations
        this.clock = new THREE.Clock();
        
        // Add grid helper for reference
        const gridHelper = new THREE.GridHelper(500, 50, 0x003566, 0x001d3d);
        gridHelper.position.y = -50;
        this.scene.add(gridHelper);
        
        // Add starfield background
        this.createStarfield();
    }
    
    /**
     * Create starfield background
     */
    createStarfield() {
        const starGeometry = new THREE.BufferGeometry();
        const starCount = 2000;
        const positions = new Float32Array(starCount * 3);
        
        for (let i = 0; i < starCount * 3; i += 3) {
            positions[i] = (Math.random() - 0.5) * 2000;
            positions[i + 1] = (Math.random() - 0.5) * 2000;
            positions[i + 2] = (Math.random() - 0.5) * 2000;
        }
        
        starGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        
        const starMaterial = new THREE.PointsMaterial({
            color: 0x90e0ef,
            size: 1.5,
            transparent: true,
            opacity: 0.6
        });
        
        const starField = new THREE.Points(starGeometry, starMaterial);
        this.scene.add(starField);
    }
    
    /**
     * Initialize telemetry bus connection
     */
    initializeTelemetry() {
        // Initialize telemetry bus
        SOITelemetryBus.initialize();
        
        // Subscribe to events
        SOITelemetryBus.on('connect', (data) => {
            console.log('[SOI] Telemetry connected', data);
            this.updateTelemetryStatus(true, data.demo);
        });
        
        SOITelemetryBus.on('disconnect', (data) => {
            console.log('[SOI] Telemetry disconnected');
            this.updateTelemetryStatus(false);
        });
        
        SOITelemetryBus.on('qradle:state', (payload) => {
            this.updateQRADLEDisplay(payload);
        });
        
        SOITelemetryBus.on('aethernet:validator', (payload) => {
            this.updateValidatorDisplay(payload.stats);
        });
        
        SOITelemetryBus.on('aethernet:consensus', (payload) => {
            this.updateConsensusDisplay(payload);
        });
        
        SOITelemetryBus.on('aethernet:quorum', (payload) => {
            this.updateQuorumDisplay(payload);
        });
        
        SOITelemetryBus.on('trajectory:health', (payload) => {
            this.updateTrajectoryDisplay(payload);
        });
        
        SOITelemetryBus.on('trajectory:precursor', (payload) => {
            this.updatePrecursorDisplay(payload.signals);
        });
        
        SOITelemetryBus.on('telemetry:zone', (payload) => {
            this.updateZoneDisplay(payload.zones);
        });
        
        SOITelemetryBus.on('proof:chain', (payload) => {
            this.updateProofDisplay(payload);
        });
        
        // Start time update
        this.startTimeUpdate();
    }
    
    /**
     * Initialize domain renderers
     */
    async initializeDomains() {
        // These will be loaded from separate component files
        // For now, create basic placeholders
        
        if (typeof PlanetaryMapRenderer !== 'undefined') {
            this.domains.planetary = new PlanetaryMapRenderer(this.scene, this.camera);
            await this.domains.planetary.initialize();
        }
        
        if (typeof ExecutionTheaterRenderer !== 'undefined') {
            this.domains.qradle = new ExecutionTheaterRenderer(this.scene, this.camera);
        }
        
        if (typeof WarRoomRenderer !== 'undefined') {
            this.domains.consensus = new WarRoomRenderer(this.scene, this.camera);
        }
        
        if (typeof VerticalBaysRenderer !== 'undefined') {
            this.domains.verticals = new VerticalBaysRenderer(this.scene, this.camera);
        }
        
        // Activate initial domain
        this.activateDomain('planetary');
    }
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Window resize
        window.addEventListener('resize', () => this.handleResize());
        
        // Domain navigation
        this.elements.navDomains.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const domain = e.currentTarget.dataset.domain;
                this.switchDomain(domain);
            });
        });
        
        // Panel toggle
        const panelToggle = document.getElementById('panel-toggle');
        if (panelToggle) {
            panelToggle.addEventListener('click', () => this.togglePanel());
        }
        
        // Command buttons
        document.querySelectorAll('.cmd-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const action = e.currentTarget.dataset.action;
                this.handleCommand(action);
            });
        });
        
        // Modal close
        const modalClose = document.querySelector('.modal-close');
        if (modalClose) {
            modalClose.addEventListener('click', () => this.closeModal());
        }
        
        const modalCancel = document.querySelector('.modal-btn.cancel');
        if (modalCancel) {
            modalCancel.addEventListener('click', () => this.closeModal());
        }
        
        // Mouse interaction for 3D viewport
        this.setupViewportInteraction();
    }
    
    /**
     * Setup viewport mouse/touch interaction
     */
    setupViewportInteraction() {
        const viewport = this.elements.viewport;
        let isDragging = false;
        let previousMousePosition = { x: 0, y: 0 };
        
        viewport.addEventListener('mousedown', (e) => {
            isDragging = true;
            previousMousePosition = { x: e.clientX, y: e.clientY };
        });
        
        viewport.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            
            const deltaX = e.clientX - previousMousePosition.x;
            const deltaY = e.clientY - previousMousePosition.y;
            
            // Rotate camera around scene
            const rotationSpeed = 0.005;
            this.camera.position.applyAxisAngle(
                new THREE.Vector3(0, 1, 0),
                -deltaX * rotationSpeed
            );
            
            previousMousePosition = { x: e.clientX, y: e.clientY };
        });
        
        viewport.addEventListener('mouseup', () => {
            isDragging = false;
        });
        
        viewport.addEventListener('mouseleave', () => {
            isDragging = false;
        });
        
        // Zoom with scroll
        viewport.addEventListener('wheel', (e) => {
            e.preventDefault();
            const zoomSpeed = 0.1;
            const direction = new THREE.Vector3();
            this.camera.getWorldDirection(direction);
            
            this.camera.position.addScaledVector(direction, -e.deltaY * zoomSpeed);
        }, { passive: false });
    }
    
    /**
     * Start animation loop
     */
    startAnimation() {
        const animate = () => {
            this.animationId = requestAnimationFrame(animate);
            
            const delta = this.clock.getDelta();
            const elapsed = this.clock.getElapsedTime();
            
            // Update current domain renderer
            const currentRenderer = this.domains[this.currentDomain];
            if (currentRenderer && currentRenderer.update) {
                currentRenderer.update(delta, elapsed);
            }
            
            // Render scene
            this.renderer.render(this.scene, this.camera);
        };
        
        animate();
    }
    
    /**
     * Stop animation loop
     */
    stopAnimation() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }
    
    /**
     * Show interface after loading
     */
    async showInterface() {
        return new Promise(resolve => {
            setTimeout(() => {
                // Hide loader
                this.elements.loader.classList.add('hidden');
                
                // Show container
                this.elements.container.classList.add('visible');
                
                resolve();
            }, 1000);
        });
    }
    
    /**
     * Handle window resize
     */
    handleResize() {
        const container = this.elements.viewport;
        const width = container.clientWidth;
        const height = container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        
        this.renderer.setSize(width, height);
    }
    
    /**
     * Switch domain view
     */
    switchDomain(domain) {
        if (domain === this.currentDomain) return;
        
        console.log('[SOI] Switching to domain:', domain);
        
        // Update navigation
        this.elements.navDomains.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.domain === domain);
            btn.setAttribute('aria-selected', btn.dataset.domain === domain);
        });
        
        // Deactivate current domain
        const currentRenderer = this.domains[this.currentDomain];
        if (currentRenderer && currentRenderer.deactivate) {
            currentRenderer.deactivate();
        }
        
        // Activate new domain
        this.currentDomain = domain;
        this.activateDomain(domain);
        
        // Update HUD
        const domainNames = {
            planetary: 'PLANETARY NODE MAP',
            qradle: 'QRADLE EXECUTION THEATER',
            consensus: 'AETHERNET WAR ROOM',
            verticals: 'VERTICAL OPERATIONS'
        };
        this.elements.hudDomain.textContent = domainNames[domain] || domain.toUpperCase();
        
        // Update panel title
        const panelTitle = document.getElementById('panel-title');
        if (panelTitle) {
            const titles = {
                planetary: 'PLANETARY TELEMETRY',
                qradle: 'EXECUTION METRICS',
                consensus: 'CONSENSUS STATE',
                verticals: 'VERTICAL STATUS'
            };
            panelTitle.textContent = titles[domain] || 'TELEMETRY';
        }
    }
    
    /**
     * Activate domain renderer
     */
    activateDomain(domain) {
        const renderer = this.domains[domain];
        if (renderer && renderer.activate) {
            renderer.activate();
        }
    }
    
    /**
     * Toggle side panel
     */
    togglePanel() {
        this.elements.panel.classList.toggle('collapsed');
        const toggle = document.getElementById('panel-toggle');
        if (toggle) {
            toggle.textContent = this.elements.panel.classList.contains('collapsed') ? '⟩' : '⟨';
        }
    }
    
    /**
     * Handle command actions
     */
    handleCommand(action) {
        console.log('[SOI] Command:', action);
        
        switch (action) {
            case 'audit':
                this.showAuditTrail();
                break;
            case 'proof':
                this.showSystemProof();
                break;
            case 'rollback':
                this.showRollbackInterface();
                break;
            case 'biometric':
                this.showBiometricPrompt();
                break;
        }
    }
    
    /**
     * Show audit trail modal
     */
    showAuditTrail() {
        const events = SOITelemetryBus.getEventBuffer().slice(-50);
        
        let content = '<div class="audit-trail">';
        content += '<div class="audit-header">Recent Events (Last 50)</div>';
        events.reverse().forEach(event => {
            const time = new Date(event.timestamp).toISOString().split('T')[1].split('.')[0];
            content += `<div class="audit-item">
                <span class="audit-time">${time}</span>
                <span class="audit-type">${event.type}</span>
            </div>`;
        });
        content += '</div>';
        
        this.openModal('AUDIT TRAIL', content);
    }
    
    /**
     * Show system proof modal
     */
    showSystemProof() {
        const state = SOITelemetryBus.getState();
        
        let content = '<div class="proof-display">';
        content += '<div class="proof-section">';
        content += '<div class="proof-label">MERKLE ROOT</div>';
        content += `<div class="proof-value mono">${state.merkle.root || 'N/A'}</div>`;
        content += '</div>';
        content += '<div class="proof-section">';
        content += '<div class="proof-label">CHAIN LENGTH</div>';
        content += `<div class="proof-value">${state.merkle.chainLength.toLocaleString()}</div>`;
        content += '</div>';
        content += '<div class="proof-section">';
        content += '<div class="proof-label">LAST PROOF</div>';
        content += `<div class="proof-value mono">${state.lastProof || 'N/A'}</div>`;
        content += '</div>';
        content += '<div class="proof-section">';
        content += '<div class="proof-label">INTEGRITY STATUS</div>';
        content += `<div class="proof-value ${state.integrity ? 'success' : 'danger'}">${state.integrity ? 'VERIFIED' : 'COMPROMISED'}</div>`;
        content += '</div>';
        content += '</div>';
        
        this.openModal('SYSTEM PROOF', content);
    }
    
    /**
     * Show rollback interface (read-only view)
     */
    showRollbackInterface() {
        const proofs = SOITelemetryBus.getProofChain().slice(-10);
        
        let content = '<div class="rollback-display">';
        content += '<div class="rollback-notice">Rollback operations require biometric dual-control authorization.</div>';
        content += '<div class="checkpoint-list">';
        proofs.reverse().forEach((proof, index) => {
            const time = new Date(proof.timestamp).toISOString();
            content += `<div class="checkpoint-item">
                <div class="checkpoint-marker"></div>
                <div class="checkpoint-info">
                    <div class="checkpoint-height">Height: ${proof.height}</div>
                    <div class="checkpoint-time">${time}</div>
                    <div class="checkpoint-proof">${proof.proof.substring(0, 20)}...</div>
                </div>
            </div>`;
        });
        content += '</div></div>';
        
        this.openModal('TIME-VECTOR SELECTION', content);
    }
    
    /**
     * Show biometric prompt
     */
    showBiometricPrompt() {
        let content = '<div class="biometric-prompt">';
        content += '<div class="biometric-icon">👁️</div>';
        content += '<div class="biometric-text">Biometric Authorization Required</div>';
        content += '<div class="biometric-sub">This action requires dual-control verification.</div>';
        content += '<div class="biometric-scanner">';
        content += '<div class="scanner-ring"></div>';
        content += '<div class="scanner-ring"></div>';
        content += '<div class="scanner-ring"></div>';
        content += '</div>';
        content += '</div>';
        
        this.openModal('BIOMETRIC PORTAL', content);
    }
    
    /**
     * Open modal
     */
    openModal(title, content) {
        this.elements.modalTitle.textContent = title;
        this.elements.modalBody.innerHTML = content;
        this.elements.modal.classList.add('open');
        this.elements.modal.setAttribute('aria-hidden', 'false');
    }
    
    /**
     * Close modal
     */
    closeModal() {
        this.elements.modal.classList.remove('open');
        this.elements.modal.setAttribute('aria-hidden', 'true');
    }
    
    // ========================================
    // Display Update Methods (Read-Only)
    // ========================================
    
    updateTelemetryStatus(connected, demo = false) {
        if (this.elements.telemetryStatus) {
            this.elements.telemetryStatus.textContent = connected ? (demo ? 'DEMO' : 'LIVE') : 'OFFLINE';
            this.elements.telemetryStatus.style.color = connected ? 
                (demo ? 'var(--soi-warning)' : 'var(--soi-success)') : 'var(--soi-danger)';
        }
    }
    
    updateQRADLEDisplay(payload) {
        if (payload.epoch !== undefined && this.elements.hudEpoch) {
            this.elements.hudEpoch.textContent = payload.epoch.toLocaleString();
        }
        if (payload.merkle) {
            if (this.elements.hudMerkle) {
                const shortHash = payload.merkle.root ? 
                    payload.merkle.root.substring(0, 6) + '...' + payload.merkle.root.slice(-4) : 'N/A';
                this.elements.hudMerkle.textContent = shortHash;
            }
            if (this.elements.hudChain) {
                this.elements.hudChain.textContent = payload.merkle.chainLength.toLocaleString();
            }
        }
        
        this.addTelemetryEvent('qradle:state', 'State update');
    }
    
    updateValidatorDisplay(stats) {
        if (this.elements.healthActive) this.elements.healthActive.textContent = stats.active;
        if (this.elements.healthPending) this.elements.healthPending.textContent = stats.pending;
        if (this.elements.healthJailed) this.elements.healthJailed.textContent = stats.jailed;
        if (this.elements.healthSlashed) this.elements.healthSlashed.textContent = stats.slashed;
        if (this.elements.hudValidators) this.elements.hudValidators.textContent = stats.total;
        
        this.addTelemetryEvent('aethernet:validator', `Active: ${stats.active}`);
    }
    
    updateConsensusDisplay(payload) {
        this.addTelemetryEvent('aethernet:consensus', `Phase: ${payload.phase}`);
    }
    
    updateQuorumDisplay(payload) {
        if (this.elements.hudQuorum) {
            const percent = payload.percentage || 
                (payload.quorumPower / payload.totalPower * 100).toFixed(1);
            this.elements.hudQuorum.textContent = `${percent}%`;
            this.elements.hudQuorum.className = 'hud-value ' + 
                (parseFloat(percent) >= 66.7 ? 'success' : 'warning');
        }
        
        this.addTelemetryEvent('aethernet:quorum', `Quorum: ${payload.percentage || 'N/A'}%`);
    }
    
    updateTrajectoryDisplay(payload) {
        if (this.elements.collapseProbability) {
            const prob = (payload.collapseProbability * 100).toFixed(2);
            this.elements.collapseProbability.textContent = `${prob}%`;
            
            // Color based on risk level
            const color = payload.collapseProbability < 0.001 ? 'var(--soi-success)' :
                          payload.collapseProbability < 0.01 ? 'var(--soi-warning)' :
                          'var(--soi-danger)';
            this.elements.collapseProbability.style.color = color;
        }
        
        if (this.elements.trajectoryArc) {
            // Update gauge arc (0-110 dashoffset, 110 = 0%, 0 = 100%)
            const offset = 110 - (payload.collapseProbability * 100 * 1.1);
            this.elements.trajectoryArc.setAttribute('stroke-dashoffset', Math.max(0, offset));
        }
    }
    
    updatePrecursorDisplay(signals) {
        if (this.elements.precursorList) {
            if (!signals || signals.length === 0) {
                this.elements.precursorList.innerHTML = '<div class="no-precursors">No active precursors</div>';
            } else {
                this.elements.precursorList.innerHTML = signals.map(signal => 
                    `<div class="precursor-item">
                        <span class="precursor-dot amber"></span>
                        <span>${signal}</span>
                    </div>`
                ).join('');
            }
        }
    }
    
    updateZoneDisplay(zones) {
        if (this.elements.zoneZ0) this.elements.zoneZ0.textContent = zones.Z0;
        if (this.elements.zoneZ1) this.elements.zoneZ1.textContent = zones.Z1;
        if (this.elements.zoneZ2) this.elements.zoneZ2.textContent = zones.Z2;
        if (this.elements.zoneZ3) this.elements.zoneZ3.textContent = zones.Z3;
    }
    
    updateProofDisplay(payload) {
        if (this.elements.currentProof && payload.proof) {
            this.elements.currentProof.textContent = 
                payload.proof.substring(0, 6) + '...' + payload.proof.slice(-4);
        }
        
        this.addTelemetryEvent('proof:chain', `Height: ${payload.height}`);
    }
    
    /**
     * Add event to telemetry feed
     */
    addTelemetryEvent(type, data) {
        if (!this.elements.telemetryFeed) return;
        
        const time = new Date().toISOString().split('T')[1].split('.')[0];
        const eventEl = document.createElement('div');
        eventEl.className = 'telemetry-event';
        eventEl.innerHTML = `
            <span class="event-time">${time}</span>
            <span class="event-type">[${type}]</span>
            <span class="event-data">${data}</span>
        `;
        
        this.elements.telemetryFeed.insertBefore(eventEl, this.elements.telemetryFeed.firstChild);
        
        // Limit feed to 50 events
        while (this.elements.telemetryFeed.children.length > 50) {
            this.elements.telemetryFeed.removeChild(this.elements.telemetryFeed.lastChild);
        }
    }
    
    /**
     * Start time display update
     */
    startTimeUpdate() {
        const updateTime = () => {
            if (this.elements.hudTime) {
                const now = new Date();
                this.elements.hudTime.textContent = now.toISOString().split('T')[1].split('.')[0];
            }
        };
        
        updateTime();
        setInterval(updateTime, 1000);
    }
    
    /**
     * Show error message
     */
    showError(message) {
        console.error('[SOI]', message);
        if (this.elements.loaderSubstatus) {
            this.elements.loaderSubstatus.textContent = 'ERROR: ' + message;
            this.elements.loaderSubstatus.style.color = 'var(--soi-danger)';
        }
    }
    
    /**
     * Cleanup
     */
    destroy() {
        this.stopAnimation();
        SOITelemetryBus.disconnect();
        
        if (this.renderer) {
            this.renderer.dispose();
        }
    }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SovereignOperationsInterface;
}
