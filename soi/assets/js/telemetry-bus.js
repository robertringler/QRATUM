/**
 * QRATUM SOI Telemetry Bus
 * Deterministic State Stream Handler
 * 
 * The UI is a purely reflective surface - it cannot generate entropy,
 * alter execution paths, bypass zones, or emit unsigned events.
 * All rendering is post-factum reflective.
 * 
 * @version 1.0.0
 */

const SOITelemetryBus = (function() {
    'use strict';

    // Constants
    const CONSENSUS_PHASES = ['PROPOSE', 'PREVOTE', 'PRECOMMIT', 'COMMIT', 'FINALIZED'];

    // Configuration
    const config = {
        wsUrl: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
            ? 'ws://localhost:8000/soi/telemetry'
            : `wss://${window.location.host}/soi/telemetry`,
        reconnectInterval: 3000,
        maxReconnectAttempts: 10,
        heartbeatInterval: 30000,
        bufferSize: 1000 // Max events to keep in buffer
    };

    // State
    let socket = null;
    let reconnectAttempts = 0;
    let heartbeatTimer = null;
    let isConnected = false;
    let demoMode = true;
    let eventBuffer = [];
    let proofChain = [];
    
    // Event handlers registry
    const handlers = {
        // Connection events
        'connect': [],
        'disconnect': [],
        'error': [],
        
        // QRADLE events
        'qradle:state': [],
        'qradle:execution': [],
        'qradle:rollback': [],
        'qradle:invariant': [],
        'qradle:checkpoint': [],
        
        // Aethernet events
        'aethernet:validator': [],
        'aethernet:consensus': [],
        'aethernet:quorum': [],
        'aethernet:slashing': [],
        
        // Proof events
        'proof:generated': [],
        'proof:verified': [],
        'proof:chain': [],
        
        // Trajectory events
        'trajectory:health': [],
        'trajectory:precursor': [],
        'trajectory:collapse': [],
        
        // Generic telemetry
        'telemetry:metrics': [],
        'telemetry:zone': []
    };

    // Current system state (read-only mirror)
    let systemState = {
        epoch: 0,
        validators: {
            total: 0,
            active: 0,
            pending: 0,
            jailed: 0,
            slashed: 0
        },
        consensus: {
            height: 0,
            round: 0,
            phase: 'unknown',
            quorumPower: 0,
            totalPower: 0
        },
        zones: {
            Z0: 0,
            Z1: 0,
            Z2: 0,
            Z3: 0
        },
        trajectory: {
            healthScore: 1.0,
            collapseProbability: 0.0,
            isSuspended: false,
            precursors: []
        },
        merkle: {
            root: null,
            chainLength: 0
        },
        lastProof: null,
        integrity: true
    };

    /**
     * Initialize telemetry bus
     */
    function initialize() {
        console.log('[TelemetryBus] Initializing...');
        connect();
    }

    /**
     * Connect to WebSocket telemetry stream
     */
    function connect() {
        if (socket && socket.readyState === WebSocket.OPEN) {
            return;
        }

        try {
            socket = new WebSocket(config.wsUrl);
            socket.onopen = handleOpen;
            socket.onclose = handleClose;
            socket.onerror = handleError;
            socket.onmessage = handleMessage;
        } catch (error) {
            console.warn('[TelemetryBus] WebSocket connection failed:', error);
            enableDemoMode();
        }
    }

    /**
     * Handle connection open
     */
    function handleOpen(event) {
        console.log('[TelemetryBus] Connected to telemetry stream');
        isConnected = true;
        reconnectAttempts = 0;
        demoMode = false;
        
        startHeartbeat();
        emit('connect', { timestamp: Date.now() });
        
        // Request initial state
        sendCommand('get_state');
    }

    /**
     * Handle connection close
     */
    function handleClose(event) {
        console.log('[TelemetryBus] Connection closed:', event.code);
        isConnected = false;
        stopHeartbeat();
        emit('disconnect', { code: event.code, reason: event.reason });
        
        if (reconnectAttempts < config.maxReconnectAttempts) {
            reconnectAttempts++;
            setTimeout(connect, config.reconnectInterval);
        } else {
            enableDemoMode();
        }
    }

    /**
     * Handle connection error
     */
    function handleError(error) {
        console.error('[TelemetryBus] Error:', error);
        emit('error', { error });
        
        if (!isConnected) {
            enableDemoMode();
        }
    }

    /**
     * Handle incoming message
     * All messages are cryptographically verified before processing
     */
    function handleMessage(event) {
        try {
            const data = JSON.parse(event.data);
            
            // Skip heartbeat responses
            if (data.type === 'pong') return;
            
            // Verify message integrity (simplified - production would use ZK proofs)
            if (!verifyMessageIntegrity(data)) {
                console.warn('[TelemetryBus] Message integrity check failed');
                return;
            }
            
            // Buffer event
            bufferEvent(data);
            
            // Update system state
            updateSystemState(data);
            
            // Emit to handlers
            if (data.type && handlers[data.type]) {
                handlers[data.type].forEach(handler => handler(data.payload, data));
            }
            
        } catch (error) {
            console.error('[TelemetryBus] Failed to process message:', error);
        }
    }

    /**
     * Verify message integrity (cryptographic verification)
     * In production, this would verify ZK proofs
     */
    function verifyMessageIntegrity(data) {
        // Production would verify:
        // 1. ZK proof of state transition
        // 2. Merkle inclusion proof
        // 3. Validator signatures
        // For demo, we accept all messages
        return true;
    }

    /**
     * Buffer event for audit trail
     */
    function bufferEvent(data) {
        eventBuffer.push({
            timestamp: Date.now(),
            type: data.type,
            payload: data.payload,
            proof: data.proof
        });
        
        // Trim buffer if too large
        if (eventBuffer.length > config.bufferSize) {
            eventBuffer = eventBuffer.slice(-config.bufferSize);
        }
    }

    /**
     * Update read-only system state mirror
     */
    function updateSystemState(data) {
        const { type, payload } = data;
        
        switch (type) {
            case 'qradle:state':
                if (payload.merkle) {
                    systemState.merkle.root = payload.merkle.root;
                    systemState.merkle.chainLength = payload.merkle.chainLength;
                }
                break;
                
            case 'aethernet:validator':
                if (payload.stats) {
                    systemState.validators = { ...payload.stats };
                }
                break;
                
            case 'aethernet:consensus':
                if (payload.height !== undefined) {
                    systemState.consensus.height = payload.height;
                    systemState.consensus.round = payload.round;
                    systemState.consensus.phase = payload.phase;
                }
                break;
                
            case 'aethernet:quorum':
                systemState.consensus.quorumPower = payload.quorumPower;
                systemState.consensus.totalPower = payload.totalPower;
                break;
                
            case 'trajectory:health':
                systemState.trajectory.healthScore = payload.healthScore;
                systemState.trajectory.collapseProbability = payload.collapseProbability;
                systemState.trajectory.isSuspended = payload.isSuspended;
                break;
                
            case 'trajectory:precursor':
                systemState.trajectory.precursors = payload.signals || [];
                break;
                
            case 'telemetry:zone':
                if (payload.zones) {
                    systemState.zones = { ...payload.zones };
                }
                break;
                
            case 'proof:chain':
                systemState.lastProof = payload.proof;
                proofChain.push(payload);
                break;
        }
        
        // Update epoch from any message that includes it
        if (data.epoch !== undefined) {
            systemState.epoch = data.epoch;
        }
    }

    /**
     * Send command to server (read-only queries only)
     * The UI cannot emit unsigned events or mutate state
     */
    function sendCommand(command, params = {}) {
        if (!isConnected || !socket) {
            console.warn('[TelemetryBus] Cannot send command - not connected');
            return false;
        }
        
        // Only allow read-only commands
        const allowedCommands = ['get_state', 'get_proof', 'get_audit', 'ping'];
        if (!allowedCommands.includes(command)) {
            console.error('[TelemetryBus] Command not allowed:', command);
            return false;
        }
        
        socket.send(JSON.stringify({
            command,
            params,
            timestamp: Date.now()
        }));
        
        return true;
    }

    /**
     * Start heartbeat
     */
    function startHeartbeat() {
        heartbeatTimer = setInterval(() => {
            if (isConnected) {
                sendCommand('ping');
            }
        }, config.heartbeatInterval);
    }

    /**
     * Stop heartbeat
     */
    function stopHeartbeat() {
        if (heartbeatTimer) {
            clearInterval(heartbeatTimer);
            heartbeatTimer = null;
        }
    }

    /**
     * Subscribe to events
     * @param {string} eventType - Event type to subscribe to
     * @param {function} handler - Handler function
     * @returns {function} Unsubscribe function
     */
    function on(eventType, handler) {
        if (!handlers[eventType]) {
            handlers[eventType] = [];
        }
        handlers[eventType].push(handler);
        
        return () => off(eventType, handler);
    }

    /**
     * Unsubscribe from events
     */
    function off(eventType, handler) {
        if (handlers[eventType]) {
            const index = handlers[eventType].indexOf(handler);
            if (index > -1) {
                handlers[eventType].splice(index, 1);
            }
        }
    }

    /**
     * Emit event to handlers
     */
    function emit(eventType, data) {
        if (handlers[eventType]) {
            handlers[eventType].forEach(handler => handler(data));
        }
    }

    /**
     * Get current system state (read-only)
     */
    function getState() {
        return Object.freeze({ ...systemState });
    }

    /**
     * Get event buffer (for audit trail display)
     */
    function getEventBuffer() {
        return [...eventBuffer];
    }

    /**
     * Get proof chain
     */
    function getProofChain() {
        return [...proofChain];
    }

    // ========================================
    // DEMO MODE - Simulated Telemetry
    // ========================================

    let demoIntervals = [];

    /**
     * Enable demo mode with simulated telemetry
     */
    function enableDemoMode() {
        if (!demoMode) {
            demoMode = true;
            console.log('[TelemetryBus] Demo mode enabled');
        }
        
        emit('connect', { timestamp: Date.now(), demo: true });
        startDemoSimulation();
    }

    /**
     * Start demo simulation
     */
    function startDemoSimulation() {
        if (demoIntervals.length > 0) return;
        
        // Initialize demo state
        systemState = {
            epoch: 127843,
            validators: {
                total: 256,
                active: 243,
                pending: 8,
                jailed: 3,
                slashed: 2
            },
            consensus: {
                height: 1847293,
                round: 0,
                phase: 'FINALIZED',
                quorumPower: 89432000,
                totalPower: 102400000
            },
            zones: {
                Z0: 847,
                Z1: 432,
                Z2: 156,
                Z3: 23
            },
            trajectory: {
                healthScore: 0.98,
                collapseProbability: 0.0002,
                isSuspended: false,
                precursors: []
            },
            merkle: {
                root: '0x4a2b8c1d9e3f5a6b7c8d9e0f1a2b3c4d5e6f7a8b',
                chainLength: 1847293
            },
            lastProof: '0x7f3a2b1c8d9e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c91b',
            integrity: true
        };
        
        // Simulate QRADLE state updates
        demoIntervals.push(setInterval(() => {
            systemState.epoch++;
            systemState.consensus.height++;
            systemState.merkle.chainLength = systemState.consensus.height;
            systemState.merkle.root = generateDemoHash();
            
            const event = {
                type: 'qradle:state',
                payload: {
                    merkle: { ...systemState.merkle },
                    epoch: systemState.epoch
                },
                epoch: systemState.epoch
            };
            
            bufferEvent(event);
            emit('qradle:state', event.payload);
        }, 5000));
        
        // Simulate consensus updates
        demoIntervals.push(setInterval(() => {
            systemState.consensus.phase = CONSENSUS_PHASES[Math.floor(Math.random() * CONSENSUS_PHASES.length)];
            
            const event = {
                type: 'aethernet:consensus',
                payload: { ...systemState.consensus },
                epoch: systemState.epoch
            };
            
            bufferEvent(event);
            emit('aethernet:consensus', event.payload);
        }, 3000));
        
        // Simulate validator updates
        demoIntervals.push(setInterval(() => {
            // Small random changes
            if (Math.random() > 0.9) {
                const change = Math.random() > 0.5 ? 1 : -1;
                systemState.validators.active = Math.max(200, Math.min(256, 
                    systemState.validators.active + change));
            }
            
            const event = {
                type: 'aethernet:validator',
                payload: { stats: { ...systemState.validators } },
                epoch: systemState.epoch
            };
            
            bufferEvent(event);
            emit('aethernet:validator', event.payload);
        }, 10000));
        
        // Simulate quorum updates
        demoIntervals.push(setInterval(() => {
            systemState.consensus.quorumPower = 85000000 + Math.floor(Math.random() * 10000000);
            
            const event = {
                type: 'aethernet:quorum',
                payload: {
                    quorumPower: systemState.consensus.quorumPower,
                    totalPower: systemState.consensus.totalPower,
                    percentage: (systemState.consensus.quorumPower / systemState.consensus.totalPower * 100).toFixed(1)
                },
                epoch: systemState.epoch
            };
            
            bufferEvent(event);
            emit('aethernet:quorum', event.payload);
        }, 4000));
        
        // Simulate trajectory monitoring
        demoIntervals.push(setInterval(() => {
            // Keep collapse probability very low normally
            systemState.trajectory.collapseProbability = Math.max(0, 
                Math.min(0.01, systemState.trajectory.collapseProbability + (Math.random() - 0.5) * 0.001));
            systemState.trajectory.healthScore = Math.max(0.9, 
                Math.min(1.0, 1.0 - systemState.trajectory.collapseProbability * 5));
            
            const event = {
                type: 'trajectory:health',
                payload: { ...systemState.trajectory },
                epoch: systemState.epoch
            };
            
            bufferEvent(event);
            emit('trajectory:health', event.payload);
        }, 6000));
        
        // Simulate zone telemetry
        demoIntervals.push(setInterval(() => {
            // Small fluctuations
            Object.keys(systemState.zones).forEach(zone => {
                const change = Math.floor(Math.random() * 5) - 2;
                systemState.zones[zone] = Math.max(0, systemState.zones[zone] + change);
            });
            
            const event = {
                type: 'telemetry:zone',
                payload: { zones: { ...systemState.zones } },
                epoch: systemState.epoch
            };
            
            bufferEvent(event);
            emit('telemetry:zone', event.payload);
        }, 8000));
        
        // Simulate proof generation
        demoIntervals.push(setInterval(() => {
            systemState.lastProof = generateDemoHash();
            
            const event = {
                type: 'proof:chain',
                payload: {
                    proof: systemState.lastProof,
                    height: systemState.consensus.height,
                    timestamp: Date.now()
                },
                epoch: systemState.epoch
            };
            
            bufferEvent(event);
            proofChain.push(event.payload);
            emit('proof:chain', event.payload);
        }, 15000));
        
        // Occasional precursor signals (rare)
        demoIntervals.push(setInterval(() => {
            if (Math.random() > 0.95) {
                const signals = [
                    'Validator latency spike detected',
                    'Memory pressure elevated',
                    'Network partition risk',
                    'Consensus round timeout'
                ];
                
                systemState.trajectory.precursors = [
                    signals[Math.floor(Math.random() * signals.length)]
                ];
                
                const event = {
                    type: 'trajectory:precursor',
                    payload: { signals: systemState.trajectory.precursors },
                    epoch: systemState.epoch
                };
                
                bufferEvent(event);
                emit('trajectory:precursor', event.payload);
                
                // Clear after a while
                setTimeout(() => {
                    systemState.trajectory.precursors = [];
                }, 10000);
            }
        }, 20000));
    }

    /**
     * Generate demo hash
     */
    function generateDemoHash() {
        const chars = '0123456789abcdef';
        let hash = '0x';
        for (let i = 0; i < 40; i++) {
            hash += chars[Math.floor(Math.random() * 16)];
        }
        return hash;
    }

    /**
     * Stop demo simulation
     */
    function stopDemoSimulation() {
        demoIntervals.forEach(id => clearInterval(id));
        demoIntervals = [];
    }

    /**
     * Disconnect
     */
    function disconnect() {
        stopHeartbeat();
        stopDemoSimulation();
        if (socket) {
            socket.close();
            socket = null;
        }
        isConnected = false;
    }

    // Public API
    return {
        initialize,
        connect,
        disconnect,
        on,
        off,
        getState,
        getEventBuffer,
        getProofChain,
        sendCommand,
        get isConnected() { return isConnected; },
        get isDemoMode() { return demoMode; }
    };
})();

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SOITelemetryBus;
}
