/**
 * QRATUM Dashboard - WebSocket Handler
 * Real-time communication for live updates
 * @version 2.0.0
 */

const QratumWebSocket = (function() {
    'use strict';

    // WebSocket Configuration
    const config = {
        url: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
            ? 'ws://localhost:8000/ws'
            : `wss://${window.location.host}/ws`,
        reconnectInterval: 3000,
        maxReconnectAttempts: 10,
        heartbeatInterval: 30000
    };

    // State
    let socket = null;
    let reconnectAttempts = 0;
    let heartbeatTimer = null;
    let isConnected = false;
    let messageQueue = [];
    let demoMode = true; // Enable demo mode when no server available

    // Event handlers registry
    const handlers = {
        open: [],
        close: [],
        error: [],
        message: [],
        // Custom event types
        'job:update': [],
        'job:complete': [],
        'job:failed': [],
        'metrics:gpu': [],
        'metrics:system': [],
        'cluster:status': [],
        'alert:new': []
    };

    /**
     * Initialize WebSocket connection
     */
    function connect() {
        if (socket && socket.readyState === WebSocket.OPEN) {
            return;
        }

        try {
            socket = new WebSocket(config.url);

            socket.onopen = handleOpen;
            socket.onclose = handleClose;
            socket.onerror = handleError;
            socket.onmessage = handleMessage;
        } catch (error) {
            console.warn('[WS] WebSocket connection failed, enabling demo mode:', error);
            enableDemoMode();
        }
    }

    /**
     * Handle connection open
     */
    function handleOpen(event) {
        console.log('[WS] Connected to server');
        isConnected = true;
        reconnectAttempts = 0;
        demoMode = false;

        // Start heartbeat
        startHeartbeat();

        // Send queued messages
        flushMessageQueue();

        // Notify handlers
        handlers.open.forEach(handler => handler(event));

        // Update UI connection status
        updateConnectionStatus(true);
    }

    /**
     * Handle connection close
     */
    function handleClose(event) {
        console.log('[WS] Connection closed:', event.code, event.reason);
        isConnected = false;

        // Stop heartbeat
        stopHeartbeat();

        // Notify handlers
        handlers.close.forEach(handler => handler(event));

        // Update UI connection status
        updateConnectionStatus(false);

        // Attempt reconnection
        if (reconnectAttempts < config.maxReconnectAttempts) {
            reconnectAttempts++;
            console.log(`[WS] Reconnecting... Attempt ${reconnectAttempts}/${config.maxReconnectAttempts}`);
            setTimeout(connect, config.reconnectInterval);
        } else {
            console.log('[WS] Max reconnection attempts reached, enabling demo mode');
            enableDemoMode();
        }
    }

    /**
     * Handle connection error
     */
    function handleError(error) {
        console.error('[WS] Error:', error);
        handlers.error.forEach(handler => handler(error));
        
        if (!isConnected) {
            enableDemoMode();
        }
    }

    /**
     * Handle incoming message
     */
    function handleMessage(event) {
        try {
            const data = JSON.parse(event.data);
            
            // Handle heartbeat response
            if (data.type === 'pong') {
                return;
            }

            // Notify general message handlers
            handlers.message.forEach(handler => handler(data));

            // Notify type-specific handlers
            if (data.type && handlers[data.type]) {
                handlers[data.type].forEach(handler => handler(data.payload));
            }
        } catch (error) {
            console.error('[WS] Failed to parse message:', error);
        }
    }

    /**
     * Send message to server
     * @param {string} type - Message type
     * @param {Object} payload - Message payload
     */
    function send(type, payload = {}) {
        const message = JSON.stringify({ type, payload, timestamp: Date.now() });

        if (isConnected && socket.readyState === WebSocket.OPEN) {
            socket.send(message);
        } else {
            messageQueue.push(message);
        }
    }

    /**
     * Flush queued messages
     */
    function flushMessageQueue() {
        while (messageQueue.length > 0 && isConnected) {
            socket.send(messageQueue.shift());
        }
    }

    /**
     * Start heartbeat
     */
    function startHeartbeat() {
        heartbeatTimer = setInterval(() => {
            if (isConnected) {
                send('ping');
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
     * @param {function} handler - Event handler function
     * @returns {function} Unsubscribe function
     */
    function on(eventType, handler) {
        if (!handlers[eventType]) {
            handlers[eventType] = [];
        }
        handlers[eventType].push(handler);

        // Return unsubscribe function
        return () => {
            const index = handlers[eventType].indexOf(handler);
            if (index > -1) {
                handlers[eventType].splice(index, 1);
            }
        };
    }

    /**
     * Unsubscribe from events
     * @param {string} eventType - Event type
     * @param {function} handler - Handler to remove
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
     * Update UI connection status
     * @param {boolean} connected - Connection state
     */
    function updateConnectionStatus(connected) {
        const statusEl = document.getElementById('connection-status');
        if (statusEl) {
            statusEl.textContent = connected ? '🟢 Connected' : '🔴 Disconnected';
        }
        
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.querySelector('.status-text');
        if (statusDot) {
            statusDot.classList.toggle('online', connected);
            statusDot.classList.toggle('offline', !connected);
        }
        if (statusText) {
            statusText.textContent = connected ? 'System Online' : (demoMode ? 'Demo Mode' : 'Reconnecting...');
            statusText.style.color = connected ? 'var(--success)' : (demoMode ? 'var(--warning)' : 'var(--error)');
        }
    }

    /**
     * Disconnect WebSocket
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

    // ========================================
    // Demo Mode - Simulated Real-Time Updates
    // ========================================

    let demoIntervals = [];

    /**
     * Enable demo mode with simulated data
     */
    function enableDemoMode() {
        demoMode = true;
        console.log('[WS] Demo mode enabled - using simulated data');
        
        updateConnectionStatus(false);
        
        // Update status to show demo mode
        const statusText = document.querySelector('.status-text');
        if (statusText) {
            statusText.textContent = 'Demo Mode';
            statusText.style.color = 'var(--warning)';
        }
        
        const statusDot = document.querySelector('.status-dot');
        if (statusDot) {
            statusDot.classList.remove('online', 'offline');
            statusDot.style.background = 'var(--warning)';
            statusDot.style.boxShadow = '0 0 10px var(--warning)';
        }

        // Start demo data simulation
        startDemoSimulation();
    }

    /**
     * Start demo data simulation
     */
    function startDemoSimulation() {
        if (demoIntervals.length > 0) return;

        // Simulate GPU metrics updates
        demoIntervals.push(setInterval(() => {
            const gpuData = {
                utilization: Math.floor(Math.random() * 40) + 50,
                vramUsed: Math.floor(Math.random() * 30) + 40,
                vramTotal: 80,
                temperature: Math.floor(Math.random() * 20) + 45,
                powerDraw: Math.floor(Math.random() * 100) + 200
            };
            handlers['metrics:gpu'].forEach(handler => handler(gpuData));
        }, 2000));

        // Simulate job progress updates
        demoIntervals.push(setInterval(() => {
            const jobUpdate = {
                jobId: `job_demo_${Math.floor(Math.random() * 5)}`,
                progress: Math.floor(Math.random() * 100),
                status: 'running',
                metrics: {
                    loss: Math.random() * 0.5,
                    accuracy: 0.85 + Math.random() * 0.1
                }
            };
            handlers['job:update'].forEach(handler => handler(jobUpdate));
        }, 3000));

        // Simulate system metrics
        demoIntervals.push(setInterval(() => {
            const systemData = {
                cpuUsage: Math.floor(Math.random() * 30) + 20,
                memoryUsage: Math.floor(Math.random() * 40) + 30,
                networkIn: Math.floor(Math.random() * 100),
                networkOut: Math.floor(Math.random() * 50)
            };
            handlers['metrics:system'].forEach(handler => handler(systemData));
        }, 5000));

        // Simulate occasional job completion
        demoIntervals.push(setInterval(() => {
            if (Math.random() > 0.7) {
                const completedJob = {
                    jobId: `job_demo_${Math.floor(Math.random() * 5)}`,
                    status: 'completed',
                    results: {
                        finalEnergy: -45.234 + Math.random() * 5,
                        convergence: 0.999,
                        iterations: Math.floor(Math.random() * 500) + 100,
                        runtime: `${Math.floor(Math.random() * 60) + 5} min`
                    }
                };
                handlers['job:complete'].forEach(handler => handler(completedJob));
            }
        }, 15000));

        // Simulate occasional alerts
        demoIntervals.push(setInterval(() => {
            if (Math.random() > 0.8) {
                const alerts = [
                    { type: 'warning', title: 'GPU memory usage high', time: new Date().toISOString() },
                    { type: 'info', title: 'Job batch completed', time: new Date().toISOString() },
                    { type: 'warning', title: 'Budget threshold reached', time: new Date().toISOString() }
                ];
                const alert = alerts[Math.floor(Math.random() * alerts.length)];
                handlers['alert:new'].forEach(handler => handler(alert));
            }
        }, 20000));
    }

    /**
     * Stop demo simulation
     */
    function stopDemoSimulation() {
        demoIntervals.forEach(id => clearInterval(id));
        demoIntervals = [];
    }

    /**
     * Check if in demo mode
     * @returns {boolean}
     */
    function isDemoMode() {
        return demoMode;
    }

    // Public API
    return {
        connect,
        disconnect,
        send,
        on,
        off,
        isDemoMode,
        enableDemoMode,
        get isConnected() {
            return isConnected;
        }
    };
})();

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = QratumWebSocket;
}
