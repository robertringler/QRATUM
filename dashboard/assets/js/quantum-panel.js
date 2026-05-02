/**
 * QRATUM Dashboard - Quantum Simulation Panel (Phase 4)
 * Interactive quantum gate visualization and WASM pod isolation
 * @version 4.0.0
 */

const QratumQuantum = (function() {
    'use strict';

    // State
    let quantumState = {
        numQubits: 12,
        states: [],
        entropy: 0,
        gateCount: 0
    };
    
    // Color maps for visualization
    const colorMaps = {
        amplitude: {
            high: '#00f5ff',
            medium: '#7b2cbf',
            low: '#1a1a2e'
        },
        phase: {
            0: '#00ff88',
            90: '#00f5ff',
            180: '#ff006e',
            270: '#ffaa00'
        }
    };

    /**
     * Initialize quantum panel
     */
    function init() {
        console.log('[Quantum] Initializing Phase 4 Quantum Panel');
        
        initQubitGrid();
        initGateButtons();
        initResetButton();
        updateVisualization();
    }

    /**
     * Initialize qubit state grid
     */
    function initQubitGrid() {
        const grid = document.getElementById('qubit-grid');
        if (!grid) return;
        
        // Create 12-qubit visualization grid
        let html = '<div class="qubit-row header">';
        html += '<span class="qubit-label">State</span>';
        for (let q = 0; q < quantumState.numQubits; q++) {
            html += `<span class="qubit-header">q${q}</span>`;
        }
        html += '<span class="qubit-amplitude">Amplitude</span>';
        html += '<span class="qubit-phase">Phase</span>';
        html += '</div>';
        
        // Initially show |0⟩ state
        quantumState.states = [{
            index: 0,
            amplitude: 1.0,
            phase: 0,
            probability: 1.0
        }];
        
        html += renderStateRows();
        
        grid.innerHTML = html;
    }

    /**
     * Render state rows for current quantum state
     */
    function renderStateRows() {
        let html = '';
        const maxRows = 16; // Limit display for performance
        
        quantumState.states.slice(0, maxRows).forEach(state => {
            const binary = state.index.toString(2).padStart(quantumState.numQubits, '0');
            const ampColor = getAmplitudeColor(state.amplitude);
            const phaseColor = getPhaseColor(state.phase);
            
            html += '<div class="qubit-row state">';
            html += `<span class="qubit-label">|${binary}⟩</span>`;
            
            // Show each qubit bit
            for (let q = quantumState.numQubits - 1; q >= 0; q--) {
                const bit = (state.index >> q) & 1;
                const bitClass = bit === 1 ? 'bit-one' : 'bit-zero';
                html += `<span class="qubit-bit ${bitClass}">${bit}</span>`;
            }
            
            // Amplitude bar
            html += `<span class="qubit-amplitude">
                <div class="amplitude-bar" style="width: ${state.amplitude * 100}%; background: ${ampColor};">
                    ${state.amplitude.toFixed(3)}
                </div>
            </span>`;
            
            // Phase indicator
            const phaseDeg = (state.phase * 180 / Math.PI).toFixed(1);
            html += `<span class="qubit-phase">
                <div class="phase-indicator" style="transform: rotate(${phaseDeg}deg); background: ${phaseColor};"></div>
                <span class="phase-value">${phaseDeg}°</span>
            </span>`;
            
            html += '</div>';
        });
        
        if (quantumState.states.length > maxRows) {
            html += `<div class="qubit-row more">... and ${quantumState.states.length - maxRows} more states</div>`;
        }
        
        return html;
    }

    /**
     * Initialize gate buttons
     */
    function initGateButtons() {
        document.querySelectorAll('.gate-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const gate = btn.dataset.gate;
                applyGate(gate);
            });
        });
    }

    /**
     * Initialize reset button
     */
    function initResetButton() {
        const resetBtn = document.getElementById('quantum-reset-btn');
        if (resetBtn) {
            resetBtn.addEventListener('click', resetQuantumState);
        }
    }

    /**
     * Apply a quantum gate
     * @param {string} gate - Gate type (H, X, Y, Z, S, T, CNOT, Toffoli)
     */
    async function applyGate(gate) {
        const targetQubit = parseInt(document.getElementById('target-qubit')?.value || '0');
        const controlQubit = parseInt(document.getElementById('control-qubit')?.value || '1');
        const controlQubit2 = parseInt(document.getElementById('control-qubit2')?.value || '2');
        
        let qubits = [targetQubit];
        
        if (gate === 'CNOT') {
            qubits = [controlQubit, targetQubit];
        } else if (gate === 'Toffoli') {
            qubits = [controlQubit, controlQubit2, targetQubit];
        }
        
        console.log(`[Quantum] Applying ${gate} gate to qubits:`, qubits);
        
        // Try to call Tauri backend if available
        if (window.__TAURI__) {
            try {
                const result = await window.__TAURI__.tauri.invoke('apply_quantum_gate', {
                    gateOp: {
                        gate_type: gate,
                        qubits: qubits
                    }
                });
                
                if (result.success) {
                    quantumState.states = result.visualization.states;
                    quantumState.entropy = result.visualization.entropy;
                    quantumState.gateCount++;
                    updateVisualization();
                    showGateNotification(gate, true);
                } else {
                    showGateNotification(gate, false, result.message);
                }
            } catch (error) {
                console.error('[Quantum] Gate application error:', error);
                // Fall back to simulation
                simulateGate(gate, qubits);
            }
        } else {
            // Web demo mode - simulate
            simulateGate(gate, qubits);
        }
    }

    /**
     * Simulate gate application in demo mode
     */
    function simulateGate(gate, qubits) {
        // Validate qubit indices
        const targetQubit = qubits[0];
        if (targetQubit === undefined || targetQubit < 0 || targetQubit >= quantumState.numQubits) {
            showGateNotification(gate, false, 'Invalid qubit index');
            return;
        }
        
        quantumState.gateCount++;
        
        // Simple simulation for demo
        if (quantumState.states.length === 1 && quantumState.states[0].index === 0) {
            // Initial |0⟩ state
            if (gate === 'H') {
                // Hadamard creates superposition
                const q = qubits[0];
                quantumState.states = [
                    { index: 0, amplitude: 0.707, phase: 0, probability: 0.5 },
                    { index: 1 << q, amplitude: 0.707, phase: 0, probability: 0.5 }
                ];
            } else if (gate === 'X') {
                // X flips the qubit
                const q = qubits[0];
                quantumState.states = [
                    { index: 1 << q, amplitude: 1.0, phase: 0, probability: 1.0 }
                ];
            } else if (gate === 'S' || gate === 'T' || gate === 'Z') {
                // Phase gates don't change |0⟩
            } else if (gate === 'Y') {
                // Y|0⟩ = i|1⟩
                const q = qubits[0];
                quantumState.states = [
                    { index: 1 << q, amplitude: 1.0, phase: Math.PI / 2, probability: 1.0 }
                ];
            } else if (gate === 'CNOT' || gate === 'Toffoli') {
                // Multi-qubit gates on |0⟩ state have no effect
                console.log(`[Quantum] ${gate} gate on |0⟩ state - no effect`);
            } else {
                // Unknown gate in demo mode
                console.log(`[Quantum] Gate '${gate}' not fully simulated in demo mode`);
            }
        } else if (gate === 'CNOT' && quantumState.states.length === 2) {
            // If in superposition and applying CNOT, create entanglement
            // This is a simplified demo - Bell state creation
            if (quantumState.states[0].index === 0 && quantumState.states[1].index === 1) {
                quantumState.states = [
                    { index: 0, amplitude: 0.707, phase: 0, probability: 0.5 },
                    { index: 3, amplitude: 0.707, phase: 0, probability: 0.5 }
                ];
            }
        } else {
            // Log unsimulated gate combinations in demo mode
            console.log(`[Quantum] Gate '${gate}' simulation limited in demo mode for current state`);
        }
        
        // Calculate entropy (using log base 2 for quantum information theory)
        let entropy = 0;
        quantumState.states.forEach(s => {
            const p = s.probability;
            if (p > 0.001) {
                entropy -= p * Math.log2(p);
            }
        });
        quantumState.entropy = entropy;
        
        updateVisualization();
        showGateNotification(gate, true);
    }

    /**
     * Reset quantum state to |0⟩
     */
    async function resetQuantumState() {
        quantumState.states = [{
            index: 0,
            amplitude: 1.0,
            phase: 0,
            probability: 1.0
        }];
        quantumState.entropy = 0;
        quantumState.gateCount = 0;
        
        updateVisualization();
        showNotification('Quantum state reset to |0⟩', 'info');
    }

    /**
     * Update visualization
     */
    function updateVisualization() {
        const grid = document.getElementById('qubit-grid');
        if (grid) {
            let html = '<div class="qubit-row header">';
            html += '<span class="qubit-label">State</span>';
            for (let q = 0; q < quantumState.numQubits; q++) {
                html += `<span class="qubit-header">q${q}</span>`;
            }
            html += '<span class="qubit-amplitude">Amplitude</span>';
            html += '<span class="qubit-phase">Phase</span>';
            html += '</div>';
            html += renderStateRows();
            grid.innerHTML = html;
        }
        
        // Update metrics
        const entropyEl = document.getElementById('quantum-entropy');
        if (entropyEl) {
            entropyEl.textContent = quantumState.entropy.toFixed(3);
        }
        
        const statesEl = document.getElementById('nonzero-states');
        if (statesEl) {
            statesEl.textContent = quantumState.states.length;
        }
        
        const gateCountEl = document.getElementById('gate-count');
        if (gateCountEl) {
            gateCountEl.textContent = quantumState.gateCount;
        }
    }

    /**
     * Get color for amplitude value
     */
    function getAmplitudeColor(amplitude) {
        if (amplitude > 0.7) return colorMaps.amplitude.high;
        if (amplitude > 0.3) return colorMaps.amplitude.medium;
        return colorMaps.amplitude.low;
    }

    /**
     * Get color for phase value
     */
    function getPhaseColor(phase) {
        const degrees = (phase * 180 / Math.PI + 360) % 360;
        if (degrees < 45 || degrees >= 315) return colorMaps.phase[0];
        if (degrees < 135) return colorMaps.phase[90];
        if (degrees < 225) return colorMaps.phase[180];
        return colorMaps.phase[270];
    }

    /**
     * Show gate application notification
     */
    function showGateNotification(gate, success, message) {
        if (window.showToast) {
            const msg = success 
                ? `Applied ${gate} gate successfully`
                : `Failed to apply ${gate} gate: ${message || 'Unknown error'}`;
            window.showToast(msg, success ? 'success' : 'error');
        }
    }

    /**
     * Show notification
     */
    function showNotification(message, type) {
        if (window.showToast) {
            window.showToast(message, type);
        }
    }

    // Public API
    return {
        init,
        applyGate,
        resetQuantumState,
        getState: () => quantumState
    };
})();

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Delay initialization to ensure main dashboard is ready
    setTimeout(() => {
        QratumQuantum.init();
    }, 100);
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = QratumQuantum;
}
