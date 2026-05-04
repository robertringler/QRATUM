/**
 * QRATUM SOI API
 * Read-only API for Sovereign Operations Interface
 * 
 * This API only allows read operations - the UI cannot mutate state.
 * All state changes must go through QRADLE contracts.
 * 
 * @version 1.0.0
 */

const SOIAPI = (function() {
    'use strict';
    
    // Configuration
    const config = {
        baseUrl: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
            ? 'http://localhost:8000'
            : '',
        timeout: 30000
    };
    
    // Default headers
    const headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    };
    
    /**
     * Make API request
     */
    async function request(endpoint, options = {}) {
        const url = `${config.baseUrl}${endpoint}`;
        
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), config.timeout);
            
            const response = await fetch(url, {
                ...options,
                headers: { ...headers, ...options.headers },
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('[SOIAPI] Request failed:', error);
            throw error;
        }
    }
    
    // ========================================
    // READ-ONLY API METHODS
    // ========================================
    
    /**
     * Get QRADLE system state
     */
    async function getQRADLEState() {
        return request('/api/v1/soi/qradle/state');
    }
    
    /**
     * Get QRADLE audit trail
     */
    async function getAuditTrail(contractId = null, limit = 100) {
        const params = new URLSearchParams({ limit });
        if (contractId) params.append('contract_id', contractId);
        return request(`/api/v1/soi/qradle/audit?${params}`);
    }
    
    /**
     * Get QRADLE checkpoints
     */
    async function getCheckpoints(limit = 50) {
        return request(`/api/v1/soi/qradle/checkpoints?limit=${limit}`);
    }
    
    /**
     * Get system proof
     */
    async function getSystemProof() {
        return request('/api/v1/soi/qradle/proof');
    }
    
    /**
     * Get Aethernet validator registry
     */
    async function getValidatorRegistry() {
        return request('/api/v1/soi/aethernet/validators');
    }
    
    /**
     * Get specific validator info
     */
    async function getValidator(validatorId) {
        return request(`/api/v1/soi/aethernet/validators/${encodeURIComponent(validatorId)}`);
    }
    
    /**
     * Get consensus state
     */
    async function getConsensusState() {
        return request('/api/v1/soi/aethernet/consensus');
    }
    
    /**
     * Get slashing history
     */
    async function getSlashingHistory(limit = 50) {
        return request(`/api/v1/soi/aethernet/slashing?limit=${limit}`);
    }
    
    /**
     * Get trajectory health status
     */
    async function getTrajectoryHealth() {
        return request('/api/v1/soi/trajectory/health');
    }
    
    /**
     * Get collapse precursor signals
     */
    async function getPrecursorSignals() {
        return request('/api/v1/soi/trajectory/precursors');
    }
    
    /**
     * Get zone statistics
     */
    async function getZoneStats() {
        return request('/api/v1/soi/zones/stats');
    }
    
    /**
     * Get nodes by zone
     */
    async function getNodesByZone(zone) {
        return request(`/api/v1/soi/zones/${encodeURIComponent(zone)}/nodes`);
    }
    
    /**
     * Get vertical module status
     */
    async function getVerticalStatus(vertical = null) {
        const endpoint = vertical 
            ? `/api/v1/soi/verticals/${encodeURIComponent(vertical)}`
            : '/api/v1/soi/verticals';
        return request(endpoint);
    }
    
    /**
     * Get finalized blocks
     */
    async function getFinalizedBlocks(limit = 10) {
        return request(`/api/v1/soi/blocks?limit=${limit}`);
    }
    
    /**
     * Get block by height
     */
    async function getBlock(height) {
        return request(`/api/v1/soi/blocks/${height}`);
    }
    
    // ========================================
    // MOCK DATA FOR DEMO MODE
    // ========================================
    
    const mock = {
        qradleState: () => ({
            merkle_root: '0x4a2b8c1d9e3f5a6b7c8d9e0f1a2b3c4d5e6f7a8b',
            chain_length: 1847293,
            contracts_executed: 45231,
            integrity_verified: true,
            last_checkpoint: {
                id: 'cp_abc123',
                timestamp: new Date().toISOString(),
                description: 'Hourly checkpoint'
            }
        }),
        
        validators: () => ({
            total: 256,
            active: 243,
            pending: 8,
            jailed: 3,
            slashed: 2,
            total_stake: 102400000,
            validators: Array.from({ length: 10 }, (_, i) => ({
                id: `val_${i.toString().padStart(4, '0')}`,
                status: i < 8 ? 'active' : i < 9 ? 'pending' : 'jailed',
                stake: 400000 + Math.floor(Math.random() * 100000),
                uptime: 0.95 + Math.random() * 0.05,
                blocks_proposed: Math.floor(Math.random() * 1000)
            }))
        }),
        
        consensus: () => ({
            height: 1847293,
            round: 0,
            phase: 'FINALIZED',
            proposer: 'val_0127',
            quorum_power: 89432000,
            total_power: 102400000,
            quorum_percentage: 87.3
        }),
        
        trajectory: () => ({
            health_score: 0.98,
            collapse_probability: 0.0002,
            is_suspended: false,
            precursor_signals: [],
            last_updated: new Date().toISOString()
        }),
        
        zones: () => ({
            Z0: { count: 847, label: 'Public' },
            Z1: { count: 432, label: 'Private' },
            Z2: { count: 156, label: 'Regulated' },
            Z3: { count: 23, label: 'Air-gapped' }
        }),
        
        verticals: () => ([
            { id: 'VITRA', name: 'VITRA-E0', status: 'active', operations: 1234 },
            { id: 'CAPRA', name: 'CAPRA', status: 'active', operations: 5678 },
            { id: 'JURIS', name: 'JURIS', status: 'active', operations: 3456 },
            { id: 'ECORA', name: 'ECORA', status: 'active', operations: 2345 },
            { id: 'FLUXA', name: 'FLUXA', status: 'active', operations: 4567 }
        ])
    };
    
    // Public API (read-only)
    return {
        config,
        
        // QRADLE queries
        getQRADLEState,
        getAuditTrail,
        getCheckpoints,
        getSystemProof,
        
        // Aethernet queries
        getValidatorRegistry,
        getValidator,
        getConsensusState,
        getSlashingHistory,
        
        // Trajectory queries
        getTrajectoryHealth,
        getPrecursorSignals,
        
        // Zone queries
        getZoneStats,
        getNodesByZone,
        
        // Vertical queries
        getVerticalStatus,
        
        // Block queries
        getFinalizedBlocks,
        getBlock,
        
        // Mock data for demo
        mock
    };
})();

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SOIAPI;
}
