/**
 * QRATUM Dashboard - API Integration Layer
 * Handles all communication with the backend API
 * @version 2.0.0
 */

const QratumAPI = (function() {
    'use strict';

    // API Configuration
    const config = {
        baseUrl: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
            ? 'http://localhost:8000'
            : '',
        timeout: 30000,
        retryAttempts: 3,
        retryDelay: 1000
    };

    // Request headers
    const defaultHeaders = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    };

    /**
     * Make an API request with retry logic
     * @param {string} endpoint - API endpoint
     * @param {Object} options - Fetch options
     * @returns {Promise<Object>} Response data
     */
    async function request(endpoint, options = {}) {
        const url = `${config.baseUrl}${endpoint}`;
        const fetchOptions = {
            ...options,
            headers: {
                ...defaultHeaders,
                ...options.headers
            }
        };

        let lastError;
        for (let attempt = 0; attempt < config.retryAttempts; attempt++) {
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), config.timeout);
                
                const response = await fetch(url, {
                    ...fetchOptions,
                    signal: controller.signal
                });
                
                clearTimeout(timeoutId);

                if (!response.ok) {
                    throw new APIError(
                        `API request failed: ${response.status} ${response.statusText}`,
                        response.status,
                        await response.text()
                    );
                }

                const data = await response.json();
                return data;
            } catch (error) {
                lastError = error;
                if (error.name === 'AbortError') {
                    throw new APIError('Request timeout', 408);
                }
                if (attempt < config.retryAttempts - 1) {
                    await delay(config.retryDelay * (attempt + 1));
                }
            }
        }
        throw lastError;
    }

    /**
     * Custom API Error class
     */
    class APIError extends Error {
        constructor(message, status, body = null) {
            super(message);
            this.name = 'APIError';
            this.status = status;
            this.body = body;
        }
    }

    /**
     * Delay helper
     * @param {number} ms - Milliseconds to delay
     */
    function delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // ========================================
    // Health & System Status
    // ========================================

    /**
     * Check API health
     * @returns {Promise<Object>} Health status
     */
    async function checkHealth() {
        return request('/health');
    }

    /**
     * Get cluster status
     * @returns {Promise<Object>} Cluster information
     */
    async function getClusterStatus() {
        return request('/api/v1/cluster/status');
    }

    // ========================================
    // Simulation Jobs
    // ========================================

    /**
     * Submit a new simulation job
     * @param {Object} jobData - Job configuration
     * @returns {Promise<Object>} Job submission response
     */
    async function submitJob(jobData) {
        return request('/api/v1/qc/simulate', {
            method: 'POST',
            body: JSON.stringify(jobData)
        });
    }

    /**
     * Get job status
     * @param {string} jobId - Job identifier
     * @returns {Promise<Object>} Job status and results
     */
    async function getJobStatus(jobId) {
        return request(`/api/v1/qc/jobs/${encodeURIComponent(jobId)}`);
    }

    /**
     * List all jobs
     * @param {Object} filters - Optional filters
     * @returns {Promise<Array>} List of jobs
     */
    async function listJobs(filters = {}) {
        const params = new URLSearchParams(filters);
        return request(`/api/v1/jobs?${params}`);
    }

    /**
     * Cancel a job
     * @param {string} jobId - Job identifier
     * @returns {Promise<Object>} Cancellation result
     */
    async function cancelJob(jobId) {
        return request(`/api/v1/jobs/${encodeURIComponent(jobId)}/cancel`, {
            method: 'POST'
        });
    }

    // ========================================
    // Digital Twin
    // ========================================

    /**
     * Create a digital twin
     * @param {Object} twinData - Twin configuration
     * @returns {Promise<Object>} Digital twin info
     */
    async function createDigitalTwin(twinData) {
        return request('/api/v1/dtwin/create', {
            method: 'POST',
            body: JSON.stringify(twinData)
        });
    }

    /**
     * Simulate digital twin
     * @param {string} twinId - Twin identifier
     * @param {number} timeSteps - Number of simulation steps
     * @returns {Promise<Object>} Simulation results
     */
    async function simulateDigitalTwin(twinId, timeSteps = 100) {
        return request(`/api/v1/dtwin/${encodeURIComponent(twinId)}/simulate`, {
            method: 'POST',
            body: JSON.stringify({ time_steps: timeSteps })
        });
    }

    // ========================================
    // Optimization
    // ========================================

    /**
     * Run optimization
     * @param {Object} optConfig - Optimization configuration
     * @returns {Promise<Object>} Optimization job info
     */
    async function runOptimization(optConfig) {
        return request('/api/v1/opt/optimize', {
            method: 'POST',
            body: JSON.stringify(optConfig)
        });
    }

    // ========================================
    // Metrics & Telemetry
    // ========================================

    /**
     * Get GPU metrics
     * @returns {Promise<Object>} GPU utilization data
     */
    async function getGPUMetrics() {
        return request('/api/v1/metrics/gpu');
    }

    /**
     * Get cost tracking data
     * @param {string} period - Time period (7d, 30d, 90d)
     * @returns {Promise<Object>} Cost data
     */
    async function getCostData(period = '30d') {
        return request(`/api/v1/metrics/cost?period=${encodeURIComponent(period)}`);
    }

    /**
     * Get usage history
     * @param {string} period - Time period
     * @returns {Promise<Array>} Usage history data
     */
    async function getUsageHistory(period = '30d') {
        return request(`/api/v1/metrics/usage?period=${encodeURIComponent(period)}`);
    }

    // ========================================
    // Results & Export
    // ========================================

    /**
     * Get job results
     * @param {string} jobId - Job identifier
     * @returns {Promise<Object>} Job results
     */
    async function getJobResults(jobId) {
        return request(`/api/v1/jobs/${encodeURIComponent(jobId)}/results`);
    }

    /**
     * Export results
     * @param {string} jobId - Job identifier
     * @param {string} format - Export format (pdf, json, hdf5, csv)
     * @returns {Promise<Blob>} Exported data
     */
    async function exportResults(jobId, format) {
        const url = `${config.baseUrl}/api/v1/jobs/${encodeURIComponent(jobId)}/export?format=${encodeURIComponent(format)}`;
        const response = await fetch(url, {
            headers: defaultHeaders
        });
        
        if (!response.ok) {
            throw new APIError(`Export failed: ${response.status}`, response.status);
        }
        
        return response.blob();
    }

    // ========================================
    // File Upload
    // ========================================

    /**
     * Upload input file
     * @param {File} file - File to upload
     * @param {function} onProgress - Progress callback
     * @returns {Promise<Object>} Upload result
     */
    async function uploadFile(file, onProgress) {
        const formData = new FormData();
        formData.append('file', file);

        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            xhr.open('POST', `${config.baseUrl}/api/v1/files/upload`);

            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable && onProgress) {
                    onProgress((e.loaded / e.total) * 100);
                }
            });

            xhr.addEventListener('load', () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    resolve(JSON.parse(xhr.responseText));
                } else {
                    reject(new APIError('Upload failed', xhr.status, xhr.responseText));
                }
            });

            xhr.addEventListener('error', () => {
                reject(new APIError('Upload error', 0));
            });

            xhr.send(formData);
        });
    }

    // ========================================
    // Mock Data Generators (for demo mode)
    // ========================================

    /**
     * Generate mock jobs for demo
     */
    function generateMockJobs() {
        const statuses = ['running', 'queued', 'completed', 'failed'];
        const types = ['quantum', 'tensor', 'materials', 'elastomeric', 'digital-twin', 'optimization'];
        const priorities = ['low', 'normal', 'high', 'critical'];
        
        return Array.from({ length: 12 }, (_, i) => ({
            id: `job_${Date.now()}_${i}`,
            name: [
                'Tire Compound Analysis',
                'Quantum Circuit Optimization',
                'Materials Stress Test',
                'Elastomer Deformation Sim',
                'Digital Twin Update',
                'QAOA Optimization Run'
            ][i % 6],
            status: statuses[i % 4],
            type: types[i % 6],
            priority: priorities[i % 4],
            progress: statuses[i % 4] === 'running' ? Math.floor(Math.random() * 90) + 10 : 
                      statuses[i % 4] === 'completed' ? 100 : 0,
            createdAt: new Date(Date.now() - Math.random() * 86400000).toISOString(),
            eta: statuses[i % 4] === 'running' ? 
                 `~${Math.floor(Math.random() * 60) + 5} min` : '-'
        }));
    }

    /**
     * Generate mock GPU metrics
     */
    function generateMockGPUMetrics() {
        return {
            utilization: Math.floor(Math.random() * 40) + 50,
            vramUsed: Math.floor(Math.random() * 30) + 40,
            vramTotal: 80,
            temperature: Math.floor(Math.random() * 20) + 45,
            powerDraw: Math.floor(Math.random() * 100) + 200
        };
    }

    /**
     * Generate mock convergence data
     */
    function generateMockConvergenceData() {
        const data = [];
        let value = 1.0;
        for (let i = 0; i < 50; i++) {
            value *= (0.9 + Math.random() * 0.05);
            data.push({
                iteration: i,
                loss: Math.max(0.001, value + Math.random() * 0.05)
            });
        }
        return data;
    }

    /**
     * Generate mock cluster status
     */
    function generateMockClusterStatus() {
        return {
            eks: {
                status: 'online',
                nodes: 8,
                gpus: 32,
                utilization: Math.floor(Math.random() * 30) + 50
            },
            gke: {
                status: 'online',
                nodes: 6,
                gpus: 24,
                utilization: Math.floor(Math.random() * 30) + 40
            },
            aks: {
                status: Math.random() > 0.7 ? 'standby' : 'online',
                nodes: 4,
                gpus: 16,
                utilization: Math.floor(Math.random() * 20) + 10
            }
        };
    }

    /**
     * Generate mock cost data
     */
    function generateMockCostData() {
        return {
            total: 12450,
            budget: 20000,
            breakdown: {
                eks: 6200,
                gke: 4100,
                aks: 2150
            },
            history: Array.from({ length: 30 }, (_, i) => ({
                date: new Date(Date.now() - (29 - i) * 86400000).toISOString().split('T')[0],
                cost: Math.floor(Math.random() * 200) + 300
            }))
        };
    }

    // Public API
    return {
        // Configuration
        config,
        
        // Health & Status
        checkHealth,
        getClusterStatus,
        
        // Jobs
        submitJob,
        getJobStatus,
        listJobs,
        cancelJob,
        
        // Digital Twin
        createDigitalTwin,
        simulateDigitalTwin,
        
        // Optimization
        runOptimization,
        
        // Metrics
        getGPUMetrics,
        getCostData,
        getUsageHistory,
        
        // Results
        getJobResults,
        exportResults,
        
        // File Upload
        uploadFile,
        
        // Mock Data (for demo mode)
        mock: {
            generateJobs: generateMockJobs,
            generateGPUMetrics: generateMockGPUMetrics,
            generateConvergenceData: generateMockConvergenceData,
            generateClusterStatus: generateMockClusterStatus,
            generateCostData: generateMockCostData
        },
        
        // Error class
        APIError
    };
})();

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = QratumAPI;
}
