// QRATUM Desktop Edition - Frontend JavaScript

// Tauri API will be available via window.__TAURI__
const { invoke } = window.__TAURI__.core;

// State
let updateInterval = null;

// Utility functions
function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatPercent(value) {
    return value.toFixed(1) + '%';
}

function addActivityLog(message) {
    const log = document.getElementById('activity-log');
    const item = document.createElement('div');
    item.className = 'activity-item';
    
    const timestamp = new Date().toLocaleTimeString();
    item.innerHTML = `
        <div class="timestamp">${timestamp}</div>
        <div class="message">${message}</div>
    `;
    
    log.insertBefore(item, log.firstChild);
    
    // Keep only last 50 items
    while (log.children.length > 50) {
        log.removeChild(log.lastChild);
    }
}

// Update system health
async function updateSystemHealth() {
    try {
        // Get CPU usage
        const cpuUsage = await invoke('get_cpu_usage');
        document.getElementById('cpu-usage').textContent = formatPercent(cpuUsage);
        document.getElementById('cpu-bar').style.width = cpuUsage + '%';
        
        // Get memory usage
        const [memoryUsed, memoryTotal] = await invoke('get_memory_usage');
        const memoryPercent = (memoryUsed / memoryTotal) * 100;
        document.getElementById('memory-usage').textContent = formatPercent(memoryPercent);
        document.getElementById('memory-detail').textContent = `${formatBytes(memoryUsed)} / ${formatBytes(memoryTotal)}`;
        document.getElementById('memory-bar').style.width = memoryPercent + '%';
        
        // Get disk usage
        const [diskUsed, diskTotal] = await invoke('get_disk_usage');
        const diskPercent = (diskUsed / diskTotal) * 100;
        document.getElementById('disk-usage').textContent = formatPercent(diskPercent);
        document.getElementById('disk-detail').textContent = `${formatBytes(diskUsed)} / ${formatBytes(diskTotal)}`;
        document.getElementById('disk-bar').style.width = diskPercent + '%';
        
    } catch (error) {
        console.error('Failed to update system health:', error);
        addActivityLog(`Error: ${error}`);
    }
}

// Update kernel status
async function updateKernelStatus() {
    try {
        const status = await invoke('get_kernel_status');
        document.getElementById('kernel-status').textContent = status.initialized ? 'Initialized' : 'Not Initialized';
        document.getElementById('kernel-version').textContent = status.version;
    } catch (error) {
        console.error('Failed to get kernel status:', error);
    }
}

// Execute test computation
async function executeComputation() {
    const btn = document.getElementById('execute-btn');
    const resultDiv = document.getElementById('computation-result');
    const output = document.getElementById('computation-output');
    
    btn.disabled = true;
    btn.textContent = 'Executing...';
    
    try {
        addActivityLog('Starting test computation...');
        
        const result = await invoke('execute_computation', {
            input: {
                type: 'test',
                params: { test: true }
            }
        });
        
        output.textContent = JSON.stringify(result, null, 2);
        resultDiv.classList.remove('hidden');
        
        addActivityLog('Computation completed successfully');
        
    } catch (error) {
        output.textContent = `Error: ${error}`;
        resultDiv.classList.remove('hidden');
        addActivityLog(`Computation failed: ${error}`);
        
    } finally {
        btn.disabled = false;
        btn.textContent = 'Execute Test Computation';
    }
}

// Initialize app
async function initApp() {
    try {
        // Get app info
        const appInfo = await invoke('get_app_info');
        document.getElementById('app-version').textContent = `v${appInfo.version}`;
        
        addActivityLog('Application started');
        addActivityLog(`Version: ${appInfo.version}`);
        
        // Update kernel status
        await updateKernelStatus();
        
        // Initial health update
        await updateSystemHealth();
        
        // Set up periodic updates (every 2 seconds)
        updateInterval = setInterval(updateSystemHealth, 2000);
        
        // Set up event listeners
        document.getElementById('execute-btn').addEventListener('click', executeComputation);
        
        addActivityLog('System health monitoring active');
        
    } catch (error) {
        console.error('Failed to initialize app:', error);
        addActivityLog(`Initialization error: ${error}`);
    }
}

// Cleanup on unload
window.addEventListener('beforeunload', () => {
    if (updateInterval) {
        clearInterval(updateInterval);
    }
});

// Start the app when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}
