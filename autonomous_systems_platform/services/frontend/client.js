function getBackendUrl() {
    return window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
        ? 'http://localhost:8000'
        : '';
}

async function runKernel() {
    const backendUrl = getBackendUrl();
    
    const response = await fetch(`${backendUrl}/kernel`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ seed: 0, scale: 1.0 })
    });
    const data = await response.json();
    document.getElementById('output').innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
    
    // Automatically refresh history after kernel execution
    await loadHistory();
}

async function loadHistory() {
    const backendUrl = getBackendUrl();
    
    try {
        const response = await fetch(`${backendUrl}/history`);
        const data = await response.json();
        
        const historyDiv = document.getElementById('history');
        
        if (!data.transactions || data.transactions.length === 0) {
            historyDiv.innerHTML = '<p>No transactions yet. Click "Run Kernel" to create a transaction.</p>';
            return;
        }
        
        // Display transactions in reverse order (newest first)
        const transactionsHtml = data.transactions.slice().reverse().map(tx => {
            const timestamp = new Date(tx.timestamp).toLocaleString();
            return `
                <div class="transaction">
                    <div class="transaction-time">${timestamp}</div>
                    <div class="transaction-details">
                        <strong>Seed:</strong> ${tx.seed} | <strong>Scale:</strong> ${tx.scale} | <strong>Result:</strong> ${tx.result}
                    </div>
                </div>
            `;
        }).join('');
        
        historyDiv.innerHTML = transactionsHtml;
    } catch (error) {
        document.getElementById('history').innerHTML = '<p style="color: red;">Error loading history: ' + error.message + '</p>';
    }
}

// Load history on page load
window.addEventListener('DOMContentLoaded', loadHistory);
