async function runKernel() {
    const response = await fetch('/kernel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ seed: 0, scale: 1.0 })
    });
    const data = await response.json();
    document.getElementById('output').innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
}
