"""Flask application for QuASIM autonomous systems backend."""
import os
import logging
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, generate_latest

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Metrics
REQUEST_COUNT = Counter('autonomous_systems_requests_total', 'Total requests', ['endpoint'])
REQUEST_LATENCY = Histogram('autonomous_systems_request_latency_seconds', 'Request latency')

try:
    from quasim.kernels.autonomous_systems import autonomous_systems_kernel
except ImportError:
    from quasim_fallback import autonomous_systems_kernel
    logging.warning("Using fallback kernel (JAX not available)")

@app.route('/health', methods=['GET'])
def health():
    REQUEST_COUNT.labels(endpoint='/health').inc()
    return jsonify({"status": "healthy"}), 200

@app.route('/kernel', methods=['POST'])
@REQUEST_LATENCY.time()
def kernel():
    REQUEST_COUNT.labels(endpoint='/kernel').inc()
    data = request.get_json() or {}
    seed = data.get('seed', 0)
    scale = data.get('scale', 1.0)
    result = autonomous_systems_kernel(seed=seed, scale=scale)
    return jsonify({"result": result}), 200

@app.route('/metrics', methods=['GET'])
def metrics():
    return generate_latest(), 200, {'Content-Type': 'text/plain'}

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
