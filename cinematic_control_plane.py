#!/usr/bin/env python3
"""
QRATUM Cinematic Control Plane

Unified web interface for accessing all QRATUM services
"""

import os
import requests
from flask import Flask, render_template_string, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Service endpoints (internal Docker network)
SERVICES = {
    'qradle': {'url': 'http://qradle:8000', 'external_url': 'http://10.0.0.1:8001', 'name': 'QRADLE Foundation Engine', 'description': 'Core blockchain and cryptographic operations'},
    'platform': {'url': 'http://qratum-platform:8000', 'external_url': 'http://10.0.0.1:8002', 'name': 'QRATUM Platform', 'description': '14 vertical AI modules'},
    'asi': {'url': 'http://qratum-asi:8000', 'external_url': 'http://10.0.0.1:8003', 'name': 'QRATUM-ASI', 'description': 'Autonomous Systems Intelligence'},
    'grafana': {'url': 'http://grafana:3000', 'external_url': 'http://10.0.0.1:3000', 'name': 'Grafana', 'description': 'Monitoring & Visualization'},
    'prometheus': {'url': 'http://prometheus:9090', 'external_url': 'http://10.0.0.1:9090', 'name': 'Prometheus', 'description': 'Metrics Collection'},
    'loki': {'url': 'http://loki:3100', 'external_url': 'http://10.0.0.1:3100', 'name': 'Loki', 'description': 'Log Aggregation'}
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QRATUM Cinematic Control Plane</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
        }
        h1 {
            text-align: center;
            color: white;
            margin-bottom: 10px;
            font-size: 3em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .subtitle {
            text-align: center;
            color: rgba(255,255,255,0.8);
            margin-bottom: 40px;
            font-size: 1.2em;
        }
        .service-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 25px;
        }
        .service-card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
            border: 2px solid transparent;
            position: relative;
            overflow: hidden;
        }
        .service-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #667eea, #764ba2);
        }
        .service-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 25px 50px rgba(0,0,0,0.3);
            border-color: rgba(102, 126, 234, 0.5);
        }
        .service-card h3 {
            margin-top: 0;
            color: #333;
            font-size: 1.5em;
            margin-bottom: 10px;
        }
        .service-card p {
            color: #666;
            margin-bottom: 20px;
            line-height: 1.5;
        }
        .service-link {
            display: inline-block;
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            text-decoration: none;
            padding: 12px 24px;
            border-radius: 25px;
            transition: all 0.3s ease;
            font-weight: bold;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        .service-link:hover {
            background: linear-gradient(45deg, #5a6fd8, #6a4190);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        .status {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
            margin-left: 10px;
            vertical-align: middle;
        }
        .status.healthy {
            background: #4CAF50;
            color: white;
        }
        .status.unhealthy {
            background: #f44336;
            color: white;
        }
        .status.unknown {
            background: #ff9800;
            color: white;
        }
        .footer {
            text-align: center;
            margin-top: 60px;
            color: rgba(255,255,255,0.8);
            font-size: 0.9em;
        }
        .refresh-btn {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(255,255,255,0.2);
            color: white;
            border: 1px solid rgba(255,255,255,0.3);
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 0.9em;
            transition: all 0.3s ease;
        }
        .refresh-btn:hover {
            background: rgba(255,255,255,0.3);
            transform: scale(1.05);
        }
    </style>
</head>
<body>
    <button class="refresh-btn" onclick="location.reload()">🔄 Refresh Status</button>

    <div class="container">
        <h1>🎬 QRATUM Cinematic Control Plane</h1>
        <p class="subtitle">Unified sovereign AI infrastructure access portal</p>

        <div class="service-grid">
            {% for service_id, service in services.items() %}
            <div class="service-card">
                <h3>{{ service.icon }} {{ service.name }} <span class="status {{ service.status_class }}">{{ service.status_icon }}</span></h3>
                <p>{{ service.description }}</p>
                <p><strong>Endpoint:</strong> {{ service.external_url }}</p>
                <a href="{{ service.external_url }}" class="service-link" target="_blank">Access Service →</a>
            </div>
            {% endfor %}
        </div>

        <div class="footer">
            <p>🔒 Sovereign • 🔐 Secure • 🚀 Production Ready</p>
            <p>QRATUM v2.0.0 | Built for the future of AI | Status auto-refreshes every 30 seconds</p>
        </div>
    </div>

    <script>
        // Auto-refresh every 30 seconds
        setTimeout(() => {
            location.reload();
        }, 30000);
    </script>
</body>
</html>
"""

def check_service_status(url):
    """Check if a service is healthy."""
    try:
        response = requests.get(f"{url}/health", timeout=5)
        if response.status_code == 200:
            return "healthy"
        else:
            return "unhealthy"
    except:
        return "unknown"

@app.route('/')
def index():
    """Main control plane interface."""
    services_data = {}

    for service_id, service_info in SERVICES.items():
        status = check_service_status(service_info['url'])
        status_class = status
        status_icon = "●"

        # Add icons
        icons = {
            'qradle': '🔗',
            'platform': '🏗️',
            'asi': '🧠',
            'grafana': '📊',
            'prometheus': '📈',
            'loki': '📝'
        }

        services_data[service_id] = {
            **service_info,
            'status_class': status_class,
            'status_icon': status_icon,
            'icon': icons.get(service_id, '🔧')
        }

    return render_template_string(HTML_TEMPLATE, services=services_data)

@app.route('/api/status')
def api_status():
    """API endpoint for service status."""
    status_data = {}

    for service_id, service_info in SERVICES.items():
        status = check_service_status(service_info['url'])
        status_data[service_id] = {
            **service_info,
            'status': status,
            'reachable': status != 'unknown'
        }

    return jsonify({
        'timestamp': os.times()[4],  # Using process time as timestamp
        'services': status_data,
        'overall_status': 'healthy' if all(s['status'] == 'healthy' for s in status_data.values()) else 'degraded'
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', '8080'))
    app.run(host='0.0.0.0', port=port, debug=False)