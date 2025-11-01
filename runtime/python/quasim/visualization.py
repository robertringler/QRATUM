"""Visualization and benchmark dashboard generation.

Generates interactive Plotly dashboards with roofline models and performance metrics.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class BenchmarkResult:
    """Benchmark result with performance metrics."""
    name: str
    latency_ms: float
    throughput_gflops: float
    energy_joules: float
    efficiency_gflops_per_watt: float
    backend: str
    timestamp: float


class DashboardGenerator:
    """Generate interactive benchmark dashboards."""
    
    def __init__(self) -> None:
        self.results: List[BenchmarkResult] = []
        
    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result."""
        self.results.append(result)
        
    def generate_html(self, output_path: str = "docs/benchmarks.html") -> None:
        """Generate HTML dashboard with Plotly charts."""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>QuASIM Phase II Benchmark Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #2c3e50; }}
        .chart {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #3498db; }}
        .metric-label {{ color: #7f8c8d; margin-top: 5px; }}
    </style>
</head>
<body>
    <h1>QuASIM Phase II Benchmark Dashboard</h1>
    
    <div class="metrics">
        <div class="metric-card">
            <div class="metric-value">{avg_throughput:.1f}</div>
            <div class="metric-label">Avg Throughput (GFLOPs)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{avg_latency:.2f}</div>
            <div class="metric-label">Avg Latency (ms)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{avg_efficiency:.1f}</div>
            <div class="metric-label">Avg Efficiency (GFLOPs/W)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{total_energy:.2f}</div>
            <div class="metric-label">Total Energy (J)</div>
        </div>
    </div>
    
    <div class="chart" id="throughput-chart"></div>
    <div class="chart" id="energy-chart"></div>
    <div class="chart" id="roofline-chart"></div>
    
    <script>
        // Throughput comparison
        var throughputData = {{
            x: {names},
            y: {throughputs},
            type: 'bar',
            marker: {{ color: '#3498db' }}
        }};
        Plotly.newPlot('throughput-chart', [throughputData], {{
            title: 'Throughput Comparison',
            xaxis: {{ title: 'Benchmark' }},
            yaxis: {{ title: 'GFLOPs' }}
        }});
        
        // Energy efficiency
        var energyData = {{
            x: {names},
            y: {efficiencies},
            type: 'bar',
            marker: {{ color: '#2ecc71' }}
        }};
        Plotly.newPlot('energy-chart', [energyData], {{
            title: 'Energy Efficiency',
            xaxis: {{ title: 'Benchmark' }},
            yaxis: {{ title: 'GFLOPs/W' }}
        }});
        
        // Roofline model
        var rooflineData = {{
            x: {arithmetic_intensities},
            y: {throughputs},
            mode: 'markers',
            type: 'scatter',
            marker: {{ size: 10, color: '#e74c3c' }},
            text: {names}
        }};
        Plotly.newPlot('roofline-chart', [rooflineData], {{
            title: 'Roofline Performance Model',
            xaxis: {{ title: 'Arithmetic Intensity (FLOPs/Byte)', type: 'log' }},
            yaxis: {{ title: 'Performance (GFLOPs)', type: 'log' }}
        }});
    </script>
</body>
</html>
"""
        
        # Compute metrics
        if not self.results:
            avg_throughput = 0.0
            avg_latency = 0.0
            avg_efficiency = 0.0
            total_energy = 0.0
            names = []
            throughputs = []
            efficiencies = []
            arithmetic_intensities = []
        else:
            avg_throughput = sum(r.throughput_gflops for r in self.results) / len(self.results)
            avg_latency = sum(r.latency_ms for r in self.results) / len(self.results)
            avg_efficiency = sum(r.efficiency_gflops_per_watt for r in self.results) / len(self.results)
            total_energy = sum(r.energy_joules for r in self.results)
            
            names = [r.name for r in self.results]
            throughputs = [r.throughput_gflops for r in self.results]
            efficiencies = [r.efficiency_gflops_per_watt for r in self.results]
            # Simulate arithmetic intensity
            arithmetic_intensities = [t / 10.0 for t in throughputs]
        
        html = html_template.format(
            avg_throughput=avg_throughput,
            avg_latency=avg_latency,
            avg_efficiency=avg_efficiency,
            total_energy=total_energy,
            names=json.dumps(names),
            throughputs=json.dumps(throughputs),
            efficiencies=json.dumps(efficiencies),
            arithmetic_intensities=json.dumps(arithmetic_intensities),
        )
        
        # Write to file
        import pathlib
        output = pathlib.Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html)
        
    def export_json(self, output_path: str = "docs/benchmarks.json") -> None:
        """Export results as JSON."""
        import pathlib
        data = {
            "results": [
                {
                    "name": r.name,
                    "latency_ms": r.latency_ms,
                    "throughput_gflops": r.throughput_gflops,
                    "energy_joules": r.energy_joules,
                    "efficiency_gflops_per_watt": r.efficiency_gflops_per_watt,
                    "backend": r.backend,
                    "timestamp": r.timestamp,
                }
                for r in self.results
            ]
        }
        output = pathlib.Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, indent=2))
