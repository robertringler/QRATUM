// QuASIM Technical Portal - Chart Visualizations
// Chart.js integration for performance metrics and compliance dashboards

(function() {
  'use strict';

  // Wait for Chart.js to load
  if (typeof Chart === 'undefined') {
    console.warn('Chart.js not loaded. Visualizations will not render.');
    return;
  }

  // Global Chart.js defaults
  Chart.defaults.color = '#999';
  Chart.defaults.borderColor = 'rgba(0, 255, 136, 0.2)';
  Chart.defaults.font.family = "'JetBrains Mono', monospace";

  // ============================================================================
  // Qubit Scaling Chart
  // ============================================================================

  window.QuASIM = window.QuASIM || {};

  window.QuASIM.QubitScalingChart = function(canvas, data) {
    return new Chart(canvas, {
      type: 'line',
      data: {
        labels: data.labels,
        datasets: data.datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            position: 'top',
            labels: { color: '#e0e0e0' }
          },
          title: {
            display: true,
            text: 'Qubit Scaling Performance',
            color: '#00ff88',
            font: { size: 16, weight: 'bold' }
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: data.yAxisLabel || 'Execution Time (s)',
              color: '#e0e0e0'
            },
            grid: { color: 'rgba(0, 255, 136, 0.1)' }
          },
          x: {
            title: {
              display: true,
              text: data.xAxisLabel || 'Number of Qubits',
              color: '#e0e0e0'
            },
            grid: { color: 'rgba(0, 255, 136, 0.1)' }
          }
        }
      }
    });
  };

  // ============================================================================
  // GPU Gauge
  // ============================================================================

  window.QuASIM.GPUGauge = function(canvas, value, label) {
    const percent = value / 100;
    const angle = Math.PI * percent;

    return new Chart(canvas, {
      type: 'doughnut',
      data: {
        datasets: [{
          data: [value, 100 - value],
          backgroundColor: [
            `rgba(0, 255, 136, ${percent})`,
            'rgba(18, 18, 26, 0.5)'
          ],
          borderWidth: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '75%',
        circumference: 180,
        rotation: -90,
        plugins: {
          legend: { display: false },
          tooltip: { enabled: false }
        }
      },
      plugins: [{
        afterDraw: (chart) => {
          const ctx = chart.ctx;
          const width = chart.width;
          const height = chart.height;

          ctx.restore();
          ctx.font = 'bold 24px "JetBrains Mono"';
          ctx.textBaseline = 'middle';
          ctx.textAlign = 'center';
          ctx.fillStyle = '#00ff88';
          ctx.fillText(value + '%', width / 2, height / 2 - 10);

          ctx.font = '12px "JetBrains Mono"';
          ctx.fillStyle = '#999';
          ctx.fillText(label || 'Utilization', width / 2, height / 2 + 15);
          ctx.save();
        }
      }]
    });
  };

  // ============================================================================
  // Throughput-Latency Chart
  // ============================================================================

  window.QuASIM.ThroughputLatencyChart = function(canvas, data) {
    return new Chart(canvas, {
      type: 'scatter',
      data: {
        datasets: [{
          label: 'Performance',
          data: data.data.map(d => ({ x: d.throughput, y: d.latency })),
          backgroundColor: 'rgba(0, 255, 136, 0.5)',
          borderColor: '#00ff88',
          borderWidth: 2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          title: {
            display: true,
            text: 'Throughput vs Latency',
            color: '#00ff88',
            font: { size: 16, weight: 'bold' }
          }
        },
        scales: {
          x: {
            title: {
              display: true,
              text: data.xAxisLabel || 'Throughput (ops/sec)',
              color: '#e0e0e0'
            },
            grid: { color: 'rgba(0, 255, 136, 0.1)' }
          },
          y: {
            title: {
              display: true,
              text: data.yAxisLabel || 'Latency (ms)',
              color: '#e0e0e0'
            },
            grid: { color: 'rgba(0, 255, 136, 0.1)' }
          }
        }
      }
    });
  };

  // ============================================================================
  // Multi-GPU Scaling Chart
  // ============================================================================

  window.QuASIM.MultiGPUScalingChart = function(canvas, data) {
    return new Chart(canvas, {
      type: 'bar',
      data: {
        labels: data.labels,
        datasets: data.datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            position: 'top',
            labels: { color: '#e0e0e0' }
          },
          title: {
            display: true,
            text: 'Multi-GPU Scaling Efficiency',
            color: '#00ff88',
            font: { size: 16, weight: 'bold' }
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: data.yAxisLabel || 'Speedup Factor',
              color: '#e0e0e0'
            },
            grid: { color: 'rgba(0, 255, 136, 0.1)' }
          },
          x: {
            title: {
              display: true,
              text: data.xAxisLabel || 'GPU Configuration',
              color: '#e0e0e0'
            },
            grid: { color: 'rgba(0, 255, 136, 0.1)' }
          }
        }
      }
    });
  };

  // ============================================================================
  // Memory Calculator
  // ============================================================================

  window.QuASIM.MemoryCalculator = function(container, data) {
    const qubits = data.qubits;
    const precisions = ['fp64', 'fp32', 'fp16'];

    const table = document.createElement('table');
    table.className = 'table';
    table.innerHTML = `
      <thead>
        <tr>
          <th>Qubits</th>
          <th>FP64 (GB)</th>
          <th>FP32 (GB)</th>
          <th>FP16 (GB)</th>
        </tr>
      </thead>
      <tbody></tbody>
    `;

    const tbody = table.querySelector('tbody');

    qubits.forEach((q, i) => {
      const row = document.createElement('tr');
      row.innerHTML = `
        <td><strong>${q}</strong></td>
        <td>${data.fp64[i].toFixed(6)}</td>
        <td>${data.fp32[i].toFixed(6)}</td>
        <td>${data.fp16[i].toFixed(6)}</td>
      `;
      tbody.appendChild(row);
    });

    container.appendChild(table);
  };

  // ============================================================================
  // State Vector Visualization
  // ============================================================================

  window.QuASIM.StateVectorViz = function(container, amplitudes) {
    const stateVector = document.createElement('div');
    stateVector.className = 'state-vector';

    amplitudes.forEach((amp, i) => {
      const stateItem = document.createElement('div');
      stateItem.className = 'state-item';

      const label = document.createElement('div');
      label.className = 'state-label';
      label.textContent = `|${i.toString(2).padStart(Math.log2(amplitudes.length), '0')}⟩`;

      const bar = document.createElement('div');
      bar.className = 'state-bar';

      const barFill = document.createElement('div');
      barFill.className = 'state-bar-fill';
      barFill.style.width = (Math.abs(amp) * 100) + '%';

      const value = document.createElement('div');
      value.className = 'state-value';
      value.textContent = amp.toFixed(4);

      bar.appendChild(barFill);
      stateItem.appendChild(label);
      stateItem.appendChild(bar);
      stateItem.appendChild(value);
      stateVector.appendChild(stateItem);
    });

    container.appendChild(stateVector);
  };

  // ============================================================================
  // Compliance Chart
  // ============================================================================

  window.QuASIM.ComplianceChart = function(canvas, data) {
    return new Chart(canvas, {
      type: 'radar',
      data: {
        labels: data.families.map(f => f.id),
        datasets: [{
          label: 'Compliance %',
          data: data.families.map(f => f.percentage),
          backgroundColor: 'rgba(0, 255, 136, 0.2)',
          borderColor: '#00ff88',
          borderWidth: 2,
          pointBackgroundColor: '#00ff88',
          pointBorderColor: '#fff',
          pointHoverBackgroundColor: '#fff',
          pointHoverBorderColor: '#00ff88'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          title: {
            display: true,
            text: 'NIST 800-53 Compliance',
            color: '#00ff88',
            font: { size: 16, weight: 'bold' }
          }
        },
        scales: {
          r: {
            beginAtZero: true,
            max: 100,
            ticks: {
              stepSize: 20,
              color: '#999'
            },
            grid: { color: 'rgba(0, 255, 136, 0.2)' },
            pointLabels: { color: '#e0e0e0', font: { size: 12 } }
          }
        }
      }
    });
  };

  // ============================================================================
  // Auto-initialize charts from data attributes
  // ============================================================================

  document.addEventListener('DOMContentLoaded', async function() {
    // Load benchmark data
    const benchmarkData = await window.QuASIM.getJSON('assets/data/benchmarks.json');
    const complianceData = await window.QuASIM.getJSON('assets/data/compliance.json');

    if (!benchmarkData || !complianceData) {
      console.error('Failed to load chart data');
      return;
    }

    // Initialize charts based on data attributes
    const charts = document.querySelectorAll('[data-chart]');

    charts.forEach(element => {
      const chartType = element.getAttribute('data-chart');
      const canvas = element.querySelector('canvas');

      if (!canvas) return;

      try {
        switch (chartType) {
          case 'qubit-scaling':
            window.QuASIM.QubitScalingChart(canvas, benchmarkData.qubitScaling);
            break;
          case 'gpu-scaling':
            window.QuASIM.MultiGPUScalingChart(canvas, benchmarkData.gpuScaling);
            break;
          case 'throughput-latency':
            window.QuASIM.ThroughputLatencyChart(canvas, benchmarkData.throughputLatency);
            break;
          case 'compliance':
            window.QuASIM.ComplianceChart(canvas, complianceData.nist80053);
            break;
          case 'gpu-gauge':
            const value = parseInt(element.getAttribute('data-value') || 85);
            const label = element.getAttribute('data-label') || 'GPU';
            window.QuASIM.GPUGauge(canvas, value, label);
            break;
        }
      } catch (error) {
        console.error(`Error creating ${chartType} chart:`, error);
      }
    });
  });

})();
