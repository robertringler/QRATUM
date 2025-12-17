# QRATUM Customer Dashboard

Production-grade web dashboard for the QRATUM Quantum-Classical Convergence Platform.

## Overview

The QRATUM Dashboard provides a comprehensive interface for:

- **Job Management**: Submit, monitor, and manage quantum-classical simulations
- **Real-Time Monitoring**: Live GPU utilization, convergence metrics, and system status
- **Results Visualization**: Interactive 3D tensor field rendering and circuit diagrams
- **Resource Management**: Multi-cloud cluster status and cost tracking

## Features

### Job Management Interface
- Submit quantum-classical simulations with parameter forms
- Upload ANSYS/materials science input files (.inp, .dat, .json, .yaml, .h5)
- Queue management with priority indicators
- Batch job submission templates

### Real-Time Monitoring
- WebSocket-based live status updates
- GPU utilization metrics (NVIDIA cuQuantum stats)
- Tensor network convergence visualization
- Progress bars with estimated completion times

### Results Visualization
- 3D tensor field rendering (Three.js)
- Elastomeric deformation animations
- Quantum circuit diagrams
- Exportable reports (PDF, JSON, HDF5, CSV)

### Resource Management
- Multi-cloud cluster status (EKS/GKE/AKS)
- Cost tracking and budget alerts
- Historical usage analytics

## Technical Stack

- **Frontend**: Vanilla JavaScript (ES6+)
- **3D Graphics**: Three.js r128
- **Charts**: Chart.js 4.4
- **Styling**: CSS Custom Properties, Responsive Design
- **Accessibility**: WCAG 2.1 AA Compliant

## Getting Started

### Local Development

1. Navigate to the dashboard directory:
   ```bash
   cd dashboard
   ```

2. Start a local server:
   ```bash
   python -m http.server 3000
   ```

3. Open in browser:
   ```
   http://localhost:3000
   ```

### Demo Mode

The dashboard automatically enables **Demo Mode** when no backend server is available. In demo mode:
- Simulated job data is displayed
- GPU metrics are randomly generated
- WebSocket events are simulated locally

### Backend Integration

The dashboard expects a backend API at:
- REST API: `http://localhost:8000` (development) or same origin (production)
- WebSocket: `ws://localhost:8000/ws` (development) or `wss://host/ws` (production)

## API Endpoints

### Jobs
- `POST /api/v1/qc/simulate` - Submit simulation job
- `GET /api/v1/qc/jobs/{job_id}` - Get job status
- `POST /api/v1/jobs/{job_id}/cancel` - Cancel job

### Metrics
- `GET /api/v1/metrics/gpu` - GPU utilization
- `GET /api/v1/metrics/cost` - Cost tracking
- `GET /api/v1/cluster/status` - Cluster information

### WebSocket Events
- `job:update` - Job progress updates
- `job:complete` - Job completion notification
- `metrics:gpu` - Real-time GPU metrics
- `metrics:system` - System metrics
- `alert:new` - New alerts

## File Structure

```
dashboard/
├── index.html          # Main dashboard application
├── landing.html        # Marketing landing page
├── README.md           # This file
├── assets/
│   ├── css/
│   │   └── dashboard.css    # Dashboard styles
│   └── js/
│       ├── api.js           # API integration layer
│       ├── websocket.js     # WebSocket handler
│       ├── visualization.js # 3D visualization
│       └── dashboard.js     # Main application
└── marketing/
    ├── marketing.css        # Marketing page styles
    ├── marketing.js         # Marketing page scripts
    ├── pages.css            # Additional page styles
    ├── pricing.html         # Pricing page
    ├── security.html        # Security/Trust Center
    └── demo.html            # Demo request form
```

## Marketing Pages

The dashboard includes integrated marketing pages accessible via `landing.html`:

- **Landing Page** (`landing.html`): Hero section, features, use cases, CTAs
- **Pricing** (`marketing/pricing.html`): Tiered pricing plans with feature comparison
- **Security** (`marketing/security.html`): Compliance certifications and security architecture
- **Demo Request** (`marketing/demo.html`): Lead capture form for sales demos

Marketing pages share the dashboard's design system and link directly to the dashboard application.

## Accessibility

The dashboard is designed to meet WCAG 2.1 AA compliance:

- Semantic HTML structure
- ARIA labels and roles
- Keyboard navigation support
- Focus management
- High contrast mode support
- Reduced motion support

### Keyboard Shortcuts

- `Alt + 1`: Jobs view
- `Alt + 2`: Monitoring view
- `Alt + 3`: Results view
- `Alt + 4`: Resources view
- `Escape`: Close modal

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Compliance

This dashboard is designed for use in regulated environments:
- DO-178C Level A compatible
- CMMC 2.0 L2 ready
- Secure handling of sensitive simulation data

## License

Apache 2.0 - See LICENSE in repository root
