# QuASIM Technical Portal Website

This directory contains the production-ready static website for the QuASIM (Quantum-Accelerated Simulation and Modeling Engine) Technical Portal.

## Overview

QuASIM is a production-grade quantum simulation platform engineered for regulated industries requiring:
- Aerospace certification (DO-178C Level A)
- Defense compliance (NIST 800-53/171, CMMC 2.0 L2, DFARS)
- Deterministic reproducibility
- GPU-accelerated tensor network simulation
- Multi-cloud Kubernetes orchestration with 99.95% SLA

## Website Structure

### HTML Pages
- `index.html` - Landing page with hero section, KPI cards, and feature highlights
- `architecture.html` - Full-stack architecture diagrams and visualizations
- `whitepaper.html` - Technical whitepaper with mathematical details
- `compliance.html` - Security dashboard with compliance metrics
- `performance.html` - Benchmark visualizations and performance data
- `api.html` - API documentation with code examples
- `roadmap.html` - Development timeline and feature roadmap
- `404.html` - Custom error page

### Stylesheets (css/)
- `main.css` - Core styles, layout, and typography
- `components.css` - UI component library
- `animations.css` - Animation keyframes and effects
- `visualizations.css` - Chart and diagram styles

### JavaScript (js/)
- `main.js` - Core initialization and utilities
- `particles.js` - Quantum particle background system
- `terminal.js` - Terminal typing and simulation effects
- `visualizations.js` - Chart.js integration and data visualizations
- `three-scene.js` - Three.js 3D scenes (Bloch sphere, circuits)
- `navigation.js` - Keyboard shortcuts and page navigation

### Data (assets/data/)
- `benchmarks.json` - Performance benchmark data
- `compliance.json` - Security and compliance metrics

## Design System

### Color Palette
- **Void Black**: `#0a0a0f` - Primary background
- **Tactical Dark**: `#12121a` - Secondary background
- **Surface Elevated**: `#1a1a24` - Card surfaces
- **Quantum Green**: `#00ff88` - Primary accent
- **Plasma Cyan**: `#00d4ff` - Secondary accent
- **Alert Crimson**: `#ff3366` - Errors and warnings
- **Warning Amber**: `#ffaa00` - Cautions
- **Ghost White**: `#e0e0e0` - Text

### Typography
- **Display**: Orbitron (hero titles)
- **Headings**: JetBrains Mono
- **Body**: Inter
- **Code**: IBM Plex Mono / JetBrains Mono

## Features

- ✅ Particle background system with quantum-themed effects
- ✅ Glitch text effects on hero titles
- ✅ CRT scanline overlay animation
- ✅ Terminal typing effects with command demos
- ✅ Interactive architecture layer diagrams
- ✅ Animated circular gauge visualizations
- ✅ Chart.js benchmark visualizations
- ✅ Three.js 3D Bloch sphere with gate animations
- ✅ STRIDE threat matrix with progress bars
- ✅ Keyboard shortcut navigation system
- ✅ Reading progress indicator
- ✅ Back to top button
- ✅ Mobile-responsive navigation
- ✅ Classification banner on all pages
- ✅ WCAG 2.1 AA compliant accessibility

## External Dependencies (CDN)

- [Google Fonts](https://fonts.google.com): Inter, JetBrains Mono, Orbitron
- [Chart.js](https://www.chartjs.org) v4.4.1
- [Three.js](https://threejs.org) v0.160.0 with OrbitControls
- [KaTeX](https://katex.org) v0.16.9 (for mathematical notation)

## Development

To preview locally:
1. Install a local web server (e.g., `python -m http.server`)
2. Navigate to the docs directory
3. Open `http://localhost:8000` in your browser

For GitHub Pages deployment:
- The site is automatically deployed from the `docs/` directory
- Access at: https://robertringler.github.io/QRATUM

## Accessibility

This site follows WCAG 2.1 AA guidelines:
- Proper heading hierarchy
- ARIA labels and landmarks
- Keyboard navigation support
- Skip to content links
- Sufficient color contrast
- Focus indicators

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile browsers (iOS Safari, Chrome Mobile)

## License

Copyright © 2025 QuASIM Project. All rights reserved.
Licensed under Apache 2.0 License.
