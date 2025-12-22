# Marketing Assets

This directory contains visual assets for QuASIM × QuNimbus marketing materials, including diagrams, illustrations, icons, and other design elements exported from Figma.

## Directory Structure

```
marketing/
├── README.md           # This file
├── diagrams/           # Architecture and flow diagrams (SVG)
├── icons/              # Custom icons (SVG, 24×24 and 48×48)
├── illustrations/      # Quantum-themed illustrations (SVG)
├── screenshots/        # Product screenshots (PNG)
├── badges/             # Certification and compliance badges (SVG)
└── templates/          # Reusable marketing templates (SVG)
```

## Asset Guidelines

### File Naming Convention

Use the following pattern for all files:

```
{category}_{description}_{variant}.{ext}

Examples:
- diagram_architecture_overview.svg
- icon_gpu_acceleration_48x48.svg
- illustration_quantum_entanglement.svg
- screenshot_dashboard_1920x1080.png
- badge_do178c_level_a.svg
```

### File Formats

- **SVG**: For logos, icons, diagrams, and vector illustrations
  - Keep files optimized (< 100KB when possible)
  - Use meaningful IDs for elements
  - Preserve editability when practical

- **PNG**: For screenshots and raster graphics
  - Export at 1x and 2x (retina) resolutions
  - Use transparent backgrounds where appropriate
  - Optimize file size (target < 500KB for web)

### Color Usage

All assets should follow the QuASIM brand color palette:

| Color | Hex Code | Usage |
|-------|----------|-------|
| Base Dark | `#002b36` | Backgrounds |
| Base Medium | `#073642` | Panels |
| Accent Blue | `#268bd2` | Primary accent |
| Accent Cyan | `#2aa198` | Quantum glow |
| Light Text | `#fdf6e3` | Text |
| Orange | `#cb4b16` | Alerts |

See [Brand Guidelines](../marketing/FIGMA_INTEGRATION_GUIDE.md#brand-guidelines) for complete palette.

## Usage

### In Documentation

Reference assets using relative paths:

```markdown
![Architecture Diagram](../assets/marketing/diagrams/diagram_architecture_overview.svg)
```

### In Web Content

```html
<img src="docs/assets/marketing/icons/icon_quantum_simulation_48x48.svg" 
     alt="Quantum Simulation Icon" 
     width="48" 
     height="48">
```

### In Presentations

Export PNG versions at appropriate resolution (1920×1080 for slides).

## Creating New Assets

1. **Design in Figma**: Use the `QuASIM_QuNimbus_Marketing` project
2. **Follow Guidelines**: Adhere to brand colors and typography
3. **Export**: Use Figma export settings (SVG or PNG as appropriate)
4. **Optimize**: Run through SVGO (for SVG) or ImageOptim (for PNG)
5. **Place**: Add to appropriate subdirectory
6. **Document**: Update this README with asset description
7. **Commit**: Use conventional commit message format

## Subdirectory Details

### diagrams/

**Purpose**: Technical and architectural diagrams

**Content**:

- Architecture overview
- Data flow diagrams
- Quantum runtime visualization
- Φ_QEVF control flow
- QMP market mesh
- Integration diagrams

**Format**: SVG (vector)

### icons/

**Purpose**: UI and marketing icons

**Content**:

- Feature icons (GPU, compliance, RL optimization)
- Technology icons (Kubernetes, cuQuantum, etc.)
- Action icons (download, deploy, monitor)
- Custom quantum-themed icons

**Format**: SVG (24×24px, 48×48px)

### illustrations/

**Purpose**: Decorative and explanatory illustrations

**Content**:

- Quantum entanglement visualizations
- Tensor network representations
- Abstract quantum backgrounds
- Hero section graphics
- Conceptual diagrams

**Format**: SVG (vector)

### screenshots/

**Purpose**: Product interface captures

**Content**:

- Dashboard views
- CLI output examples
- Monitoring interfaces
- Kubernetes deployments
- Simulation results

**Format**: PNG (1920×1080px standard)

### badges/

**Purpose**: Certification and compliance indicators

**Content**:

- DO-178C Level A badge
- CMMC 2.0 Level 2 badge
- NIST 800-53 Rev 5 badge
- DFARS compliance badge
- License badges (Apache 2.0)

**Format**: SVG (vector)

### templates/

**Purpose**: Reusable marketing templates

**Content**:

- Feature card templates
- Metric display templates
- Testimonial templates
- Case study templates

**Format**: SVG (vector)

## Asset Inventory

### Current Assets

| Asset | Location | Format | Description |
|-------|----------|--------|-------------|
| (Placeholder) | - | - | Assets to be added |

### Planned Assets

| Asset | Priority | Assignee | Status |
|-------|----------|----------|--------|
| Architecture Diagram | High | - | Pending |
| Feature Icons Set | High | - | Pending |
| Certification Badges | High | - | Pending |
| Hero Illustration | Medium | - | Pending |
| Product Screenshots | Medium | - | Pending |

## Maintenance

- **Review Frequency**: Monthly
- **Optimization**: Quarterly review of file sizes
- **Deprecation**: Archive unused assets to `archive/` subdirectory
- **Updates**: Document changes in top-level CHANGELOG.md

## Related Documentation

- [Figma Integration Guide](../marketing/FIGMA_INTEGRATION_GUIDE.md)
- [Brand Guidelines](../marketing/FIGMA_INTEGRATION_GUIDE.md#brand-guidelines)
- [Main README](../../README.md)

## Questions?

For questions about marketing assets or requests for new designs:

- Open a GitHub issue with label `design` or `documentation`
- Tag the marketing team in Figma comments
- Reference this README in discussions

---

**Last Updated**: November 12, 2025
