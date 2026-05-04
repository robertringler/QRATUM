# QuASIM × QuNimbus Figma Assets - Quick Start

This document provides a quick overview of the Figma assets integration for QuASIM × QuNimbus.

## Overview

QuASIM × QuNimbus now has a complete infrastructure for creating and managing professional marketing assets and visualizations using Figma. This includes:

- **Comprehensive documentation** for design workflows
- **Brand guidelines** ensuring visual consistency
- **Directory structure** for organized asset management
- **Templates** for common design needs

## Quick Links

### Documentation
- **[Figma Integration Guide](FIGMA_INTEGRATION_GUIDE.md)** - Complete workflow guide
- **[Brand Guidelines](BRAND_GUIDELINES.md)** - Colors, typography, and visual identity
- **[Marketing Assets README](../assets/marketing/README.md)** - Using marketing assets
- **[Social Media Assets README](../assets/social/README.md)** - Social media specifications

### Directories
- **`docs/assets/marketing/`** - Diagrams, icons, illustrations, screenshots
- **`docs/assets/social/`** - Platform-specific social media assets
- **`figma/`** - Figma project references and templates

## Getting Started

### 1. Familiarize Yourself with Brand Guidelines

Read the [Brand Guidelines](BRAND_GUIDELINES.md) to understand:
- Color palette (#002b36, #073642, #268bd2, #2aa198)
- Typography (Inter for UI, Roboto Mono for code)
- Logo usage rules
- Design principles

### 2. Access Figma Projects

Request access to QuASIM Figma workspace:
1. Open GitHub issue with label `access-request`
2. Provide your Figma account email
3. Specify required access level (view/edit)

**Primary Projects**:
- `QuASIM_QuNimbus_Marketing` - Marketing pages, social media
- `QuASIM_Design_System` - Brand components and templates
- `QuASIM_Pitch_Deck` - Investor presentations

### 3. Create Your First Asset

Follow the workflow in the [Figma Integration Guide](FIGMA_INTEGRATION_GUIDE.md#integration-workflow):

1. **Design** in Figma using brand guidelines
2. **Export** as SVG (vector) or PNG (raster)
3. **Optimize** file size using SVGO or ImageOptim
4. **Place** in appropriate directory
5. **Commit** to repository

### 4. Use Templates

Browse available templates in `figma/templates/` or within Figma projects:
- Landing page layouts
- Social media post templates
- Pitch deck slide templates
- Diagram templates
- Feature cards
- Certification badges

## Common Use Cases

### Creating a Social Media Post

1. Open `QuASIM_QuNimbus_Marketing` in Figma
2. Navigate to `04_Social_Media`
3. Duplicate appropriate template (Twitter, LinkedIn, Instagram)
4. Customize headline, body text, and visuals
5. Export as PNG at specified dimensions
6. Save to `docs/assets/social/{platform}/`

**Dimensions Reference**:
- Twitter Card: 1200×675px
- LinkedIn Post: 1200×627px
- Instagram Square: 1080×1080px

### Creating an Architecture Diagram

1. Open `QuASIM_QuNimbus_Marketing` in Figma
2. Navigate to `05_Diagrams`
3. Use diagram template or start from scratch
4. Follow brand colors for component boxes
5. Export as SVG (vector format)
6. Optimize with SVGO
7. Save to `docs/assets/marketing/diagrams/`

### Creating Certification Badges

1. Open `QuASIM_Design_System` in Figma
2. Navigate to `Components → Badges`
3. Duplicate certification badge template
4. Customize certification name and level
5. Export as SVG
6. Save to `docs/assets/marketing/badges/`

**Available Certifications**:
- DO-178C Level A (Orange background)
- CMMC 2.0 Level 2 (Green background)
- NIST 800-53 Rev 5 (Blue background)

## Design Principles

### Visual Identity

**Dark Theme**:
- Base Dark (#002b36) for primary backgrounds
- Base Medium (#073642) for panels and cards
- Light text (#fdf6e3) on dark backgrounds

**Quantum Aesthetic**:
- Clean vector lines (no blur effects)
- Geometric shapes
- Cyan/blue accents for "quantum glow"
- Minimalist design language

**Typography**:
- Inter: UI elements, marketing copy
- Roboto Mono: Code snippets, technical specs

### Consistency Rules

1. **Always** use colors from the brand palette
2. **Always** use Inter or Roboto Mono fonts
3. **Always** maintain minimum 20px clear space around logo
4. **Always** ensure WCAG AA contrast ratios (4.5:1 for body text)
5. **Never** modify logo colors or proportions
6. **Never** use fonts outside the approved set

## File Organization

### Naming Conventions

```
{category}_{description}_{variant}_{size}.{ext}

Examples:
✅ diagram_architecture_overview.svg
✅ icon_gpu_acceleration_48x48.svg
✅ twitter_card_certification_announcement_v1.png
✅ badge_cmmc_level2.svg

❌ new_diagram.svg (too generic)
❌ icon1.svg (not descriptive)
❌ twitterCard.png (wrong case)
```

### Directory Structure

```
docs/assets/
├── marketing/              # Marketing and documentation assets
│   ├── diagrams/          # SVG architecture and flow diagrams
│   ├── icons/             # SVG icons (24×24, 48×48)
│   ├── illustrations/     # SVG illustrations and graphics
│   ├── screenshots/       # PNG product screenshots
│   ├── badges/            # SVG certification badges
│   └── templates/         # Reusable template files
└── social/                # Social media assets
    ├── twitter/           # Twitter/X cards and headers
    ├── linkedin/          # LinkedIn posts and banners
    ├── instagram/         # Instagram posts and stories
    ├── facebook/          # Facebook posts and covers
    └── youtube/           # YouTube thumbnails

figma/
├── exports/               # Temporary staging for exports
│   ├── svg/              # SVG exports before optimization
│   └── png/              # PNG exports before optimization
└── templates/             # Template documentation
```

## Export Settings

### SVG (Vector Graphics)

**Use for**: Logos, icons, diagrams, illustrations

**Settings**:
- Format: SVG
- Include "id" attributes: ✓
- Outline text: Optional (keep editable when possible)
- Simplify paths: ✓

**Optimization**: Run through SVGO after export

### PNG (Raster Graphics)

**Use for**: Screenshots, social media, photos

**Settings**:
- Format: PNG
- Scale: 1x (standard), 2x (retina for high-DPI)
- Background: Transparent (where applicable)

**Optimization**: Use ImageOptim, TinyPNG, or Squoosh

**Target file size**: < 500KB for web assets

## Workflow Summary

```
┌─────────────────┐
│ 1. Design       │
│    in Figma     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Export       │
│    SVG or PNG   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. Optimize     │
│    File Size    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. Place in     │
│    Directory    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 5. Commit to    │
│    Repository   │
└─────────────────┘
```

## Key Messages for Marketing

When creating marketing materials, emphasize these key differentiators:

1. **Certification Moat**: Only quantum platform with DO-178C Level A certification
2. **Mission Validated**: Proven with SpaceX Falcon 9 and NASA telemetry (<2% RMSE)
3. **Production Ready**: 99.95% SLA, multi-cloud Kubernetes, enterprise infrastructure
4. **Economic Integration**: QuNimbus links quantum efficiency to valuation
5. **Autonomous Evolution**: Phase III RL optimization with formal verification

## Tone & Style

**Professional + Futuristic + Technical + Accessible**

**Do**:
- Use active voice
- Be specific with metrics
- Cite validation results
- Focus on benefits

**Don't**:
- Overhype capabilities
- Use buzzwords without substance
- Over-simplify technical accuracy
- Excessive exclamation points

## Support & Questions

### For Design Questions
- Open GitHub issue with `design` label
- Reference brand guidelines or integration guide
- Include visual examples when helpful

### For Access Requests
- Open GitHub issue with `access-request` label
- Provide Figma account email
- Specify required projects and access level

### For Technical Questions
- Contact development team via GitHub
- Tag relevant team members

## Next Steps

1. **Review** [Brand Guidelines](BRAND_GUIDELINES.md) and [Figma Integration Guide](FIGMA_INTEGRATION_GUIDE.md)
2. **Request** Figma access if you'll be creating designs
3. **Create** your first asset using templates
4. **Export** and optimize following the workflow
5. **Commit** to repository with descriptive commit message

---

## Additional Resources

### External Links
- [Figma Website](https://www.figma.com/)
- [Figma API Documentation](https://www.figma.com/developers/api)
- [Inter Font](https://rsms.me/inter/)
- [Roboto Mono Font](https://fonts.google.com/specimen/Roboto+Mono)
- [SVGO (SVG Optimizer)](https://github.com/svg/svgo)
- [ImageOptim](https://imageoptim.com/)

### Internal Links
- [Main README](../../README.md)
- [Architecture Documentation](../technical/architecture.md)
- [Valuation Dashboard](../valuation_dashboard.html)
- [Contributing Guidelines](../../CONTRIBUTING.md)

---

**Last Updated**: November 12, 2025  
**Maintained By**: QuASIM Marketing & Design Team
