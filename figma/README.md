# Figma Project Files

This directory contains Figma project files and design system documentation for QuASIM × QuNimbus visual assets and marketing materials.

## Overview

All visual design work for QuASIM × QuNimbus is managed through Figma, a collaborative design tool. This directory serves as a central reference point for Figma projects and related documentation.

## Figma Projects

### Primary Projects

| Project Name | Purpose | Figma Link | Status |
|--------------|---------|------------|--------|
| QuASIM_QuNimbus_Marketing | Marketing pages, social media, pitch deck | TBD | Planned |
| QuASIM_Design_System | Brand assets, components, templates | TBD | Planned |
| QuASIM_Pitch_Deck | Investor presentation | TBD | Planned |
| QuASIM_Documentation_Diagrams | Technical diagrams for docs | TBD | Planned |

### Project Access

To gain access to Figma projects:

1. Request access from the QuASIM team via GitHub issue
2. Label issue with `design` or `access-request`
3. Provide Figma account email
4. Specify required access level (view/edit)

## Directory Structure

```
figma/
├── README.md                   # This file
├── QuASIM_QuNimbus_Marketing.fig  # (Placeholder - actual files stored in Figma cloud)
├── QuASIM_Design_System.fig       # (Placeholder - design system components)
├── QuASIM_Pitch_Deck.fig          # (Placeholder - investor deck)
├── exports/                       # Temporary export staging area
│   ├── svg/                       # SVG exports before optimization
│   └── png/                       # PNG exports before optimization
└── templates/                     # Reusable Figma templates
    └── README.md                  # Template documentation
```

## Working with Figma Files

### Access Figma Projects

Figma projects are stored in the cloud and accessed via web browser or desktop app:

1. **Web**: Visit [figma.com](https://www.figma.com) and sign in
2. **Desktop**: Download [Figma Desktop App](https://www.figma.com/downloads/)
3. **Navigate**: Go to QuASIM workspace → Select project

### File Naming Convention

Within Figma projects, use consistent page and frame naming:

**Page Names**:

- `01_Brand_Assets`
- `02_Marketing_Pages`
- `03_Pitch_Deck`
- `04_Social_Media`
- `05_Diagrams`
- `06_Illustrations`
- `07_Exports`

**Frame Names**:

```
{page}_{component}_{variant}_{size}

Examples:
- Landing_Hero_Desktop_1920x1080
- Social_Twitter_Card_Launch_v1
- Diagram_Architecture_Overview
- Icon_GPU_Acceleration_48x48
```

### Version Control

#### Figma Native Versioning

Figma has built-in version history:

- **Save Version**: File → Save to Version History
- **Name Version**: Use descriptive names (e.g., "v1.0 - Initial Marketing Pages")
- **Restore**: File → Show Version History → Select version

#### Git Synchronization

Actual `.fig` files are binary and stored in Figma's cloud. This directory contains:

- References to Figma projects (links in README)
- Exported assets (staged in `exports/` before moving to `docs/assets/`)
- Documentation and metadata

## Export Workflow

### Step 1: Design in Figma

Create or update designs in the appropriate Figma project following [brand guidelines](../docs/marketing/FIGMA_INTEGRATION_GUIDE.md#brand-guidelines).

### Step 2: Export from Figma

#### Manual Export (via Figma UI)

1. Select frame or layer
2. In right panel, scroll to "Export" section
3. Click "+" to add export settings:
   - **SVG**: For logos, icons, diagrams
   - **PNG**: For screenshots, social media (1x or 2x scale)
4. Click "Export [Name]"
5. Save to `figma/exports/svg/` or `figma/exports/png/`

#### Batch Export (via Figma API)

For automated exports, use the Figma REST API:

```bash
# Set environment variables
export FIGMA_TOKEN="your_figma_token"
export FIGMA_FILE_KEY="your_file_key"

# Example: Get file metadata
curl -H "X-Figma-Token: $FIGMA_TOKEN" \
  "https://api.figma.com/v1/files/$FIGMA_FILE_KEY" | jq

# Example: Export specific nodes as PNG
curl -H "X-Figma-Token: $FIGMA_TOKEN" \
  "https://api.figma.com/v1/images/$FIGMA_FILE_KEY?ids=NODE_ID&format=png&scale=2"
```

See [Figma API Documentation](https://www.figma.com/developers/api) for complete reference.

### Step 3: Optimize Exports

Before moving to final location, optimize file sizes:

#### SVG Optimization

```bash
# Using SVGO
npm install -g svgo
svgo figma/exports/svg/*.svg --multipass

# Or using svgcleaner
svgcleaner input.svg output.svg
```

#### PNG Optimization

```bash
# Using ImageOptim (macOS)
imageoptim figma/exports/png/*.png

# Using pngquant (cross-platform)
pngquant --quality=85-95 --ext .png --force figma/exports/png/*.png

# Using TinyPNG API
curl --user api:YOUR_API_KEY \
  --data-binary @input.png \
  https://api.tinify.com/shrink > output.png
```

### Step 4: Move to Final Location

After optimization, move files to appropriate directories:

```bash
# Marketing assets (SVG)
mv figma/exports/svg/diagram_*.svg docs/assets/marketing/diagrams/

# Social media (PNG)
mv figma/exports/png/twitter_*.png docs/assets/social/twitter/

# Icons (SVG)
mv figma/exports/svg/icon_*.svg docs/assets/marketing/icons/
```

### Step 5: Commit to Repository

```bash
# Stage files
git add docs/assets/marketing/
git add docs/assets/social/

# Commit with conventional commit format
git commit -m "docs(assets): add marketing diagrams and social media cards

- Add architecture overview diagram (SVG)
- Add Twitter launch announcement card (PNG)
- Add feature icons set (48×48px SVG)

Exported from Figma QuASIM_QuNimbus_Marketing v1.2"

# Push to feature branch
git push origin marketing/figma-assets-v1
```

## Design System

### Components Library

Reusable components are maintained in `QuASIM_Design_System` Figma project:

**Component Categories**:

- Buttons (primary, secondary, text, icon)
- Cards (feature, metric, testimonial)
- Forms (inputs, dropdowns, checkboxes)
- Navigation (header, footer, sidebar)
- Icons (custom 24×24 and 48×48)
- Badges (certification, compliance)

### Using Components

To use design system components in new projects:

1. Open `QuASIM_Design_System` in Figma
2. Go to Assets panel (Shift + I)
3. Drag component into your canvas
4. Customize instance properties as needed

### Updating Components

When updating a component:

1. Edit in `QuASIM_Design_System`
2. Publish changes (right panel → Publish)
3. In consuming projects, accept updates (Assets panel → Update icon)

## Collaboration

### Roles & Permissions

| Role | Permissions | Who |
|------|-------------|-----|
| Owner | Full control, billing | Project lead |
| Editor | Edit, comment, view | Design team, marketing |
| Viewer | View, comment | Development, stakeholders |

### Communication

**In Figma**:

- Use comments (C key) for feedback
- Tag users with @ mentions
- Mark resolved when addressed

**In GitHub**:

- Open issues for design requests
- Use `design` label
- Link to Figma frames in issue description

### Design Reviews

Schedule regular design reviews:

- **Frequency**: Bi-weekly or before major releases
- **Attendees**: Design team, marketing, key stakeholders
- **Agenda**: Review new designs, discuss feedback, plan next sprint
- **Output**: Action items logged in GitHub issues

## Templates

### Available Templates

| Template | Purpose | Location in Figma |
|----------|---------|-------------------|
| Landing Page | Marketing website | 02_Marketing_Pages |
| Feature Card | Highlight features | QuASIM_Design_System |
| Social Media Post | Twitter/LinkedIn | 04_Social_Media |
| Pitch Slide | Investor deck | 03_Pitch_Deck |
| Diagram | Technical docs | 05_Diagrams |

### Using Templates

1. Navigate to template in Figma
2. Duplicate frame (Cmd/Ctrl + D)
3. Rename duplicate with descriptive name
4. Customize content
5. Export when ready

### Creating New Templates

To add a new template:

1. Design in appropriate Figma project
2. Test with sample content
3. Document usage in this README
4. Publish component (if reusable)
5. Notify team via GitHub issue or Slack

## Tools & Plugins

### Recommended Figma Plugins

- **Iconify**: Access icon libraries (Feather, Heroicons)
- **Unsplash**: Stock photos
- **Autoflow**: Create flow arrows for diagrams
- **Chart**: Generate data visualizations
- **Content Reel**: Populate with placeholder content
- **Stark**: Accessibility checker (contrast, colorblind simulation)

### Installing Plugins

1. Go to Figma menu → Plugins → Manage Plugins
2. Search for plugin name
3. Click "Install"
4. Access via Figma menu → Plugins → [Plugin Name]

## Best Practices

### Design Workflow

1. **Start with wireframes**: Low-fidelity sketches first
2. **Use components**: Leverage design system for consistency
3. **Mobile-first**: Design for mobile, then scale up
4. **Prototype**: Create interactive flows for testing
5. **Get feedback**: Share with team early and often

### Performance

- **Minimize layers**: Flatten when possible
- **Optimize images**: Compress before importing to Figma
- **Use components**: Reduces file size and improves performance
- **Archive old versions**: Keep project files lean

### Accessibility

- **Color contrast**: Use Stark plugin to check WCAG compliance
- **Text size**: Minimum 16px for body text
- **Touch targets**: Minimum 44×44px for interactive elements
- **Alt text**: Add descriptions for exported images

## Resources

### Figma Learning

- [Figma Official Tutorial](https://help.figma.com/hc/en-us/articles/360040314193)
- [Figma YouTube Channel](https://www.youtube.com/c/Figmadesign)
- [Figma Community](https://www.figma.com/community)

### Design Inspiration

- [Dribbble](https://dribbble.com/tags/quantum)
- [Behance](https://www.behance.net/search/projects?search=quantum%20computing)
- [Awwwards](https://www.awwwards.com/)

### QuASIM Documentation

- [Figma Integration Guide](../docs/marketing/FIGMA_INTEGRATION_GUIDE.md)
- [Marketing Assets README](../docs/assets/marketing/README.md)
- [Social Assets README](../docs/assets/social/README.md)

## Troubleshooting

### Common Issues

**Problem**: Exported SVG doesn't match Figma preview

- **Solution**: Ensure "Include 'id' attribute" is enabled in export settings

**Problem**: PNG file size too large

- **Solution**: Reduce scale (2x → 1x) or optimize with ImageOptim/TinyPNG

**Problem**: Fonts not rendering correctly in exports

- **Solution**: Outline text (Cmd/Ctrl + Shift + O) before exporting

**Problem**: Can't access Figma project

- **Solution**: Request access via GitHub issue with `access-request` label

## Questions?

For questions about Figma projects or design process:

- **Design Questions**: Open GitHub issue with `design` label
- **Access Issues**: Open GitHub issue with `access-request` label
- **Technical Issues**: Contact development team
- **Urgent Matters**: Tag @QuASIM-Team in Figma comments

---

**Last Updated**: November 12, 2025
**Figma Version**: Latest (web/desktop app)
**Maintainer**: QuASIM Design Team
