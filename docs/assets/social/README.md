# Social Media Assets

This directory contains visual assets optimized for social media platforms, including Twitter/X, LinkedIn, Instagram, and other social networks.

## Directory Structure

```
social/
├── README.md           # This file
├── twitter/            # Twitter/X cards and headers (PNG)
├── linkedin/           # LinkedIn posts and banners (PNG)
├── instagram/          # Instagram posts and stories (PNG)
├── facebook/           # Facebook posts and covers (PNG)
└── youtube/            # YouTube thumbnails and banners (PNG)
```

## Social Media Specifications

### Twitter/X

| Asset Type | Dimensions | Format | Notes |
|------------|------------|--------|-------|
| Card (Summary) | 1200×628px | PNG | Min 2:1 ratio |
| Card (Large) | 1200×675px | PNG | 16:9 ratio |
| Header | 1500×500px | PNG | Safe zone: center 1280×427px |
| Profile Image | 400×400px | PNG | Displayed at 200×200px |

**File Naming**: `twitter_{type}_{description}_v{version}.png`

Examples:

- `twitter_card_quantum_simulation_launch_v1.png`
- `twitter_header_quasim_qunimbus_v2.png`

### LinkedIn

| Asset Type | Dimensions | Format | Notes |
|------------|------------|--------|-------|
| Post | 1200×627px | PNG | Recommended |
| Article Header | 1200×627px | PNG | Same as post |
| Company Banner | 1128×191px | PNG | Desktop display |
| Logo | 300×300px | PNG | Min 60×60px |

**File Naming**: `linkedin_{type}_{description}_v{version}.png`

Examples:

- `linkedin_post_certification_announcement_v1.png`
- `linkedin_banner_enterprise_quantum_v1.png`

### Instagram

| Asset Type | Dimensions | Format | Notes |
|------------|------------|--------|-------|
| Square Post | 1080×1080px | PNG | Standard format |
| Portrait Post | 1080×1350px | PNG | 4:5 ratio |
| Story | 1080×1920px | PNG | 9:16 ratio |
| Profile Image | 320×320px | PNG | Min 110×110px |

**File Naming**: `instagram_{type}_{description}_v{version}.png`

Examples:

- `instagram_square_spacex_validation_v1.png`
- `instagram_story_feature_highlight_v2.png`

### Facebook

| Asset Type | Dimensions | Format | Notes |
|------------|------------|--------|-------|
| Post | 1200×630px | PNG | Link preview |
| Cover Photo | 820×312px | PNG | Desktop display |
| Profile Image | 170×170px | PNG | Min 170×170px |
| Event Cover | 1920×1080px | PNG | 16:9 ratio |

**File Naming**: `facebook_{type}_{description}_v{version}.png`

### YouTube

| Asset Type | Dimensions | Format | Notes |
|------------|------------|--------|-------|
| Thumbnail | 1280×720px | PNG | Max 2MB |
| Banner | 2560×1440px | PNG | Safe zone: 1546×423px |
| Profile Image | 800×800px | PNG | Displayed at various sizes |

**File Naming**: `youtube_{type}_{description}_v{version}.png`

## Design Guidelines

### Brand Consistency

All social media assets must follow QuASIM brand guidelines:

- **Color Palette**: Use primary brand colors (#002b36, #073642, #268bd2)
- **Typography**: Inter (primary), Roboto Mono (technical content)
- **Logo**: Include QuASIM logo in safe zone
- **Style**: Dark theme, minimalist, professional

### Text Overlay

- **Headline**: 32-48px, Inter Bold, max 2 lines
- **Body Text**: 18-24px, Inter Regular, max 3 lines
- **Call-to-Action**: 20-28px, Inter SemiBold
- **Contrast**: Ensure WCAG AA compliance (4.5:1 ratio minimum)

### Safe Zones

Always keep critical content within safe zones:

- **Twitter Header**: Center 1280×427px
- **LinkedIn Banner**: Center area (avoid edges)
- **Instagram Story**: Center vertical strip, avoid top 250px and bottom 250px
- **YouTube Banner**: Center 1546×423px

## Content Categories

### Announcement Posts

Purpose: Product launches, feature releases, certifications

Elements:

- Eye-catching headline
- Key benefit or metric
- QuASIM logo
- Call-to-action or link

### Educational Posts

Purpose: Technical explainers, how-to content

Elements:

- Diagram or illustration
- Concise explanation
- Code snippet or example (if applicable)
- Learn more link

### Testimonial/Case Study Posts

Purpose: Customer success, validation results

Elements:

- Quote or metric
- Company logo (SpaceX, NASA, etc.)
- Validation data
- Link to full case study

### Event Promotion

Purpose: Webinars, conferences, meetups

Elements:

- Event details (date, time)
- Speaker or topic
- Registration link
- Branded background

## Export Settings

### From Figma

1. Select frame for export
2. Set format to PNG
3. Set scale to 2x (for high-DPI displays)
4. Export and downsize to 1x if file size is too large
5. Optimize with ImageOptim or TinyPNG

### Optimization

- **Target File Size**: < 1MB for all social posts
- **Tools**: ImageOptim, TinyPNG, Squoosh
- **Quality**: 85-90% JPEG quality (if converting from PNG)
- **Metadata**: Remove EXIF data

## Usage Workflow

### 1. Create in Figma

Use templates from `QuASIM_QuNimbus_Marketing` Figma project:

- Navigate to `04_Social_Media` page
- Duplicate template for desired platform
- Customize content
- Follow brand guidelines

### 2. Export

Export at appropriate dimensions for each platform (see specs above).

### 3. Optimize

Run through optimization tools to reduce file size while maintaining quality.

### 4. Add to Repository

```bash
# Add to appropriate subdirectory
cp exported_image.png docs/assets/social/twitter/twitter_card_new_feature_v1.png

# Commit
git add docs/assets/social/twitter/twitter_card_new_feature_v1.png
git commit -m "docs(assets): add Twitter card for new feature announcement"
git push
```

### 5. Upload to Platform

Use exported asset for social media post or update.

### 6. Track Performance

Document performance metrics in a separate tracking spreadsheet (not in repo).

## Content Schedule

Maintain a content calendar for social media posts:

### Weekly Posts (Minimum)

- **Monday**: Educational content (tip, tutorial, or explainer)
- **Wednesday**: Product update or feature highlight
- **Friday**: Community spotlight or industry news

### Monthly Posts

- Achievement announcements (certifications, milestones)
- Case study or customer success story
- Behind-the-scenes content
- Event promotions

## Asset Inventory

### Current Assets

| Asset | Platform | Type | Description | Status |
|-------|----------|------|-------------|--------|
| (Placeholder) | - | - | - | - |

### Planned Assets

| Asset | Platform | Priority | Deadline | Status |
|-------|----------|----------|----------|--------|
| Launch Announcement | All | High | TBD | Pending |
| DO-178C Certification | Twitter, LinkedIn | High | TBD | Pending |
| SpaceX Validation | All | Medium | TBD | Pending |
| Feature Grid | Instagram | Low | TBD | Pending |

## Best Practices

### General

1. **Test Across Devices**: Preview on mobile and desktop
2. **A/B Testing**: Create variants to test messaging
3. **Consistency**: Maintain visual style across platforms
4. **Timing**: Post during optimal engagement windows
5. **Engagement**: Include questions or calls-to-action

### Platform-Specific

#### Twitter/X

- Keep text concise (280 character limit)
- Use 1-2 hashtags (#QuASIM #QuantumComputing)
- Include link to GitHub or docs
- Tag relevant accounts (@NASA, @SpaceX if applicable)

#### LinkedIn

- Professional tone, longer-form content acceptable
- Use industry-relevant hashtags
- Tag company pages and collaborators
- Include metrics and credibility indicators

#### Instagram

- Visual-first, minimal text
- Use stories for behind-the-scenes content
- Leverage hashtags (up to 30)
- Engage with comments

## Hashtag Strategy

### Primary Hashtags

- #QuASIM
- #QuNimbus
- #QuantumSimulation
- #MarketIntelligence

### Industry Hashtags

- #QuantumComputing
- #AerospaceEngineering
- #DefenseTechnology
- #EnterpriseAI
- #HPC (High Performance Computing)

### Campaign-Specific Hashtags

Create unique hashtags for campaigns:

- #QuASIMCertified (for DO-178C announcement)
- #QuantumValidated (for SpaceX/NASA results)
- #QuNimbusLaunch (for economic activation features)

## Compliance & Legal

### Content Review

All social media assets must be reviewed before posting:

- **Technical Accuracy**: Verify claims and metrics
- **ITAR Compliance**: No export-controlled information
- **Trademark Usage**: Follow guidelines for third-party logos
- **Copyright**: Only use licensed or created content

### Attribution

When featuring customer results or testimonials:

- Obtain written permission
- Include proper attribution
- Link to official sources when possible

## Related Documentation

- [Figma Integration Guide](../marketing/FIGMA_INTEGRATION_GUIDE.md)
- [Brand Guidelines](../marketing/FIGMA_INTEGRATION_GUIDE.md#brand-guidelines)
- [Marketing Assets README](../marketing/README.md)
- [Main README](../../README.md)

## Questions?

For questions about social media assets or content strategy:

- Open a GitHub issue with label `marketing` or `design`
- Contact the marketing team
- Reference this README in discussions

---

**Last Updated**: November 12, 2025
