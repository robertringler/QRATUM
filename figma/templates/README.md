# Figma Templates

This directory contains reusable Figma templates for creating consistent visual assets for QuASIM × QuNimbus.

## Available Templates

### 1. Landing Page Template

**Purpose**: Marketing website landing pages

**Includes**:
- Hero section (1920×1080px)
- Feature grid (2×2 or 3×2 layout)
- Metrics/stats section
- Call-to-action section
- Footer

**Usage**: Duplicate frame, customize content, maintain brand colors and typography.

**Location**: QuASIM_QuNimbus_Marketing → 02_Marketing_Pages → Landing_Page_Template

---

### 2. Social Media Post Templates

#### Twitter/X Card
- **Dimensions**: 1200×675px (16:9)
- **Elements**: Headline, body text, logo, background
- **Location**: 04_Social_Media → Twitter_Card_Template

#### LinkedIn Post
- **Dimensions**: 1200×627px
- **Elements**: Headline, body text, metric/stat, logo
- **Location**: 04_Social_Media → LinkedIn_Post_Template

#### Instagram Square
- **Dimensions**: 1080×1080px
- **Elements**: Visual-first with minimal text, logo
- **Location**: 04_Social_Media → Instagram_Square_Template

#### Instagram Story
- **Dimensions**: 1080×1920px
- **Elements**: Vertical format, safe zones marked
- **Location**: 04_Social_Media → Instagram_Story_Template

---

### 3. Pitch Deck Slide Templates

**Standard Slides**:
- Title slide (logo + tagline)
- Content slide (title + bullet points)
- Two-column slide (split content)
- Full-image slide (background image + overlay text)
- Closing slide (contact info + CTA)

**Location**: QuASIM_Pitch_Deck → Templates

---

### 4. Diagram Templates

#### Architecture Diagram
- **Purpose**: System architecture visualization
- **Elements**: Component boxes, connection lines, labels
- **Style**: Clean, minimal, color-coded components

#### Flow Diagram
- **Purpose**: Process flows, data flows
- **Elements**: Start/end nodes, process boxes, decision diamonds, arrows
- **Style**: Professional, easy to follow

**Location**: 05_Diagrams → Diagram_Templates

---

### 5. Feature Card Template

**Purpose**: Highlighting product features on website or in presentations

**Includes**:
- Icon placeholder (64×64px)
- Title (24px Inter SemiBold)
- Description (16px Inter Regular)
- Optional: Learn more link

**Variants**: Light background, dark background

**Location**: QuASIM_Design_System → Components → Cards → Feature_Card

---

### 6. Certification Badge Template

**Purpose**: Displaying compliance and certification credentials

**Includes**:
- Badge shape (rounded rectangle or shield)
- Certification name
- Level/tier indicator
- Color coding by certification type

**Available Badges**:
- DO-178C Level A (Orange)
- CMMC 2.0 Level 2 (Green)
- NIST 800-53 Rev 5 (Blue)
- Custom badge template

**Location**: QuASIM_Design_System → Components → Badges

---

## Using Templates

### Step 1: Access Template

1. Open appropriate Figma project
2. Navigate to template location (see above)
3. Select template frame

### Step 2: Duplicate

- **Keyboard**: Cmd/Ctrl + D
- **Menu**: Right-click → Duplicate
- **Drag**: Hold Alt/Option and drag

### Step 3: Customize

1. Rename duplicated frame with descriptive name
2. Update text content
3. Replace placeholder images
4. Adjust colors if needed (stay within brand palette)
5. Maintain spacing and layout structure

### Step 4: Export

1. Select customized frame
2. Add export settings (right panel)
   - SVG for logos, icons, diagrams
   - PNG for screenshots, social media
3. Export to `figma/exports/`
4. Optimize and move to final location

---

## Creating New Templates

To create a new template for the team:

### 1. Design Template

- Start with existing templates as reference
- Follow [brand guidelines](../../docs/marketing/BRAND_GUIDELINES.md)
- Use components from design system when possible
- Make reusable elements into components

### 2. Generalize Content

- Use placeholder text (Lorem ipsum or generic headlines)
- Use placeholder images (can use Unsplash plugin)
- Name all layers clearly
- Group related elements

### 3. Document Usage

- Add notes/comments in Figma explaining usage
- Document in this README
- Include example use case
- Specify dimensions and export settings

### 4. Share with Team

- Publish to appropriate project page
- Notify team via GitHub issue or Slack
- Add to template inventory (below)

---

## Template Inventory

| Template Name | Type | Dimensions | Location | Status |
|---------------|------|------------|----------|--------|
| Landing Page Hero | Web | 1920×1080px | 02_Marketing_Pages | Planned |
| Twitter Card | Social | 1200×675px | 04_Social_Media | Planned |
| LinkedIn Post | Social | 1200×627px | 04_Social_Media | Planned |
| Instagram Square | Social | 1080×1080px | 04_Social_Media | Planned |
| Pitch Title Slide | Deck | 1920×1080px | 03_Pitch_Deck | Planned |
| Architecture Diagram | Diagram | Variable | 05_Diagrams | Planned |
| Feature Card | Component | 320×240px | Design System | Planned |
| Certification Badge | Component | 120×40px | Design System | Planned |

---

## Template Best Practices

### Design Principles

1. **Consistency**: Use design system components
2. **Flexibility**: Make easy to customize without breaking layout
3. **Clarity**: Clear hierarchy and labeling
4. **Efficiency**: Minimize layers, use auto-layout
5. **Documentation**: Add instructions as comments

### Naming Conventions

**Template Frames**: `Template_{Type}_{Variant}`
- Example: `Template_Social_Twitter_Card`

**Template Components**: `Component/{Category}/{Name}`
- Example: `Component/Cards/Feature_Card`

### Components vs Frames

- **Use Components** for reusable UI elements (buttons, cards, icons)
- **Use Frames** for page layouts and one-off designs

### Auto Layout

Enable auto-layout on containers for responsive behavior:
- Shift + A (keyboard shortcut)
- Allows content to reflow when text changes
- Essential for templates that will be customized

---

## Maintenance

### Review Schedule

- **Frequency**: Quarterly
- **Activities**:
  - Review usage and feedback
  - Update to match brand guideline changes
  - Archive unused templates
  - Create new templates based on needs

### Template Updates

When a template needs updates:
1. Create new version (don't modify original directly)
2. Test with sample content
3. Document changes
4. Notify team of new version
5. Archive old version if replaced

### Feedback

To suggest template improvements:
- Add comments in Figma
- Open GitHub issue with `design` or `template` label
- Discuss in design review meetings

---

## Resources

- [Figma Integration Guide](../../docs/marketing/FIGMA_INTEGRATION_GUIDE.md)
- [Brand Guidelines](../../docs/marketing/BRAND_GUIDELINES.md)
- [Figma Auto-Layout Guide](https://help.figma.com/hc/en-us/articles/360040451373)
- [Figma Components Guide](https://help.figma.com/hc/en-us/articles/360038662654)

---

**Last Updated**: November 12, 2025
**Maintained By**: QuASIM Design Team
