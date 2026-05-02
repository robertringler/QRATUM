/**
 * manifold.frag — CIIR Manifold Fragment Shader (GLSL 460)
 *
 * Colors fragments based on constraint intensity and observer state.
 *
 * Color mapping:
 *   constraint_value → cool-warm colormap (blue = satisfied, red = violated)
 *   purity → opacity (pure states more opaque, mixed states transparent)
 *
 * Lighting: Blinn-Phong with ambient/diffuse/specular.
 */
#version 460

// ---- Inputs from vertex shader ----
layout(location = 0) in vec3 frag_position;
layout(location = 1) in vec3 frag_normal;
layout(location = 2) in vec4 frag_color;
layout(location = 3) in float frag_constraint;
layout(location = 4) in float frag_purity;

// ---- Output ----
layout(location = 0) out vec4 out_color;

// ---- Uniforms ----
layout(push_constant) uniform PushConstants {
    mat4  view_projection;
    mat4  model;
    float morph_strength;
    float time;
    float purity;
    float entropy;
} pc;

// ---- Cool-warm colormap ----
vec3 cool_warm(float t) {
    // t ∈ [0, 1]: 0 = deep blue (satisfied), 1 = deep red (violated)
    t = clamp(t, 0.0, 1.0);
    vec3 cool = vec3(0.1, 0.3, 0.8);   // Blue
    vec3 warm = vec3(0.9, 0.2, 0.1);   // Red
    vec3 mid  = vec3(0.95, 0.95, 0.95); // White at midpoint
    if (t < 0.5) {
        return mix(cool, mid, t * 2.0);
    } else {
        return mix(mid, warm, (t - 0.5) * 2.0);
    }
}

// ---- Viridis-like colormap (for gradient magnitude) ----
vec3 viridis(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.267, 0.004, 0.329);
    vec3 c1 = vec3(0.282, 0.140, 0.458);
    vec3 c2 = vec3(0.127, 0.566, 0.551);
    vec3 c3 = vec3(0.993, 0.906, 0.144);
    if (t < 0.33) return mix(c0, c1, t * 3.0);
    if (t < 0.66) return mix(c1, c2, (t - 0.33) * 3.0);
    return mix(c2, c3, (t - 0.66) * 3.0);
}

void main() {
    // ---- Normalise constraint to [0, 1] for colormap ----
    // Constraint violations squared, so sqrt for perceptual linearity.
    float cv = sqrt(abs(frag_constraint));
    cv = clamp(cv * 2.0, 0.0, 1.0); // Scale for visibility

    // ---- Color: constraint heatmap blended with base color ----
    vec3 constraint_color = cool_warm(cv);
    vec3 base_color = mix(frag_color.rgb, constraint_color, 0.7);

    // ---- Lighting: Blinn-Phong ----
    vec3 light_dir = normalize(vec3(0.5, 1.0, 0.3));
    vec3 view_dir  = normalize(-frag_position);
    vec3 half_dir  = normalize(light_dir + view_dir);
    vec3 normal    = normalize(frag_normal);

    float ambient  = 0.15;
    float diffuse  = max(dot(normal, light_dir), 0.0) * 0.7;
    float specular = pow(max(dot(normal, half_dir), 0.0), 32.0) * 0.3;

    vec3 lit_color = base_color * (ambient + diffuse) + vec3(specular);

    // ---- Opacity: driven by purity Tr(ρ²) ----
    // Pure states (purity → 1) are fully opaque.
    // Maximally mixed (purity → 1/D) are semi-transparent.
    float alpha = mix(0.3, 1.0, clamp(frag_purity, 0.0, 1.0));

    // ---- Constraint edge glow ----
    // Highlight the constraint surface boundary (cv ≈ 0.5).
    float edge = exp(-50.0 * (cv - 0.5) * (cv - 0.5));
    lit_color += vec3(0.2, 0.8, 0.4) * edge * 0.5;

    out_color = vec4(lit_color, alpha * frag_color.a);
}
