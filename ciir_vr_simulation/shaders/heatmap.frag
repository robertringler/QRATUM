/**
 * heatmap.frag — Tensor Heatmap Fragment Shader (GLSL 460)
 *
 * Colors the heatmap surface using a perceptually uniform colormap.
 * The normalised scalar value drives the color gradient.
 *
 * Colormaps available:
 *   0 = Cool-Warm (diverging: blue → white → red)
 *   1 = Viridis (sequential: purple → green → yellow)
 *   2 = Inferno (sequential: black → red → yellow → white)
 *   3 = Plasma  (sequential: purple → orange → yellow)
 */
#version 460

// ---- Inputs ----
layout(location = 0) in vec3 frag_position;
layout(location = 1) in vec3 frag_normal;
layout(location = 2) in vec4 frag_color;
layout(location = 3) in float frag_value_normalised;

// ---- Output ----
layout(location = 0) out vec4 out_color;

// ---- Push constants ----
layout(push_constant) uniform PushConstants {
    mat4  view_projection;
    mat4  model;
    float height_scale;
    float time;
    float value_min;
    float value_max;
} pc;

// ---- Colormaps ----

vec3 inferno(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.001, 0.000, 0.014);
    vec3 c1 = vec3(0.416, 0.090, 0.432);
    vec3 c2 = vec3(0.878, 0.345, 0.200);
    vec3 c3 = vec3(0.988, 0.998, 0.645);
    if (t < 0.33) return mix(c0, c1, t * 3.0);
    if (t < 0.66) return mix(c1, c2, (t - 0.33) * 3.0);
    return mix(c2, c3, (t - 0.66) * 3.0);
}

vec3 plasma(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.050, 0.030, 0.528);
    vec3 c1 = vec3(0.494, 0.012, 0.658);
    vec3 c2 = vec3(0.940, 0.392, 0.140);
    vec3 c3 = vec3(0.940, 0.975, 0.131);
    if (t < 0.33) return mix(c0, c1, t * 3.0);
    if (t < 0.66) return mix(c1, c2, (t - 0.33) * 3.0);
    return mix(c2, c3, (t - 0.66) * 3.0);
}

void main() {
    float t = frag_value_normalised;

    // Use inferno colormap for loss/gradient heatmaps.
    vec3 heatmap_color = inferno(t);

    // ---- Lighting ----
    vec3 light_dir = normalize(vec3(0.3, 1.0, 0.5));
    vec3 normal    = normalize(frag_normal);
    float diffuse  = max(dot(normal, light_dir), 0.0) * 0.6;
    float ambient  = 0.25;

    vec3 lit = heatmap_color * (ambient + diffuse);

    // ---- Contour lines ----
    // Draw subtle contour lines at regular intervals of the scalar value.
    float contour_freq = 10.0; // 10 contour levels
    float contour = abs(fract(t * contour_freq) - 0.5) * 2.0;
    contour = smoothstep(0.0, 0.05, contour);
    lit *= mix(0.7, 1.0, contour);

    // ---- Grid lines (subtle) ----
    float grid_x = abs(fract(frag_position.x * 2.0) - 0.5) * 2.0;
    float grid_z = abs(fract(frag_position.z * 2.0) - 0.5) * 2.0;
    float grid = min(grid_x, grid_z);
    grid = smoothstep(0.0, 0.03, grid);
    lit *= mix(0.85, 1.0, grid);

    out_color = vec4(lit, 0.95);
}
