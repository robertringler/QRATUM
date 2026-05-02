/**
 * heatmap.vert — Tensor Heatmap Vertex Shader (GLSL 460)
 *
 * Displaces a flat grid into a 3D surface based on tensor values.
 * Used for visualizing:
 *   - Loss landscape L(ρ) as a function of two perturbation directions
 *   - Gradient magnitude fields ‖∇L‖
 *   - Constraint violation maps Σ C_i(ρ)²
 *
 * The height of each vertex is driven by the constraint_value attribute
 * (repurposed here as "scalar field value").
 */
#version 460

// ---- Vertex attributes ----
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec4 in_color;
layout(location = 3) in float in_constraint_value;

// ---- Push constants ----
layout(push_constant) uniform PushConstants {
    mat4  view_projection;
    mat4  model;
    float height_scale;       // Vertical scale for the scalar field
    float time;
    float value_min;          // Minimum scalar value (for normalisation)
    float value_max;          // Maximum scalar value
} pc;

// ---- Outputs ----
layout(location = 0) out vec3 frag_position;
layout(location = 1) out vec3 frag_normal;
layout(location = 2) out vec4 frag_color;
layout(location = 3) out float frag_value_normalised;

void main() {
    // ---- Height displacement from scalar field ----
    float range = max(pc.value_max - pc.value_min, 1e-6);
    float normalised = (in_constraint_value - pc.value_min) / range;
    float height = normalised * pc.height_scale;

    vec3 displaced = in_position;
    displaced.y = height;

    // ---- Transform ----
    vec4 world_pos = pc.model * vec4(displaced, 1.0);
    gl_Position    = pc.view_projection * world_pos;

    // ---- Pass to fragment ----
    frag_position        = world_pos.xyz;
    frag_normal          = normalize(mat3(pc.model) * in_normal);
    frag_color           = in_color;
    frag_value_normalised = normalised;
}
