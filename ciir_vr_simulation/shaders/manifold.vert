/**
 * manifold.vert — CIIR Manifold Vertex Shader (GLSL 460)
 *
 * Morphs CIIR manifold vertices dynamically based on constraint intensity.
 * The constraint_value attribute drives deformation along the normal direction,
 * creating a visual representation of constraint satisfaction.
 *
 * Deformation model:
 *   position' = position + normal * constraint_value * morph_strength
 *
 * This maps the abstract CIIR constraint surface S_C = {ρ : C_i(ρ) ≈ 0}
 * into a physically deformed 3D mesh where:
 *   - Zero constraint → flat surface (satisfied)
 *   - High constraint → displaced surface (violated)
 */
#version 460

// ---- Vertex attributes ----
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec4 in_color;
layout(location = 3) in float in_constraint_value;

// ---- Push constants ----
layout(push_constant) uniform PushConstants {
    mat4  view_projection;    // Combined VP matrix
    mat4  model;              // Model transform
    float morph_strength;     // Constraint deformation amplitude
    float time;               // Animation time (seconds)
    float purity;             // Tr(ρ²) — controls opacity
    float entropy;            // S(ρ) — controls radial scale
} pc;

// ---- Outputs to fragment shader ----
layout(location = 0) out vec3 frag_position;
layout(location = 1) out vec3 frag_normal;
layout(location = 2) out vec4 frag_color;
layout(location = 3) out float frag_constraint;
layout(location = 4) out float frag_purity;

void main() {
    // ---- Constraint-driven morphing ----
    // Deform along normal proportional to constraint violation.
    // Includes a subtle time-based oscillation for visual appeal.
    float morph = in_constraint_value * pc.morph_strength;
    float wave  = sin(pc.time * 2.0 + in_position.x * 3.0 + in_position.z * 3.0) * 0.02;
    vec3 displaced = in_position + in_normal * (morph + wave);

    // ---- Entropy-based radial scaling ----
    // Higher entropy → more spread out (larger radius).
    // Entropy range: [0, log(D)], normalised to [0.8, 1.2].
    float entropy_scale = mix(0.8, 1.2, clamp(pc.entropy / 3.0, 0.0, 1.0));
    displaced *= entropy_scale;

    // ---- Transform ----
    vec4 world_pos = pc.model * vec4(displaced, 1.0);
    gl_Position    = pc.view_projection * world_pos;

    // ---- Pass to fragment shader ----
    frag_position   = world_pos.xyz;
    frag_normal     = normalize(mat3(pc.model) * in_normal);
    frag_color      = in_color;
    frag_constraint = in_constraint_value;
    frag_purity     = pc.purity;
}
