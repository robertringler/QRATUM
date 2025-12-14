// Stress field visualization shader
#version 330 core

// Vertex shader
#ifdef VERTEX_SHADER

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in float stress;

out vec3 fragPosition;
out vec3 fragNormal;
out float fragStress;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    vec4 worldPos = model * vec4(position, 1.0);
    fragPosition = worldPos.xyz;
    fragNormal = mat3(transpose(inverse(model))) * normal;
    fragStress = stress;
    
    gl_Position = projection * view * worldPos;
}

#endif

// Fragment shader
#ifdef FRAGMENT_SHADER

in vec3 fragPosition;
in vec3 fragNormal;
in float fragStress;

out vec4 FragColor;

uniform float stressMin;
uniform float stressMax;
uniform vec3 cameraPosition;
uniform vec3 lightDirection;

// Viridis-like colormap for stress
vec3 stressColor(float s) {
    // Clamp and normalize stress
    float normalized = clamp((s - stressMin) / (stressMax - stressMin), 0.0, 1.0);
    
    vec3 color;
    if (normalized < 0.2) {
        // Dark purple to purple
        float f = normalized / 0.2;
        color = mix(vec3(0.267, 0.004, 0.329), vec3(0.282, 0.141, 0.458), f);
    } else if (normalized < 0.4) {
        // Purple to blue-purple
        float f = (normalized - 0.2) / 0.2;
        color = mix(vec3(0.282, 0.141, 0.458), vec3(0.253, 0.265, 0.530), f);
    } else if (normalized < 0.6) {
        // Blue-purple to teal
        float f = (normalized - 0.4) / 0.2;
        color = mix(vec3(0.253, 0.265, 0.530), vec3(0.163, 0.471, 0.558), f);
    } else if (normalized < 0.8) {
        // Teal to green
        float f = (normalized - 0.6) / 0.2;
        color = mix(vec3(0.163, 0.471, 0.558), vec3(0.369, 0.788, 0.383), f);
    } else {
        // Green to yellow
        float f = (normalized - 0.8) / 0.2;
        color = mix(vec3(0.369, 0.788, 0.383), vec3(0.993, 0.906, 0.144), f);
    }
    
    return color;
}

void main() {
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(cameraPosition - fragPosition);
    vec3 L = normalize(-lightDirection);
    
    // Basic diffuse lighting
    float diffuse = max(dot(N, L), 0.0);
    float ambient = 0.3;
    float lighting = ambient + diffuse * 0.7;
    
    // Get stress color
    vec3 color = stressColor(fragStress);
    
    // Apply lighting
    color *= lighting;
    
    // Gamma correction
    color = pow(color, vec3(1.0/2.2));
    
    FragColor = vec4(color, 1.0);
}

#endif
