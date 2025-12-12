// Thermal field visualization shader
#version 330 core

// Vertex shader
#ifdef VERTEX_SHADER

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in float temperature;

out vec3 fragPosition;
out vec3 fragNormal;
out float fragTemperature;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    vec4 worldPos = model * vec4(position, 1.0);
    fragPosition = worldPos.xyz;
    fragNormal = mat3(transpose(inverse(model))) * normal;
    fragTemperature = temperature;
    
    gl_Position = projection * view * worldPos;
}

#endif

// Fragment shader
#ifdef FRAGMENT_SHADER

in vec3 fragPosition;
in vec3 fragNormal;
in float fragTemperature;

out vec4 FragColor;

uniform float tempMin;
uniform float tempMax;
uniform vec3 cameraPosition;
uniform vec3 lightDirection;

// Heat colormap (blue -> cyan -> green -> yellow -> red)
vec3 heatColor(float t) {
    // Clamp and normalize temperature
    float normalized = clamp((t - tempMin) / (tempMax - tempMin), 0.0, 1.0);
    
    vec3 color;
    if (normalized < 0.25) {
        // Blue to Cyan
        float f = normalized / 0.25;
        color = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), f);
    } else if (normalized < 0.5) {
        // Cyan to Green
        float f = (normalized - 0.25) / 0.25;
        color = mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), f);
    } else if (normalized < 0.75) {
        // Green to Yellow
        float f = (normalized - 0.5) / 0.25;
        color = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), f);
    } else {
        // Yellow to Red
        float f = (normalized - 0.75) / 0.25;
        color = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), f);
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
    
    // Get heat color
    vec3 color = heatColor(fragTemperature);
    
    // Apply lighting
    color *= lighting;
    
    // Gamma correction
    color = pow(color, vec3(1.0/2.2));
    
    FragColor = vec4(color, 1.0);
}

#endif
