#version 450
// room2 - UI fragment stage. The font atlas is a single-channel coverage texture.
#include "common.glsl"

layout(location = 0) in vec2 vUv;
layout(location = 1) in vec4 vColor;

layout(location = 0) out vec4 outColor;

layout(set = 2, binding = 0) uniform sampler2D uAtlas;

layout(push_constant) uniform UiPush {
    vec2 screenSize;
    float useTexture;
    float opacity;
} pc;

void main() {
    vec4 c = vColor;
    if (pc.useTexture > 0.5) {
        float coverage = texture(uAtlas, vUv).r;
        c.a *= coverage;
    }
    if (c.a <= 0.001) discard;
    outColor = c;
}
