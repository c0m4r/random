#version 450
// room2 - UI vertex stage. Positions arrive in pixels with the origin at the top left.
#include "common.glsl"

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec2 inUv;
layout(location = 2) in vec4 inColor;

layout(push_constant) uniform UiPush {
    vec2 screenSize;
    float useTexture;
    float opacity;
} pc;

layout(location = 0) out vec2 vUv;
layout(location = 1) out vec4 vColor;

void main() {
    vUv = inUv;
    vColor = inColor * pc.opacity;
    // Pixel space has its origin at the top-left; Vulkan's NDC has y = -1 at the top of
    // the viewport, so this is a plain scale-and-bias with no extra flip.
    vec2 ndc = vec2(inPosition.x / pc.screenSize.x * 2.0 - 1.0,
                    inPosition.y / pc.screenSize.y * 2.0 - 1.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
}
