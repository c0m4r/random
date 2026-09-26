#version 450
// room2 - shadow map vertex stage. Renders the scene from a cube-shadow face.
#include "common.glsl"

layout(location = 0) in vec3 inPosition;

layout(push_constant) uniform ShadowPush {
    uint faceIndex;   // cube face 0..5
    uint lightIndex;  // index into the light buffer
} pc;

void main() {
    GpuInstance inst = instances[gl_InstanceIndex];
    vec4 worldPos = inst.model * vec4(inPosition, 1.0);
    gl_Position = lights[pc.lightIndex].shadowViewProj[0] * worldPos;
}
