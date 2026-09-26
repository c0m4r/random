#version 450
// room2 - transparent (glass) vertex stage. Same geometry path as the G-buffer but
// additionally exports the view-space data needed for refraction.
#include "common.glsl"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inTangent;
layout(location = 3) in vec2 inUv;
layout(location = 4) in vec2 inUv1;

layout(location = 0) out vec3 vWorldPos;
layout(location = 1) out vec3 vNormal;
layout(location = 2) out vec4 vTangent;
layout(location = 3) out vec2 vUv;
layout(location = 4) out vec4 vClipPos;
layout(location = 5) flat out uint vMaterialIndex;

layout(push_constant) uniform DrawPush {
    uint materialIndex;
    uint flags;
} pc;

void main() {
    GpuInstance inst = instances[gl_InstanceIndex];
    vec4 worldPos = inst.model * vec4(inPosition, 1.0);
    mat3 nrm = mat3(inst.normalMatrix);
    vWorldPos = worldPos.xyz;
    vNormal = normalize(nrm * inNormal);
    vTangent = vec4(normalize(nrm * inTangent.xyz), inTangent.w);
    vUv = inUv;
    vMaterialIndex = pc.materialIndex;
    vec4 clip = g.viewProjection * worldPos;
    vClipPos = clip;
    gl_Position = clip;
}
