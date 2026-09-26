#version 450
// room2 - G-buffer fragment stage. Outputs a physically-based material description
// into four render targets (deferred shading).
#include "common.glsl"

layout(location = 0) in vec3 vWorldPos;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec4 vTangent;
layout(location = 3) in vec2 vUv;
layout(location = 4) in vec2 vUv1;
layout(location = 5) flat in uint vMaterialIndex;
layout(location = 6) in float vViewDepth;

layout(location = 0) out vec4 outAlbedoAo;    // rgb = albedo, a = ambient occlusion
layout(location = 1) out vec4 outNormalRough; // rgb = world normal, a = roughness
layout(location = 2) out vec4 outMetalKind;   // r = metallic, g = 1, b = kind, a = 1
layout(location = 3) out vec4 outEmissive;    // rgb = emissive radiance, a = linear view depth

void main() {
    GpuMaterial m = materials[vMaterialIndex];

    vec4 baseSample = texture(uTextures[nonuniformEXT(m.texIndices.x)], vUv);
    vec4 base = m.baseColorFactor * baseSample;

    uvec4 ormSample = uvec4(0u);
    vec4 orm = vec4(1.0, m.params0.y, m.params0.x, 1.0);
    if (m.texIndices.z != 0xFFFFFFFFu) {
        orm = texture(uTextures[nonuniformEXT(m.texIndices.z)], vUv);
    }
    float metallic = clamp(orm.b * m.params0.x, 0.0, 1.0);
    float roughness = clamp(orm.g * m.params0.y, 0.015, 1.0);
    float ao = mix(1.0, orm.r, clamp(m.params0.w, 0.0, 1.0));

    vec3 N = normalize(vNormal);
    if (dot(N, N) < 1e-8) N = vec3(0.0, 1.0, 0.0);
    vec3 T = vTangent.xyz;
    T = T - N * dot(N, T);
    float tlen = length(T);
    if (tlen < 1e-5) {
        // Degenerate tangents: build an arbitrary orthonormal basis.
        vec3 up = abs(N.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        T = normalize(cross(up, N));
    } else {
        T /= tlen;
    }
    vec3 B = cross(N, T) * (vTangent.w < 0.0 ? -1.0 : 1.0);

    if (m.texIndices.y != 0xFFFFFFFFu) {
        vec4 nrmSample = texture(uTextures[nonuniformEXT(m.texIndices.y)], vUv);
        N = r2_applyNormalMap(nrmSample, N, T, B, m.params0.z);
    }

    vec3 emissive = m.emissiveFactor.rgb;
    if (m.texIndices.w != 0xFFFFFFFFu) {
        emissive *= texture(uTextures[nonuniformEXT(m.texIndices.w)], vUv).rgb;
    }

    outAlbedoAo = vec4(base.rgb, ao);
    outNormalRough = vec4(N * 0.5 + 0.5, roughness);
    outMetalKind = vec4(metallic, m.params1.x, m.params1.w / 255.0, 1.0);
    // The alpha channel carries the linear view depth so the deferred passes can
    // rebuild world positions without inverting a projection matrix.
    outEmissive = vec4(emissive, vViewDepth);
}
