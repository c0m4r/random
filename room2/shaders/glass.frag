#version 450
// room2 - physically based glass. Screen-space refraction of the opaque pass, an
// environment reflection with Fresnel weighting, Beer-Lambert absorption through the
// authored wall thickness, and a shadowed specular highlight from the room light.
#include "common.glsl"

layout(location = 0) in vec3 vWorldPos;
layout(location = 1) in vec3 vNormal;
layout(location = 2) in vec4 vTangent;
layout(location = 3) in vec2 vUv;
layout(location = 4) in vec4 vClipPos;
layout(location = 5) flat in uint vMaterialIndex;

layout(location = 0) out vec4 outColor;

layout(set = 2, binding = 0) uniform sampler2D uOpaqueColor;   // HDR scene without glass
layout(set = 2, binding = 1) uniform sampler2D uDepth;         // opaque depth
layout(set = 2, binding = 2) uniform samplerCube uEnvCube;
layout(set = 2, binding = 3) uniform sampler2DArray uShadowMaps;

float evaluateShadowMap(vec3 worldPos, vec3 N, uint layer, vec3 lightPos) {
    vec4 clip = lights[0].shadowViewProj[0] * vec4(worldPos, 1.0);
    if (clip.w <= 1e-5) return 1.0;
    vec3 ndc = clip.xyz / clip.w;
    vec2 uv = ndc.xy * 0.5 + 0.5;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return 1.0;
    if (ndc.z <= 0.0 || ndc.z >= 1.0) return 1.0;
    vec3 toFrag = worldPos - lightPos;
    float dist = length(toFrag);
    vec3 dir = dist > 1e-5 ? toFrag / dist : vec3(0.0, -1.0, 0.0);
    float slope = clamp(1.0 - dot(N, -dir), 0.0, 1.0);
    float bias = g.shadowParams.w * (1.0 + 2.0 * slope) * (1.0 + dist * 0.35);
    float refDepth = ndc.z - bias;
    vec2 texel = g.shadowParams.zz * 2.4;
    float sum = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            float d = texture(uShadowMaps, vec3(uv + vec2(x, y) * texel, float(layer))).r;
            sum += d < refDepth ? 0.0 : 1.0;
        }
    }
    return sum / 9.0;
}

void main() {
    GpuMaterial m = materials[vMaterialIndex];

    vec3 N = normalize(vNormal);
    vec3 T = normalize(vTangent.xyz - N * dot(N, vTangent.xyz) + vec3(1e-6));
    vec3 B = cross(N, T) * (vTangent.w < 0.0 ? -1.0 : 1.0);
    if (m.texIndices.y != 0xFFFFFFFFu) {
        vec4 nrmSample = texture(uTextures[nonuniformEXT(m.texIndices.y)], vUv);
        N = r2_applyNormalMap(nrmSample, N, T, B, m.params0.z);
    }

    vec3 V = normalize(g.cameraPosition.xyz - vWorldPos);
    bool frontFace = dot(N, V) >= 0.0;
    if (!frontFace) N = -N;

    vec2 screenUv = (vClipPos.xy / vClipPos.w) * 0.5 + 0.5;
    float sceneDepth = texture(uDepth, screenUv).r;

    float transmission = clamp(m.params1.x, 0.0, 1.0);
    float ior = max(m.params1.y, 1.0);
    float thickness = max(m.attenuationColor.a, 0.0);
    float roughness = clamp(m.params0.y, 0.01, 1.0);
    vec3 tint = m.attenuationColor.rgb;

    float NdotV = clamp(dot(N, V), 1e-4, 1.0);
    float F0 = pow((ior - 1.0) / (ior + 1.0), 2.0);
    float fresnel = F0 + (1.0 - F0) * pow(1.0 - NdotV, 5.0);

    // ---- refraction ---------------------------------------------------------
    vec3 refracted = refract(-V, N, 1.0 / ior);
    // Project the refracted ray onto the screen; scale by the glass thickness so the
    // distortion stays physically motivated rather than arbitrary.
    vec3 refractedWorld = vWorldPos + refracted * max(thickness, 0.004) * 3.0;
    vec4 refrClip = g.viewProjection * vec4(refractedWorld, 1.0);
    vec2 refrUv = (refrClip.xy / refrClip.w) * 0.5 + 0.5;
    refrUv = clamp(refrUv, vec2(0.001), vec2(0.999));
    vec3 refrColor = texture(uOpaqueColor, refrUv).rgb;

    // Beer-Lambert absorption along the traversed thickness.
    float pathLength = max(thickness, 0.002) * (1.0 / max(NdotV, 0.25));
    vec3 absorbance = (1.0 - tint) * 0.6;
    vec3 transmittance = exp(-absorbance * pathLength * 40.0);
    refrColor *= mix(vec3(1.0), transmittance, transmission);
    // Slight blur for rough glass.
    if (roughness > 0.12) {
        vec2 blurStep = vec2(g.renderSize.z, g.renderSize.w) * roughness * 6.0;
        refrColor = refrColor * 0.5
                  + texture(uOpaqueColor, refrUv + blurStep).rgb * 0.1667
                  + texture(uOpaqueColor, refrUv - blurStep).rgb * 0.1667
                  + texture(uOpaqueColor, refrUv + vec2(blurStep.x, -blurStep.y)).rgb * 0.0834
                  + texture(uOpaqueColor, refrUv - vec2(blurStep.x, -blurStep.y)).rgb * 0.0834;
    }

    // ---- reflection ---------------------------------------------------------
    vec3 R = reflect(-V, N);
    float lod = sqrt(roughness) * 5.0;
    vec3 envRefl = textureLod(uEnvCube, R, lod).rgb;

    // Screen-space reflection attempt: if the reflected ray hits on-screen geometry
    // that is in front of the glass, blend it in. Falls back to the probe.
    vec3 reflWorld = vWorldPos + R * max(thickness, 0.01) * 8.0;
    vec4 reflClip = g.viewProjection * vec4(reflWorld, 1.0);
    vec2 reflUv = (reflClip.xy / reflClip.w) * 0.5 + 0.5;
    vec3 reflection = envRefl;
    if (reflUv.x > 0.0 && reflUv.x < 1.0 && reflUv.y > 0.0 && reflUv.y < 1.0) {
        float reflDepth = texture(uDepth, reflUv).r;
        float reflClipZ = reflClip.z / reflClip.w;
        if (reflDepth < 1.0 && reflDepth < reflClipZ) {
            // The reflected sample is closer than the reflected ray endpoint: visible.
            vec3 ssr = texture(uOpaqueColor, reflUv).rgb;
            reflection = mix(envRefl, ssr, 0.55 * (1.0 - roughness));
        }
    }

    // ---- specular highlight from the room light ------------------------------
    vec3 highlight = vec3(0.0);
    uint lightCount = uint(g.misc.z + 0.5);
    for (uint i = 0u; i < lightCount && i < 4u; ++i) {
        GpuLight light = lights[i];
        vec3 toLight = light.positionRange.xyz - vWorldPos;
        float dist = length(toLight);
        if (dist > light.positionRange.w) continue;
        vec3 L = toLight / max(dist, 1e-5);
        float NdotL = dot(N, L);
        if (NdotL <= 0.0) continue;
        vec3 H = normalize(L + V);
        float NdotH = clamp(dot(N, H), 0.0, 1.0);
        float D = r2_distributionGGX(NdotH, max(roughness, 0.02));
        float G = r2_geometrySmith(NdotV, clamp(NdotL, 0.0, 1.0), max(roughness, 0.04));
        float VdotH = clamp(dot(V, H), 0.0, 1.0);
        float F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
        float shadow = 1.0;
        if (light.shadowParams.w > 0.0) {
            shadow = evaluateShadowMap(vWorldPos, N, 0u, light.positionRange.xyz);
        }
        vec3 radiance = light.colorIntensity.rgb * light.colorIntensity.a
                      / max(dist * dist, 1e-4) * shadow;
        highlight += radiance * (D * G * F) / max(4.0 * NdotV * clamp(NdotL, 1e-4, 1.0), 1e-4);
    }

    // ---- combine ------------------------------------------------------------
    vec3 color = mix(refrColor, reflection, clamp(fresnel, 0.02, 0.98)) + highlight;

    // A faint rim brightening reads as a polished edge.
    float rim = pow(1.0 - NdotV, 3.0);
    color += vec3(0.02) * rim * (1.0 - roughness);

    // The glass is thin, so the "alpha" is the fraction of background we replace.
    float alpha = clamp(mix(0.10, 0.85, fresnel) * transmission + (1.0 - transmission) * m.params1.z,
                        0.06, 0.96);
    outColor = vec4(color, alpha);
}
