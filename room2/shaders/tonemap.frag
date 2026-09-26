#version 450
// room2 - final composite: exposure, bloom, ACES tonemapping, colour grading,
// chromatic aberration, vignette and film grain.
#include "common.glsl"

layout(location = 0) in vec2 vUv;
layout(location = 0) out vec4 outColor;

layout(set = 2, binding = 0) uniform sampler2D uHdrColor;
layout(set = 2, binding = 1) uniform sampler2D uBloom;

void main() {
    vec2 uv = vUv;
    vec2 centered = uv - 0.5;
    float r2 = dot(centered, centered);

    // Subtle chromatic aberration that grows towards the frame edges.
    float aberration = 0.0016 * r2;
    vec3 color;
    color.r = texture(uHdrColor, uv + centered * aberration).r;
    color.g = texture(uHdrColor, uv).g;
    color.b = texture(uHdrColor, uv - centered * aberration).b;

    vec3 bloom = texture(uBloom, uv).rgb;
    color += bloom * g.postParams.y;

    // Exposure (already adapted on the CPU) then filmic tonemap.
    color *= g.misc.w;
    color = r2_acesFitted(max(color, vec3(0.0)));

    // ---- colour grade: slight cool shadows / warm highlights, filmic contrast ----
    float l = r2_luma(color);
    vec3 shadowTint = vec3(0.96, 0.99, 1.06);
    vec3 highlightTint = vec3(1.04, 1.00, 0.96);
    vec3 tint = mix(shadowTint, highlightTint, smoothstep(0.15, 0.75, l));
    color *= tint;

    // Gentle S-curve contrast around mid grey.
    color = clamp(color, 0.0, 1.0);
    color = mix(color, color * color * (3.0 - 2.0 * color), 0.18);

    // Saturation.
    float grey = r2_luma(color);
    color = mix(vec3(grey), color, 1.06);

    // Vignette.
    float vignette = 1.0 - 0.32 * smoothstep(0.15, 0.95, r2 * 1.8);
    color *= vignette;

    // Film grain, scaled down in bright areas (like real film).
    float grain = r2_hash12(gl_FragCoord.xy + g.misc.y * 71.3) - 0.5;
    color += grain * 0.016 * (1.0 - smoothstep(0.0, 0.9, r2_luma(color)));

    outColor = vec4(clamp(color, 0.0, 1.0), 1.0);
#ifdef R2_DEBUG_CONSTANT_OUT
    outColor = vec4(0.5, 0.2, 0.1, 1.0);
#endif
#ifdef R2_DEBUG_RAW_INPUT
    outColor = vec4(clamp(texture(uHdrColor, uv).rgb * 8.0, 0.0, 1.0), 1.0);
#endif
#ifdef R2_DEBUG_EXPOSURE
    outColor = vec4(vec3(g.misc.w * 0.25), 1.0);
#endif
#ifdef R2_DEBUG_ACES_ONLY
    outColor = vec4(r2_acesFitted(max(texture(uHdrColor, uv).rgb, vec3(0.0))), 1.0);
#endif
}
