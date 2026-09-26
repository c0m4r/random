#version 450
// room2 - fullscreen triangle. Emits a single oversized triangle with UVs that can
// be used to index the whole render target.
#include "common.glsl"

layout(location = 0) out vec2 vUv;

void main() {
    vec2 pos = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    vUv = pos;
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
