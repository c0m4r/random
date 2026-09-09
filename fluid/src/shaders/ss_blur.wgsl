@group(0) @binding(0) var depth_tex: texture_2d<f32>;
@group(0) @binding(1) var depth_sampler: sampler;

struct BlurUniforms {
    blur_dir: vec2<f32>,
    filter_radius: i32,
    depth_threshold: f32,
};

@group(0) @binding(2) var<uniform> blur_params: BlurUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    // Fullscreen triangle
    let x = f32(i32(vertex_index << 1u) & 2) * 2.0 - 1.0;
    let y = f32(i32(vertex_index & 2u) * -2) + 1.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) f32 {
    let center_depth = textureSample(depth_tex, depth_sampler, in.uv).r;
    if (center_depth <= 0.001) {
        return 0.0;
    }

    var total_depth: f32 = center_depth;
    var total_weight: f32 = 1.0;

    let tex_dim = vec2<f32>(textureDimensions(depth_tex));
    let texel_size = 1.0 / tex_dim;
    let r = blur_params.filter_radius;

    for (var i = -r; i <= r; i = i + 1) {
        if (i == 0) { continue; }

        let offset = blur_params.blur_dir * (f32(i) * texel_size);
        let sample_uv = in.uv + offset;
        let d = textureSample(depth_tex, depth_sampler, sample_uv).r;

        if (d > 0.001) {
            let spatial_dist = f32(i);
            let spatial_w = exp(-0.5 * (spatial_dist * spatial_dist) / (f32(r * r) * 0.5));

            let depth_diff = abs(d - center_depth);
            let range_w = exp(-0.5 * (depth_diff * depth_diff) / (blur_params.depth_threshold * blur_params.depth_threshold));

            let weight = spatial_w * range_w;
            total_depth = total_depth + d * weight;
            total_weight = total_weight + weight;
        }
    }

    return total_depth / total_weight;
}
