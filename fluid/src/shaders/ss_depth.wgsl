struct CameraUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    eye_pos: vec4<f32>,
    light_dir: vec4<f32>,
    viewport_size: vec2<f32>,
    near_far: vec2<f32>,
};

struct RenderParams {
    particle_radius: f32,
    render_mode: u32,
    color_theme: u32,
    max_velocity_color: f32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> render_params: RenderParams;

struct VertexInput {
    @location(0) quad_pos: vec2<f32>,
    @location(1) p_pos: vec3<f32>,
    @location(2) p_density: f32,
    @location(3) p_vel: vec3<f32>,
    @location(4) p_pressure: f32,
    @location(5) p_color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) view_pos: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.uv = in.quad_pos;

    // View-space billboard quad
    let p_view = camera.view * vec4<f32>(in.p_pos, 1.0);
    let r = render_params.particle_radius * 1.5;
    let quad_view = p_view.xyz + vec3<f32>(in.quad_pos * r, 0.0);
    out.view_pos = quad_view;
    out.clip_position = camera.proj * vec4<f32>(quad_view, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) f32 {
    let r2 = dot(in.uv, in.uv);
    if (r2 > 1.0) {
        discard;
    }

    let r = render_params.particle_radius * 1.5;
    let dz = sqrt(1.0 - r2) * r;
    // In view space RH, forward is -Z, so sphere front is view_pos.z + dz
    let view_z = in.view_pos.z + dz;
    return -view_z; // Positive linear depth
}
