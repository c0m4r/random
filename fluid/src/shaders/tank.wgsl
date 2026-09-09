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

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.world_pos = in.position;
    out.normal = in.normal;
    out.color = in.color;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(camera.light_dir.xyz);
    let view_dir = normalize(camera.eye_pos.xyz - in.world_pos);
    let n = normalize(in.normal);

    // Diffuse
    let n_dot_l = max(dot(n, light_dir), 0.0);
    let diffuse = in.color.rgb * (0.35 + 0.65 * n_dot_l);

    // Specular
    let half_vec = normalize(light_dir + view_dir);
    let n_dot_h = max(dot(n, half_vec), 0.0);
    let specular = vec3<f32>(1.0) * pow(n_dot_h, 32.0) * 0.4;

    // Floor procedural grid lines if normal is pointing up
    var final_color = diffuse + specular;
    if (abs(n.y) > 0.9) {
        let grid_size = 0.2;
        let coord = in.world_pos.xz / grid_size;
        let grid = abs(fract(coord - 0.5) - 0.5) / fwidth(coord);
        let line = min(grid.x, grid.y);
        let grid_factor = 1.0 - min(line, 1.0);
        
        let checker = (floor(coord.x) + floor(coord.y)) % 2.0;
        let base_floor = select(vec3<f32>(0.12, 0.13, 0.16), vec3<f32>(0.16, 0.18, 0.22), checker == 0.0);
        final_color = mix(base_floor, vec3<f32>(0.28, 0.32, 0.40), grid_factor * 0.7);

        // Circular distance rings from origin
        let dist = length(in.world_pos.xz);
        let ring = abs(fract(dist / 0.5 - 0.5) - 0.5) / fwidth(dist / 0.5);
        let ring_factor = 1.0 - min(ring, 1.0);
        final_color = mix(final_color, vec3<f32>(0.35, 0.45, 0.65), ring_factor * 0.3);

        // Floor contact shadow under tank
        if (abs(in.world_pos.x) < 1.05 && abs(in.world_pos.z) < 0.65) {
            final_color = final_color * 0.65;
        }
    }

    return vec4<f32>(final_color, in.color.a);
}
