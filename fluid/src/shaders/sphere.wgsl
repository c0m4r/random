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
    render_mode: u32,       // 0: Shaded, 1: Velocity Heatmap, 2: Pressure Heatmap, 3: Glass
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
    @location(1) center_world: vec3<f32>,
    @location(2) speed: f32,
    @location(3) density: f32,
    @location(4) pressure: f32,
    @location(5) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.uv = in.quad_pos;
    out.center_world = in.p_pos;
    out.speed = length(in.p_vel);
    out.density = in.p_density;
    out.pressure = in.p_pressure;
    out.color = in.p_color;

    // Camera billboard orientation
    let right = vec3<f32>(camera.view[0][0], camera.view[1][0], camera.view[2][0]);
    let up = vec3<f32>(camera.view[0][1], camera.view[1][1], camera.view[2][1]);

    let world_pos = in.p_pos + (right * in.quad_pos.x + up * in.quad_pos.y) * render_params.particle_radius;
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    return out;
}

fn turbo_colormap(t: f32) -> vec3<f32> {
    // High-contrast physics scientific colormap
    let x = clamp(t, 0.0, 1.0);
    let r = clamp(1.5 - abs(x * 4.0 - 3.0), 0.0, 1.0);
    let g = clamp(1.5 - abs(x * 4.0 - 2.0), 0.0, 1.0);
    let b = clamp(1.5 - abs(x * 4.0 - 1.0), 0.0, 1.0);
    return vec3<f32>(r, g, b);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let r2 = dot(in.uv, in.uv);
    if (r2 > 1.0) {
        discard;
    }

    let nz = sqrt(1.0 - r2);
    let right = vec3<f32>(camera.view[0][0], camera.view[1][0], camera.view[2][0]);
    let up = vec3<f32>(camera.view[0][1], camera.view[1][1], camera.view[2][1]);
    let forward = -vec3<f32>(camera.view[0][2], camera.view[1][2], camera.view[2][2]);

    let normal = normalize(right * in.uv.x + up * in.uv.y + forward * nz);

    let light_dir = normalize(camera.light_dir.xyz);
    let view_dir = normalize(camera.eye_pos.xyz - in.center_world);

    // Color computation
    var base_color: vec3<f32>;
    if (render_params.render_mode == 1u) {
        // Velocity heatmap
        let t = clamp(in.speed / render_params.max_velocity_color, 0.0, 1.0);
        base_color = turbo_colormap(t);
    } else if (render_params.render_mode == 2u) {
        // Pressure heatmap
        let t = clamp(in.pressure / 800.0, 0.0, 1.0);
        base_color = mix(vec3<f32>(0.05, 0.1, 0.4), vec3<f32>(1.0, 0.1, 0.8), t);
        if (t > 0.7) {
            base_color = mix(base_color, vec3<f32>(1.0, 1.0, 1.0), (t - 0.7) / 0.3);
        }
    } else {
        base_color = in.color.rgb;
    }

    // Shading: Diffuse + Blinn-Phong Specular + Fresnel
    let n_dot_l = max(dot(normal, light_dir), 0.0);
    let half_vec = normalize(light_dir + view_dir);
    let n_dot_h = max(dot(normal, half_vec), 0.0);
    let specular = pow(n_dot_h, 48.0) * 0.6;

    let fresnel = pow(1.0 - max(dot(normal, view_dir), 0.0), 3.0) * 0.35;
    let ambient = 0.30;

    let shaded = base_color * (ambient + 0.70 * n_dot_l) + vec3<f32>(1.0) * (specular + fresnel);
    return vec4<f32>(shaded, 1.0);
}
