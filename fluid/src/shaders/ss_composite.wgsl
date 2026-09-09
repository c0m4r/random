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

struct LiquidMaterial {
    liquid_color: vec4<f32>,    // rgb: absorption color, a: density/thickness scale
    refraction_ior: f32,        // 1.333 for water, 1.5 for honey
    roughness: f32,
    fresnel_power: f32,
    specular_intensity: f32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> material: LiquidMaterial;

@group(1) @binding(0) var depth_tex: texture_2d<f32>;
@group(1) @binding(1) var scene_tex: texture_2d<f32>;
@group(1) @binding(2) var default_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vertex_index << 1u) & 2) * 2.0 - 1.0;
    let y = f32(i32(vertex_index & 2u) * -2) + 1.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

fn uv_to_view_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc_x = uv.x * 2.0 - 1.0;
    let ndc_y = (1.0 - uv.y) * 2.0 - 1.0;
    // Unproject to view space
    let inv_proj_00 = 1.0 / camera.proj[0][0];
    let inv_proj_11 = 1.0 / camera.proj[1][1];
    let vx = ndc_x * depth * inv_proj_00;
    let vy = ndc_y * depth * inv_proj_11;
    let vz = -depth;
    return vec3<f32>(vx, vy, vz);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let depth = textureSample(depth_tex, default_sampler, in.uv).r;
    let bg_color = textureSample(scene_tex, default_sampler, in.uv);

    if (depth <= 0.001) {
        return bg_color;
    }

    let tex_dim = vec2<f32>(textureDimensions(depth_tex));
    let texel = 1.0 / tex_dim;

    // Sample neighbors to reconstruct normal
    let d_l = textureSample(depth_tex, default_sampler, in.uv - vec2<f32>(texel.x, 0.0)).r;
    let d_r = textureSample(depth_tex, default_sampler, in.uv + vec2<f32>(texel.x, 0.0)).r;
    let d_d = textureSample(depth_tex, default_sampler, in.uv - vec2<f32>(0.0, texel.y)).r;
    let d_u = textureSample(depth_tex, default_sampler, in.uv + vec2<f32>(0.0, texel.y)).r;

    let p_center = uv_to_view_pos(in.uv, depth);
    let p_l = uv_to_view_pos(in.uv - vec2<f32>(texel.x, 0.0), select(depth, d_l, d_l > 0.001));
    let p_r = uv_to_view_pos(in.uv + vec2<f32>(texel.x, 0.0), select(depth, d_r, d_r > 0.001));
    let p_d = uv_to_view_pos(in.uv - vec2<f32>(0.0, texel.y), select(depth, d_d, d_d > 0.001));
    let p_u = uv_to_view_pos(in.uv + vec2<f32>(0.0, texel.y), select(depth, d_u, d_u > 0.001));

    let dx = p_r - p_l;
    let dy = p_u - p_d;
    var normal_view = normalize(cross(dx, dy));
    if (normal_view.z < 0.0) {
        normal_view = -normal_view;
    }

    // View-space lighting
    let view_dir = vec3<f32>(0.0, 0.0, 1.0);
    let light_dir_view = normalize((camera.view * vec4<f32>(camera.light_dir.xyz, 0.0)).xyz);

    // Fresnel reflectance
    let n_dot_v = max(dot(normal_view, view_dir), 0.0);
    let f0 = pow((1.0 - material.refraction_ior) / (1.0 + material.refraction_ior), 2.0);
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - n_dot_v, material.fresnel_power);

    // Refraction UV offset
    let refract_offset = normal_view.xy * (0.035 / max(depth, 1.0));
    let refract_uv = clamp(in.uv + refract_offset, vec2<f32>(0.0), vec2<f32>(1.0));
    let refract_color = textureSample(scene_tex, default_sampler, refract_uv).rgb;

    // Beer-Lambert absorption
    let extinction = (vec3<f32>(1.0) - material.liquid_color.rgb) * 1.5;
    let absorption = exp(-extinction * (depth * 0.12));
    let water_body = refract_color * absorption * material.liquid_color.rgb;

    // Specular Highlight
    let half_vec = normalize(light_dir_view + view_dir);
    let n_dot_h = max(dot(normal_view, half_vec), 0.0);
    let specular = vec3<f32>(1.0) * pow(n_dot_h, 48.0) * material.specular_intensity;

    // Sky Reflection
    let sky_color = mix(vec3<f32>(0.5, 0.7, 0.9), vec3<f32>(0.9, 0.95, 1.0), normal_view.y * 0.5 + 0.5);

    let final_rgb = mix(water_body, sky_color, fresnel * 0.7) + specular;
    return vec4<f32>(final_rgb, 1.0);
}
