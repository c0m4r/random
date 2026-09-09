struct ObstacleGpu {
    pos: vec3<f32>,
    radius: f32,
    vel: vec3<f32>,
    obstacle_type: u32,
    params: vec4<f32>,
};

struct SimParams {
    gravity: vec4<f32>,
    boundary_min: vec4<f32>,
    boundary_max: vec4<f32>,
    mouse_pos: vec4<f32>,
    particle_count: u32,
    rest_density: f32,
    stiffness: f32,
    viscosity: f32,
    surface_tension: f32,
    smoothing_radius: f32,
    particle_radius: f32,
    damping: f32,
    dt: f32,
    mouse_radius: f32,
    mouse_strength: f32,
    pad0: f32,
    obstacles: array<ObstacleGpu, 4>,
};

struct Particle {
    pos: vec3<f32>,
    density: f32,
    vel: vec3<f32>,
    pressure: f32,
    color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> grid_heads: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> grid_links: array<i32>;

const PI: f32 = 3.141592653589793;
const GRID_MIN: vec3<f32> = vec3<f32>(-1.1, -0.1, -0.7);
const GRID_DIM_X: u32 = 48u;
const GRID_DIM_Y: u32 = 36u;
const GRID_DIM_Z: u32 = 32u;
const TOTAL_CELLS: u32 = 55296u; // 48 * 36 * 32

fn get_cell_coord(pos: vec3<f32>) -> vec3<i32> {
    let rel = pos - GRID_MIN;
    let h = params.smoothing_radius;
    let cx = i32(rel.x / h);
    let cy = i32(rel.y / h);
    let cz = i32(rel.z / h);
    return vec3<i32>(cx, cy, cz);
}

fn is_valid_cell(c: vec3<i32>) -> bool {
    return c.x >= 0 && c.x < i32(GRID_DIM_X) &&
           c.y >= 0 && c.y < i32(GRID_DIM_Y) &&
           c.z >= 0 && c.z < i32(GRID_DIM_Z);
}

fn cell_to_index(c: vec3<i32>) -> u32 {
    let ux = u32(c.x);
    let uy = u32(c.y);
    let uz = u32(c.z);
    return ux + uy * GRID_DIM_X + uz * (GRID_DIM_X * GRID_DIM_Y);
}

// Pass 0: Clear Grid Heads
@compute @workgroup_size(64)
fn cs_clear_grid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < TOTAL_CELLS) {
        atomicStore(&grid_heads[idx], -1);
    }
}

// Pass 1: Build Grid (Spatial Hash Linked List)
@compute @workgroup_size(64)
fn cs_build_grid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.particle_count) {
        return;
    }

    let c = get_cell_coord(particles[idx].pos);
    if (is_valid_cell(c)) {
        let cell_idx = cell_to_index(c);
        let prev = atomicExchange(&grid_heads[cell_idx], i32(idx));
        grid_links[idx] = prev;
    } else {
        grid_links[idx] = -1;
    }
}

// Pass 2: Density & Pressure Calculation
@compute @workgroup_size(64)
fn cs_density(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.particle_count) {
        return;
    }

    let p_i = particles[index].pos;
    let h = params.smoothing_radius;
    let h2 = h * h;
    let mass = 0.025;
    let poly6 = 315.0 / (64.0 * PI * pow(h, 9.0));

    var density: f32 = 0.0;
    let cell = get_cell_coord(p_i);

    // Loop over 27 neighboring cells
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let n_cell = cell + vec3<i32>(dx, dy, dz);
                if (is_valid_cell(n_cell)) {
                    let c_idx = cell_to_index(n_cell);
                    var j = atomicLoad(&grid_heads[c_idx]);
                    var limit: u32 = 0u;

                    while (j >= 0 && limit < 128u) {
                        let p_j = particles[j].pos;
                        let diff = p_i - p_j;
                        let r2 = dot(diff, diff);
                        if (r2 < h2) {
                            let diff2 = h2 - r2;
                            density = density + mass * poly6 * (diff2 * diff2 * diff2);
                        }
                        j = grid_links[j];
                        limit = limit + 1u;
                    }
                }
            }
        }
    }

    let rho = max(density, params.rest_density);
    particles[index].density = rho;

    let ratio = rho / params.rest_density;
    let pressure = params.stiffness * (pow(ratio, 7.0) - 1.0);
    particles[index].pressure = max(pressure, 0.0);
}

// Pass 3: Forces, Collisions, Integration
@compute @workgroup_size(64)
fn cs_forces(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.particle_count) {
        return;
    }

    let p_i = particles[index].pos;
    let v_i = particles[index].vel;
    let rho_i = particles[index].density;
    let pres_i = particles[index].pressure;

    let h = params.smoothing_radius;
    let mass = 0.025;
    let spiky_grad = -45.0 / (PI * pow(h, 6.0));
    let visc_lap = 45.0 / (PI * pow(h, 6.0));

    var f_pressure = vec3<f32>(0.0);
    var f_viscosity = vec3<f32>(0.0);
    var f_surface = vec3<f32>(0.0);

    let cell = get_cell_coord(p_i);

    // Loop over 27 neighboring cells
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let n_cell = cell + vec3<i32>(dx, dy, dz);
                if (is_valid_cell(n_cell)) {
                    let c_idx = cell_to_index(n_cell);
                    var j = atomicLoad(&grid_heads[c_idx]);
                    var limit: u32 = 0u;

                    while (j >= 0 && limit < 128u) {
                        let uj = u32(j);
                        if (uj != index) {
                            let p_j = particles[uj].pos;
                            let diff = p_i - p_j;
                            let r = length(diff);

                            if (r > 0.0001 && r < h) {
                                let dir = diff / r;
                                let rho_j = particles[uj].density;
                                let pres_j = particles[uj].pressure;
                                let v_j = particles[uj].vel;

                                let p_avg = (pres_i + pres_j) * 0.5;
                                let hr = h - r;
                                let grad_w = spiky_grad * (hr * hr) * dir;
                                f_pressure = f_pressure - mass * (p_avg / rho_j) * grad_w;

                                let lap_w = visc_lap * hr;
                                f_viscosity = f_viscosity + params.viscosity * mass * ((v_j - v_i) / rho_j) * lap_w;

                                f_surface = f_surface - params.surface_tension * mass * (diff * hr / rho_j);
                            }
                        }
                        j = grid_links[j];
                        limit = limit + 1u;
                    }
                }
            }
        }
    }

    // Mouse interactive force
    var f_mouse = vec3<f32>(0.0);
    if (params.mouse_pos.w > 0.5) {
        let m_diff = p_i - params.mouse_pos.xyz;
        let m_dist = length(m_diff);
        if (m_dist < params.mouse_radius && m_dist > 0.001) {
            let m_dir = m_diff / m_dist;
            let falloff = 1.0 - (m_dist / params.mouse_radius);
            f_mouse = m_dir * (params.mouse_strength * falloff);
        }
    }

    let acc = (f_pressure + f_viscosity + f_surface) / rho_i + params.gravity.xyz + f_mouse;

    var vel = v_i + acc * params.dt;
    var pos = p_i + vel * params.dt;

    let p_radius = params.particle_radius;
    let b_min = params.boundary_min.xyz;
    let b_max = params.boundary_max.xyz;
    let damping = params.damping;

    // Boundary Collisions (Glass Tank)
    if (pos.x - p_radius < b_min.x) {
        pos.x = b_min.x + p_radius;
        vel.x = -vel.x * damping;
    } else if (pos.x + p_radius > b_max.x) {
        pos.x = b_max.x - p_radius;
        vel.x = -vel.x * damping;
    }

    if (pos.y - p_radius < b_min.y) {
        pos.y = b_min.y + p_radius;
        vel.y = -vel.y * damping;
        vel.x = vel.x * 0.95;
        vel.z = vel.z * 0.95;
    } else if (pos.y + p_radius > b_max.y) {
        pos.y = b_max.y - p_radius;
        vel.y = -vel.y * damping;
    }

    if (pos.z - p_radius < b_min.z) {
        pos.z = b_min.z + p_radius;
        vel.z = -vel.z * damping;
    } else if (pos.z + p_radius > b_max.z) {
        pos.z = b_max.z - p_radius;
        vel.z = -vel.z * damping;
    }

    // Dynamic Obstacle Collisions
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let obs = params.obstacles[k];
        if (obs.obstacle_type == 1u) {
            let o_diff = pos - obs.pos;
            let o_dist = length(o_diff);
            let min_d = obs.radius + p_radius;
            if (o_dist < min_d && o_dist > 0.0001) {
                let n = o_diff / o_dist;
                pos = obs.pos + n * min_d;
                let rel_v = vel - obs.vel;
                let vn = dot(rel_v, n);
                if (vn < 0.0) {
                    vel = obs.vel + (rel_v - (1.0 + damping) * vn * n);
                }
            }
        } else if (obs.obstacle_type == 2u) {
            let o_diff = pos - obs.pos;
            let r_dist = length(vec2<f32>(o_diff.x, o_diff.z));
            let height = obs.params.z;
            let rotor_radius = obs.radius;
            let angle = obs.params.x;
            let ang_vel = obs.params.y;

            if (r_dist < rotor_radius && abs(o_diff.y) < height * 0.5) {
                let p_ang = atan2(o_diff.z, o_diff.x);
                let num_blades: f32 = 4.0;
                let step: f32 = 2.0 * PI / num_blades;

                for (var b: f32 = 0.0; b < num_blades; b = b + 1.0) {
                    let b_ang = angle + b * step;
                    var d_ang = (p_ang - b_ang) % (2.0 * PI);
                    if (d_ang > PI) { d_ang = d_ang - 2.0 * PI; }
                    if (d_ang < -PI) { d_ang = d_ang + 2.0 * PI; }

                    let arc = r_dist * abs(d_ang);
                    let blade_thick = 0.035;
                    if (arc < (blade_thick + p_radius)) {
                        let sign_d = select(-1.0, 1.0, d_ang > 0.0);
                        let n_ang = b_ang + sign_d * (PI * 0.5);
                        let norm = vec3<f32>(cos(n_ang), 0.0, sin(n_ang));
                        let tan_v = ang_vel * r_dist;
                        let blade_v = vec3<f32>(-sin(b_ang), 0.0, cos(b_ang)) * tan_v;

                        pos = pos + norm * ((blade_thick + p_radius) - arc);
                        vel = blade_v + norm * 2.0;
                        break;
                    }
                }
            }
        }
    }

    particles[index].pos = pos;
    particles[index].vel = vel;
}
