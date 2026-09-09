use glam::Vec3;
use rayon::prelude::*;
use std::time::Instant;

use crate::physics::{
    particle::{ParticleCpu, ParticleGpu},
    obstacle::Obstacle,
    scenarios::{create_scenario, ScenarioType},
    PhysicsEngine, SimParams, StepMetrics, EngineType,
};

pub struct CpuSph {
    pub particles: Vec<ParticleCpu>,
    pub gpu_particles_cache: Vec<ParticleGpu>,
    cell_size: f32,
    grid_min: Vec3,
    grid_dim: [usize; 3],
    head: Vec<i32>,
    next: Vec<i32>,
}

impl CpuSph {
    pub fn new(scenario: ScenarioType, target_particles: usize) -> (Self, SimParams, Vec<Obstacle>) {
        let mut params = SimParams::default();
        let config = create_scenario(scenario, target_particles);
        params.viscosity = config.viscosity;
        params.surface_tension = config.surface_tension;
        params.gravity = config.gravity;
        params.stiffness = config.stiffness;
        params.rest_density = config.rest_density;

        let mut engine = Self {
            particles: config.particles,
            gpu_particles_cache: Vec::new(),
            cell_size: params.smoothing_radius,
            grid_min: Vec3::new(-1.1, -0.1, -0.7),
            grid_dim: [48, 36, 32],
            head: Vec::new(),
            next: Vec::new(),
        };
        engine.init_grid(params.smoothing_radius);
        engine.sync_gpu_cache();
        (engine, params, config.obstacles)
    }

    fn init_grid(&mut self, h: f32) {
        self.cell_size = h;
        let grid_max = Vec3::new(1.1, 1.6, 0.7);
        let extent = grid_max - self.grid_min;
        self.grid_dim = [
            (extent.x / h).ceil() as usize + 1,
            (extent.y / h).ceil() as usize + 1,
            (extent.z / h).ceil() as usize + 1,
        ];
        let total_cells = self.grid_dim[0] * self.grid_dim[1] * self.grid_dim[2];
        self.head = vec![-1; total_cells];
        self.next = vec![-1; self.particles.len()];
    }

    fn cell_coord(&self, pos: Vec3) -> Option<[usize; 3]> {
        let rel = pos - self.grid_min;
        if rel.x < 0.0 || rel.y < 0.0 || rel.z < 0.0 {
            return None;
        }
        let cx = (rel.x / self.cell_size) as usize;
        let cy = (rel.y / self.cell_size) as usize;
        let cz = (rel.z / self.cell_size) as usize;
        if cx < self.grid_dim[0] && cy < self.grid_dim[1] && cz < self.grid_dim[2] {
            Some([cx, cy, cz])
        } else {
            None
        }
    }

    fn cell_index(&self, c: [usize; 3]) -> usize {
        c[0] + c[1] * self.grid_dim[0] + c[2] * (self.grid_dim[0] * self.grid_dim[1])
    }

    fn build_grid(&mut self) {
        self.head.fill(-1);
        if self.next.len() != self.particles.len() {
            self.next.resize(self.particles.len(), -1);
        }

        for (i, p) in self.particles.iter().enumerate() {
            if let Some(c) = self.cell_coord(p.position) {
                let idx = self.cell_index(c);
                self.next[i] = self.head[idx];
                self.head[idx] = i as i32;
            }
        }
    }

    pub fn sync_gpu_cache(&mut self) {
        self.gpu_particles_cache.clear();
        self.gpu_particles_cache.reserve(self.particles.len());
        for p in &self.particles {
            self.gpu_particles_cache.push(p.to_gpu());
        }
    }
}

impl PhysicsEngine for CpuSph {
    fn step(&mut self, dt: f32, params: &SimParams, obstacles: &[Obstacle]) -> StepMetrics {
        let start = Instant::now();
        let substeps = params.substeps.max(1);
        let sub_dt = dt / substeps as f32;

        let h = params.smoothing_radius;
        let h2 = h * h;
        let mass = 0.025; // Particle mass
        let pi = std::f32::consts::PI;
        let poly6_coeff = 315.0 / (64.0 * pi * h.powi(9));
        let spiky_grad_coeff = -45.0 / (pi * h.powi(6));
        let visc_lap_coeff = 45.0 / (pi * h.powi(6));

        let mut density_duration = 0.0;
        let mut force_duration = 0.0;
        let mut int_duration = 0.0;

        for _ in 0..substeps {
            self.build_grid();

            // 1. Density and Pressure Solve
            let d_start = Instant::now();
            let positions: Vec<Vec3> = self.particles.iter().map(|p| p.position).collect();
            let head = &self.head;
            let next = &self.next;
            let grid_dim = self.grid_dim;
            let grid_min = self.grid_min;
            let cell_size = self.cell_size;

            let n = self.particles.len();
            let mut densities = vec![0.0f32; n];
            let mut pressures = vec![0.0f32; n];

            densities.par_iter_mut().zip(pressures.par_iter_mut()).enumerate().for_each(|(i, (rho_out, pres_out))| {
                let pi_pos = positions[i];
                let rel = pi_pos - grid_min;
                let cx = (rel.x / cell_size) as isize;
                let cy = (rel.y / cell_size) as isize;
                let cz = (rel.z / cell_size) as isize;

                let mut density = 0.0f32;

                for dz in -1..=1 {
                    let nz = cz + dz;
                    if nz < 0 || nz >= grid_dim[2] as isize { continue; }
                    for dy in -1..=1 {
                        let ny = cy + dy;
                        if ny < 0 || ny >= grid_dim[1] as isize { continue; }
                        for dx in -1..=1 {
                            let nx = cx + dx;
                            if nx < 0 || nx >= grid_dim[0] as isize { continue; }

                            let c_idx = nx as usize + ny as usize * grid_dim[0] + nz as usize * (grid_dim[0] * grid_dim[1]);
                            let mut j = head[c_idx];
                            while j != -1 {
                                let pj_pos = positions[j as usize];
                                let r2 = pi_pos.distance_squared(pj_pos);
                                if r2 < h2 {
                                    density += mass * poly6_coeff * (h2 - r2).powi(3);
                                }
                                j = next[j as usize];
                            }
                        }
                    }
                }

                let rho = density.max(params.rest_density);
                *rho_out = rho;
                // Tait equation of state or linear
                *pres_out = params.stiffness * ((rho / params.rest_density).powi(7) - 1.0).max(0.0);
            });
            density_duration += d_start.elapsed().as_secs_f64() * 1000.0;

            // 2. Forces Solve
            let f_start = Instant::now();
            let velocities: Vec<Vec3> = self.particles.iter().map(|p| p.velocity).collect();
            let mut accelerations = vec![Vec3::ZERO; n];

            accelerations.par_iter_mut().enumerate().for_each(|(i, acc_out)| {
                let pi_pos = positions[i];
                let pi_vel = velocities[i];
                let pi_rho = densities[i];
                let pi_p = pressures[i];

                let rel = pi_pos - grid_min;
                let cx = (rel.x / cell_size) as isize;
                let cy = (rel.y / cell_size) as isize;
                let cz = (rel.z / cell_size) as isize;

                let mut f_pressure = Vec3::ZERO;
                let mut f_visc = Vec3::ZERO;
                let mut f_surface = Vec3::ZERO;

                for dz in -1..=1 {
                    let nz = cz + dz;
                    if nz < 0 || nz >= grid_dim[2] as isize { continue; }
                    for dy in -1..=1 {
                        let ny = cy + dy;
                        if ny < 0 || ny >= grid_dim[1] as isize { continue; }
                        for dx in -1..=1 {
                            let nx = cx + dx;
                            if nx < 0 || nx >= grid_dim[0] as isize { continue; }

                            let c_idx = nx as usize + ny as usize * grid_dim[0] + nz as usize * (grid_dim[0] * grid_dim[1]);
                            let mut j = head[c_idx];
                            while j != -1 {
                                let j_idx = j as usize;
                                if i != j_idx {
                                    let pj_pos = positions[j_idx];
                                    let diff = pi_pos - pj_pos;
                                    let r = diff.length();
                                    if r > 1e-5 && r < h {
                                        let dir = diff / r;
                                        // Spiky gradient pressure
                                        let p_avg = (pi_p + pressures[j_idx]) * 0.5;
                                        let grad_w = spiky_grad_coeff * (h - r).powi(2) * dir;
                                        f_pressure -= mass * (p_avg / densities[j_idx]) * grad_w;

                                        // Viscosity
                                        let lap_w = visc_lap_coeff * (h - r);
                                        f_visc += params.viscosity * mass * ((velocities[j_idx] - pi_vel) / densities[j_idx]) * lap_w;

                                        // Surface tension attraction
                                        f_surface -= params.surface_tension * mass * (diff * (h - r) / densities[j_idx]);
                                    }
                                }
                                j = next[j as usize];
                            }
                        }
                    }
                }

                // Interactive mouse attractor/repulsor
                let mut f_mouse = Vec3::ZERO;
                if params.mouse_active {
                    let m_diff = pi_pos - params.mouse_pos;
                    let m_dist = m_diff.length();
                    if m_dist < params.mouse_radius && m_dist > 1e-4 {
                        let m_dir = m_diff / m_dist;
                        let falloff = 1.0 - (m_dist / params.mouse_radius);
                        f_mouse = m_dir * (params.mouse_strength * falloff);
                    }
                }

                *acc_out = (f_pressure + f_visc + f_surface) / pi_rho + params.gravity + f_mouse;
            });
            force_duration += f_start.elapsed().as_secs_f64() * 1000.0;

            // 3. Integration & Collision
            let i_start = Instant::now();
            let p_radius = params.particle_radius;
            let damping = params.damping;
            let b_min = params.boundary_min;
            let b_max = params.boundary_max;

            for (i, p) in self.particles.iter_mut().enumerate() {
                p.density = densities[i];
                p.pressure = pressures[i];
                p.velocity += accelerations[i] * sub_dt;
                p.position += p.velocity * sub_dt;

                // Boundary collision (glass container)
                if p.position.x - p_radius < b_min.x {
                    p.position.x = b_min.x + p_radius;
                    p.velocity.x = -p.velocity.x * damping;
                } else if p.position.x + p_radius > b_max.x {
                    p.position.x = b_max.x - p_radius;
                    p.velocity.x = -p.velocity.x * damping;
                }

                if p.position.y - p_radius < b_min.y {
                    p.position.y = b_min.y + p_radius;
                    p.velocity.y = -p.velocity.y * damping;
                    // Ground friction
                    p.velocity.x *= 0.95;
                    p.velocity.z *= 0.95;
                } else if p.position.y + p_radius > b_max.y {
                    p.position.y = b_max.y - p_radius;
                    p.velocity.y = -p.velocity.y * damping;
                }

                if p.position.z - p_radius < b_min.z {
                    p.position.z = b_min.z + p_radius;
                    p.velocity.z = -p.velocity.z * damping;
                } else if p.position.z + p_radius > b_max.z {
                    p.position.z = b_max.z - p_radius;
                    p.velocity.z = -p.velocity.z * damping;
                }

                // Obstacle collisions
                for obs in obstacles {
                    obs.collide_particle(&mut p.position, &mut p.velocity, p_radius, damping);
                }
            }
            int_duration += i_start.elapsed().as_secs_f64() * 1000.0;
        }

        self.sync_gpu_cache();

        StepMetrics {
            step_duration_ms: start.elapsed().as_secs_f64() * 1000.0,
            substep_count: substeps,
            particle_updates: (self.particles.len() as u64) * (substeps as u64),
            density_ms: density_duration,
            force_ms: force_duration,
            integration_ms: int_duration,
        }
    }

    fn reset(
        &mut self,
        scenario: ScenarioType,
        target_particles: usize,
        params: &mut SimParams,
        obstacles: &mut Vec<Obstacle>,
    ) {
        let config = create_scenario(scenario, target_particles);
        params.viscosity = config.viscosity;
        params.surface_tension = config.surface_tension;
        params.gravity = config.gravity;
        params.stiffness = config.stiffness;
        params.rest_density = config.rest_density;

        self.particles = config.particles;
        *obstacles = config.obstacles;
        self.init_grid(params.smoothing_radius);
        self.sync_gpu_cache();
    }

    fn particle_count(&self) -> usize {
        self.particles.len()
    }

    fn engine_type(&self) -> EngineType {
        EngineType::CpuMultiThread
    }

    fn add_particles(&mut self, new_particles: &[ParticleCpu]) {
        self.particles.extend_from_slice(new_particles);
        self.sync_gpu_cache();
    }
}
