pub mod particle;
pub mod obstacle;
pub mod scenarios;
pub mod cpu_sph;
pub mod gpu_sph;

use glam::Vec3;
use bytemuck::{Pod, Zeroable};
use crate::physics::obstacle::{Obstacle, ObstacleGpu};
use crate::physics::scenarios::ScenarioType;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SimParamsGpu {
    pub gravity: [f32; 4],          // xyz, w = 0
    pub boundary_min: [f32; 4],     // xyz, w = 0
    pub boundary_max: [f32; 4],     // xyz, w = 0
    pub mouse_pos: [f32; 4],        // xyz, w = active (0.0 or 1.0)
    pub particle_count: u32,
    pub rest_density: f32,
    pub stiffness: f32,
    pub viscosity: f32,
    pub surface_tension: f32,
    pub smoothing_radius: f32,
    pub particle_radius: f32,
    pub damping: f32,
    pub dt: f32,
    pub mouse_radius: f32,
    pub mouse_strength: f32,
    pub pad0: f32,
    pub obstacles: [ObstacleGpu; 4],
}

#[derive(Clone, Debug)]
pub struct SimParams {
    pub gravity: Vec3,
    pub boundary_min: Vec3,
    pub boundary_max: Vec3,
    pub rest_density: f32,
    pub stiffness: f32,
    pub viscosity: f32,
    pub surface_tension: f32,
    pub smoothing_radius: f32,
    pub particle_radius: f32,
    pub damping: f32,
    pub substeps: u32,
    pub mouse_pos: Vec3,
    pub mouse_active: bool,
    pub mouse_radius: f32,
    pub mouse_strength: f32,
}

impl Default for SimParams {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            boundary_min: Vec3::new(-1.0, 0.0, -0.6),
            boundary_max: Vec3::new(1.0, 1.4, 0.6),
            rest_density: 1000.0,
            stiffness: 220.0,
            viscosity: 0.04,
            surface_tension: 0.02,
            smoothing_radius: 0.048,
            particle_radius: 0.016,
            damping: 0.35,
            substeps: 2,
            mouse_pos: Vec3::ZERO,
            mouse_active: false,
            mouse_radius: 0.35,
            mouse_strength: 35.0,
        }
    }
}

impl SimParams {
    pub fn to_gpu(&self, particle_count: u32, dt: f32, obstacles: &[Obstacle]) -> SimParamsGpu {
        let mut obs_gpu = [ObstacleGpu::default(); 4];
        for (i, obs) in obstacles.iter().take(4).enumerate() {
            obs_gpu[i] = obs.to_gpu();
        }

        SimParamsGpu {
            gravity: [self.gravity.x, self.gravity.y, self.gravity.z, 0.0],
            boundary_min: [self.boundary_min.x, self.boundary_min.y, self.boundary_min.z, 0.0],
            boundary_max: [self.boundary_max.x, self.boundary_max.y, self.boundary_max.z, 0.0],
            mouse_pos: [
                self.mouse_pos.x,
                self.mouse_pos.y,
                self.mouse_pos.z,
                if self.mouse_active { 1.0 } else { 0.0 },
            ],
            particle_count,
            rest_density: self.rest_density,
            stiffness: self.stiffness,
            viscosity: self.viscosity,
            surface_tension: self.surface_tension,
            smoothing_radius: self.smoothing_radius,
            particle_radius: self.particle_radius,
            damping: self.damping,
            dt,
            mouse_radius: self.mouse_radius,
            mouse_strength: self.mouse_strength,
            pad0: 0.0,
            obstacles: obs_gpu,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EngineType {
    GpuCompute,
    CpuMultiThread,
}

#[derive(Clone, Debug, Default)]
pub struct StepMetrics {
    pub step_duration_ms: f64,
    pub substep_count: u32,
    pub particle_updates: u64,
    pub density_ms: f64,
    pub force_ms: f64,
    pub integration_ms: f64,
}

pub trait PhysicsEngine {
    fn step(&mut self, dt: f32, params: &SimParams, obstacles: &[Obstacle]) -> StepMetrics;
    fn reset(&mut self, scenario: ScenarioType, target_particles: usize, params: &mut SimParams, obstacles: &mut Vec<Obstacle>);
    fn particle_count(&self) -> usize;
    fn engine_type(&self) -> EngineType;
    fn add_particles(&mut self, new_particles: &[particle::ParticleCpu]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::scenarios::{create_scenario, ScenarioType};
    use crate::physics::cpu_sph::CpuSph;

    #[test]
    fn test_scenario_generation() {
        let sc = create_scenario(ScenarioType::DamBreak, 1000);
        assert!(sc.particles.len() > 100);
        for p in &sc.particles {
            assert!(p.position.x >= -1.0 && p.position.x <= 1.0);
            assert!(p.position.y >= 0.0 && p.position.y <= 1.5);
            assert!(p.position.z >= -0.6 && p.position.z <= 0.6);
        }
    }

    #[test]
    fn test_cpu_sph_stepping() {
        let (mut engine, params, obstacles) = CpuSph::new(ScenarioType::DamBreak, 500);
        assert!(engine.particle_count() > 0);

        let metrics = engine.step(0.003, &params, &obstacles);
        assert!(metrics.particle_updates > 0);
        assert!(metrics.step_duration_ms >= 0.0);
    }

    #[test]
    fn test_sphere_obstacle_collision() {
        let obs = Obstacle::new_sphere(Vec3::new(0.0, 0.5, 0.0), 0.2);
        let mut pos = Vec3::new(0.05, 0.5, 0.0);
        let mut vel = Vec3::new(-1.0, 0.0, 0.0);
        let collided = obs.collide_particle(&mut pos, &mut vel, 0.015, 0.5);
        assert!(collided);
        assert!(pos.x > 0.15); // Pushed outward past radius + particle radius
    }
}

