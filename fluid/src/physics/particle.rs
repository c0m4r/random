use bytemuck::{Pod, Zeroable};
use glam::Vec3;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ParticleGpu {
    pub position: [f32; 3],
    pub density: f32,
    pub velocity: [f32; 3],
    pub pressure: f32,
    pub color: [f32; 4],
}

impl Default for ParticleGpu {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            density: 1000.0,
            velocity: [0.0; 3],
            pressure: 0.0,
            color: [0.15, 0.55, 0.95, 1.0],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct FoamParticleGpu {
    pub position: [f32; 3],
    pub lifetime: f32,
    pub velocity: [f32; 3],
    pub scale: f32,
}

impl Default for FoamParticleGpu {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            lifetime: 0.0,
            velocity: [0.0; 3],
            scale: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ParticleCpu {
    pub position: Vec3,
    pub velocity: Vec3,
    pub acceleration: Vec3,
    pub density: f32,
    pub pressure: f32,
    pub color: [f32; 4],
}

impl ParticleCpu {
    pub fn new(position: Vec3, velocity: Vec3, color: [f32; 4]) -> Self {
        Self {
            position,
            velocity,
            acceleration: Vec3::ZERO,
            density: 1000.0,
            pressure: 0.0,
            color,
        }
    }

    pub fn to_gpu(&self) -> ParticleGpu {
        ParticleGpu {
            position: [self.position.x, self.position.y, self.position.z],
            density: self.density,
            velocity: [self.velocity.x, self.velocity.y, self.velocity.z],
            pressure: self.pressure,
            color: self.color,
        }
    }
}
