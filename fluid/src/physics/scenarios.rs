use glam::Vec3;
use crate::physics::particle::ParticleCpu;
use crate::physics::obstacle::Obstacle;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ScenarioType {
    DamBreak,
    DoubleWave,
    DropletSplash,
    ViscousHoney,
    VortexMixer,
    ZeroG,
    BenchmarkStress,
}

impl ScenarioType {
    pub const ALL: [ScenarioType; 7] = [
        ScenarioType::DamBreak,
        ScenarioType::DoubleWave,
        ScenarioType::DropletSplash,
        ScenarioType::ViscousHoney,
        ScenarioType::VortexMixer,
        ScenarioType::ZeroG,
        ScenarioType::BenchmarkStress,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            ScenarioType::DamBreak => "Dam Break (High Energy)",
            ScenarioType::DoubleWave => "Double Wave Collision",
            ScenarioType::DropletSplash => "Crown Droplet Splash",
            ScenarioType::ViscousHoney => "Viscous Fluid (Honey)",
            ScenarioType::VortexMixer => "Turbine Vortex Mixer",
            ScenarioType::ZeroG => "Zero-G Surface Tension",
            ScenarioType::BenchmarkStress => "Extreme Benchmark Stress",
        }
    }
}

pub struct ScenarioConfig {
    pub particles: Vec<ParticleCpu>,
    pub obstacles: Vec<Obstacle>,
    pub viscosity: f32,
    pub surface_tension: f32,
    pub gravity: Vec3,
    pub stiffness: f32,
    pub rest_density: f32,
}

pub fn spawn_block(
    min: Vec3,
    max: Vec3,
    spacing: f32,
    initial_vel: Vec3,
    color: [f32; 4],
) -> Vec<ParticleCpu> {
    let mut particles = Vec::new();
    let mut x = min.x;
    while x <= max.x {
        let mut y = min.y;
        while y <= max.y {
            let mut z = min.z;
            while z <= max.z {
                // Add minor jitter to prevent perfect grid crystallization
                let jitter = Vec3::new(
                    (x.sin() * 43_758.547).fract() - 0.5,
                    (y.sin() * 23_421.63).fract() - 0.5,
                    (z.sin() * 85_321.195).fract() - 0.5,
                ) * (spacing * 0.15);

                particles.push(ParticleCpu::new(
                    Vec3::new(x, y, z) + jitter,
                    initial_vel,
                    color,
                ));
                z += spacing;
            }
            y += spacing;
        }
        x += spacing;
    }
    particles
}

pub fn spawn_sphere(
    center: Vec3,
    radius: f32,
    spacing: f32,
    initial_vel: Vec3,
    color: [f32; 4],
) -> Vec<ParticleCpu> {
    let mut particles = Vec::new();
    let r2 = radius * radius;
    let mut x = center.x - radius;
    while x <= center.x + radius {
        let mut y = center.y - radius;
        while y <= center.y + radius {
            let mut z = center.z - radius;
            while z <= center.z + radius {
                let p = Vec3::new(x, y, z);
                if p.distance_squared(center) <= r2 {
                    let jitter = Vec3::new(
                        (x.sin() * 43_758.547).fract() - 0.5,
                        (y.sin() * 23_421.63).fract() - 0.5,
                        (z.sin() * 85_321.195).fract() - 0.5,
                    ) * (spacing * 0.15);
                    particles.push(ParticleCpu::new(p + jitter, initial_vel, color));
                }
                z += spacing;
            }
            y += spacing;
        }
        x += spacing;
    }
    particles
}

pub fn create_scenario(scenario_type: ScenarioType, target_particle_count: usize) -> ScenarioConfig {
    match scenario_type {
        ScenarioType::DamBreak => {
            // High energy dam break against a central test sphere obstacle
            let spacing = (0.7 * 0.9 * 0.9 / (target_particle_count as f32).max(100.0)).cbrt();
            let particles = spawn_block(
                Vec3::new(-0.95, 0.05, -0.45),
                Vec3::new(-0.25, 0.95, 0.45),
                spacing,
                Vec3::ZERO,
                [0.15, 0.55, 0.95, 1.0], // Azure Blue
            );
            let obstacles = vec![
                Obstacle::new_sphere(Vec3::new(0.35, 0.25, 0.0), 0.22),
            ];
            ScenarioConfig {
                particles,
                obstacles,
                viscosity: 0.04,
                surface_tension: 0.015,
                gravity: Vec3::new(0.0, -9.81, 0.0),
                stiffness: 220.0,
                rest_density: 1000.0,
            }
        }
        ScenarioType::DoubleWave => {
            let spacing = (0.6 * 0.9 * 0.8 / (target_particle_count as f32).max(100.0)).cbrt();
            let mut p1 = spawn_block(
                Vec3::new(-0.95, 0.05, -0.45),
                Vec3::new(-0.45, 0.90, 0.45),
                spacing,
                Vec3::new(1.0, 0.0, 0.0),
                [0.10, 0.75, 0.85, 1.0], // Cyan
            );
            let p2 = spawn_block(
                Vec3::new(0.45, 0.05, -0.45),
                Vec3::new(0.95, 0.90, 0.45),
                spacing,
                Vec3::new(-1.0, 0.0, 0.0),
                [0.95, 0.35, 0.15, 1.0], // Amber Fire
            );
            p1.extend(p2);
            ScenarioConfig {
                particles: p1,
                obstacles: vec![],
                viscosity: 0.03,
                surface_tension: 0.02,
                gravity: Vec3::new(0.0, -9.81, 0.0),
                stiffness: 240.0,
                rest_density: 1000.0,
            }
        }
        ScenarioType::DropletSplash => {
            let spacing = (0.9 * 0.3 * 0.8 / (target_particle_count as f32).max(100.0)).cbrt();
            // Pool on bottom
            let mut pool = spawn_block(
                Vec3::new(-0.95, 0.05, -0.55),
                Vec3::new(0.95, 0.25, 0.55),
                spacing,
                Vec3::ZERO,
                [0.2, 0.6, 0.9, 1.0],
            );
            // Droplet falling from above
            let droplet = spawn_sphere(
                Vec3::new(0.0, 0.85, 0.0),
                0.22,
                spacing,
                Vec3::new(0.0, -3.5, 0.0),
                [0.9, 0.95, 1.0, 1.0], // White/Milk crown
            );
            pool.extend(droplet);
            ScenarioConfig {
                particles: pool,
                obstacles: vec![],
                viscosity: 0.025,
                surface_tension: 0.035,
                gravity: Vec3::new(0.0, -9.81, 0.0),
                stiffness: 220.0,
                rest_density: 1000.0,
            }
        }
        ScenarioType::ViscousHoney => {
            let spacing = (0.5 * 0.8 * 0.5 / (target_particle_count as f32).max(100.0)).cbrt();
            let particles = spawn_block(
                Vec3::new(-0.25, 0.40, -0.25),
                Vec3::new(0.25, 1.20, 0.25),
                spacing,
                Vec3::new(0.0, -0.5, 0.0),
                [0.95, 0.65, 0.05, 1.0], // Golden Honey
            );
            ScenarioConfig {
                particles,
                obstacles: vec![
                    Obstacle::new_sphere(Vec3::new(0.0, 0.25, 0.0), 0.18),
                ],
                viscosity: 0.75, // High viscosity!
                surface_tension: 0.04,
                gravity: Vec3::new(0.0, -9.81, 0.0),
                stiffness: 180.0,
                rest_density: 1200.0,
            }
        }
        ScenarioType::VortexMixer => {
            let spacing = (1.2 * 0.5 * 0.9 / (target_particle_count as f32).max(100.0)).cbrt();
            let particles = spawn_block(
                Vec3::new(-0.85, 0.05, -0.45),
                Vec3::new(0.85, 0.55, 0.45),
                spacing,
                Vec3::ZERO,
                [0.05, 0.85, 0.65, 1.0], // Emerald Green
            );
            let rotor = Obstacle::new_rotor(Vec3::new(0.0, 0.25, 0.0), 0.35, 0.4, 180.0); // 180 RPM
            ScenarioConfig {
                particles,
                obstacles: vec![rotor],
                viscosity: 0.035,
                surface_tension: 0.015,
                gravity: Vec3::new(0.0, -9.81, 0.0),
                stiffness: 220.0,
                rest_density: 1000.0,
            }
        }
        ScenarioType::ZeroG => {
            let spacing = (0.7 * 0.7 * 0.7 / (target_particle_count as f32).max(100.0)).cbrt();
            let particles = spawn_sphere(
                Vec3::new(0.0, 0.6, 0.0),
                0.40,
                spacing,
                Vec3::ZERO,
                [0.75, 0.25, 0.85, 1.0], // Purple Plasma
            );
            ScenarioConfig {
                particles,
                obstacles: vec![],
                viscosity: 0.02,
                surface_tension: 0.08, // High surface tension for capillary oscillations
                gravity: Vec3::ZERO,   // Zero G!
                stiffness: 200.0,
                rest_density: 1000.0,
            }
        }
        ScenarioType::BenchmarkStress => {
            let spacing = (1.6 * 0.9 * 0.9 / (target_particle_count as f32).max(100.0)).cbrt();
            let particles = spawn_block(
                Vec3::new(-0.85, 0.05, -0.45),
                Vec3::new(0.85, 0.95, 0.45),
                spacing,
                Vec3::ZERO,
                [0.2, 0.5, 0.9, 1.0],
            );
            let obstacles = vec![
                Obstacle::new_sphere(Vec3::new(0.0, 0.4, 0.0), 0.25),
            ];
            ScenarioConfig {
                particles,
                obstacles,
                viscosity: 0.04,
                surface_tension: 0.02,
                gravity: Vec3::new(0.0, -9.81, 0.0),
                stiffness: 220.0,
                rest_density: 1000.0,
            }
        }
    }
}
