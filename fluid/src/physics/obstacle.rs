use glam::Vec3;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ObstacleGpu {
    pub position: [f32; 3],
    pub radius: f32,
    pub velocity: [f32; 3],
    pub obstacle_type: u32, // 0 = disabled, 1 = sphere, 2 = rotor, 3 = box
    pub params: [f32; 4],   // rotor: [angle, angular_vel, blade_width, blade_length]
}

impl Default for ObstacleGpu {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            radius: 0.0,
            velocity: [0.0; 3],
            obstacle_type: 0,
            params: [0.0; 4],
        }
    }
}

#[derive(Clone, Debug)]
pub enum ObstacleKind {
    Sphere {
        radius: f32,
    },
    Rotor {
        radius: f32,
        height: f32,
        angular_velocity: f32,
        current_angle: f32,
        num_blades: u32,
    },
    BoxCollider {
        half_extents: Vec3,
    },
}

#[derive(Clone, Debug)]
pub struct Obstacle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub kind: ObstacleKind,
    pub is_grabbed: bool,
}

impl Obstacle {
    pub fn new_sphere(position: Vec3, radius: f32) -> Self {
        Self {
            position,
            velocity: Vec3::ZERO,
            kind: ObstacleKind::Sphere { radius },
            is_grabbed: false,
        }
    }

    pub fn new_rotor(position: Vec3, radius: f32, height: f32, rpm: f32) -> Self {
        let angular_velocity = rpm * std::f32::consts::TAU / 60.0;
        Self {
            position,
            velocity: Vec3::ZERO,
            kind: ObstacleKind::Rotor {
                radius,
                height,
                angular_velocity,
                current_angle: 0.0,
                num_blades: 4,
            },
            is_grabbed: false,
        }
    }

    pub fn update(&mut self, dt: f32) {
        if let ObstacleKind::Rotor {
            ref mut current_angle,
            angular_velocity,
            ..
        } = self.kind
        {
            *current_angle += angular_velocity * dt;
            if *current_angle > std::f32::consts::TAU {
                *current_angle -= std::f32::consts::TAU;
            }
        }
    }

    pub fn to_gpu(&self) -> ObstacleGpu {
        match self.kind {
            ObstacleKind::Sphere { radius } => ObstacleGpu {
                position: [self.position.x, self.position.y, self.position.z],
                radius,
                velocity: [self.velocity.x, self.velocity.y, self.velocity.z],
                obstacle_type: 1,
                params: [0.0; 4],
            },
            ObstacleKind::Rotor {
                radius,
                height,
                angular_velocity,
                current_angle,
                ..
            } => ObstacleGpu {
                position: [self.position.x, self.position.y, self.position.z],
                radius,
                velocity: [self.velocity.x, self.velocity.y, self.velocity.z],
                obstacle_type: 2,
                params: [current_angle, angular_velocity, height, radius],
            },
            ObstacleKind::BoxCollider { half_extents } => ObstacleGpu {
                position: [self.position.x, self.position.y, self.position.z],
                radius: half_extents.max_element(),
                velocity: [self.velocity.x, self.velocity.y, self.velocity.z],
                obstacle_type: 3,
                params: [half_extents.x, half_extents.y, half_extents.z, 0.0],
            },
        }
    }

    pub fn collide_particle(&self, pos: &mut Vec3, vel: &mut Vec3, particle_radius: f32, restitution: f32) -> bool {
        match self.kind {
            ObstacleKind::Sphere { radius } => {
                let diff = *pos - self.position;
                let dist = diff.length();
                let min_dist = radius + particle_radius;
                if dist < min_dist && dist > 1e-6 {
                    let normal = diff / dist;
                    *pos = self.position + normal * min_dist;
                    let rel_vel = *vel - self.velocity;
                    let normal_vel = rel_vel.dot(normal);
                    if normal_vel < 0.0 {
                        *vel = self.velocity + (rel_vel - (1.0 + restitution) * normal_vel * normal);
                    }
                    return true;
                }
                false
            }
            ObstacleKind::Rotor {
                radius,
                height,
                angular_velocity,
                current_angle,
                num_blades,
            } => {
                let diff = *pos - self.position;
                let r_dist = (diff.x * diff.x + diff.z * diff.z).sqrt();
                if r_dist > radius || diff.y.abs() > height * 0.5 {
                    return false;
                }

                let p_angle = diff.z.atan2(diff.x);
                let blade_step = std::f32::consts::TAU / num_blades as f32;

                for b in 0..num_blades {
                    let b_angle = current_angle + b as f32 * blade_step;
                    let mut angle_diff = (p_angle - b_angle) % std::f32::consts::TAU;
                    if angle_diff > std::f32::consts::PI {
                        angle_diff -= std::f32::consts::TAU;
                    } else if angle_diff < -std::f32::consts::PI {
                        angle_diff += std::f32::consts::TAU;
                    }

                    let arc_dist = r_dist * angle_diff.abs();
                    let blade_thickness = 0.035;
                    if arc_dist < (blade_thickness + particle_radius) {
                        let blade_normal_angle = b_angle + if angle_diff > 0.0 { std::f32::consts::FRAC_PI_2 } else { -std::f32::consts::FRAC_PI_2 };
                        let normal = Vec3::new(blade_normal_angle.cos(), 0.0, blade_normal_angle.sin());
                        let tang_vel_mag = angular_velocity * r_dist;
                        let blade_vel = Vec3::new(-b_angle.sin(), 0.0, b_angle.cos()) * tang_vel_mag;

                        *pos += normal * ((blade_thickness + particle_radius) - arc_dist);
                        *vel = blade_vel + normal * 2.0;
                        return true;
                    }
                }
                false
            }
            ObstacleKind::BoxCollider { half_extents } => {
                let diff = *pos - self.position;
                let d = diff.abs() - (half_extents + Vec3::splat(particle_radius));
                if d.x < 0.0 && d.y < 0.0 && d.z < 0.0 {
                    let overlap = -d;
                    if overlap.x < overlap.y && overlap.x < overlap.z {
                        let sign = diff.x.signum();
                        pos.x = self.position.x + sign * (half_extents.x + particle_radius);
                        vel.x = -vel.x * restitution;
                    } else if overlap.y < overlap.z {
                        let sign = diff.y.signum();
                        pos.y = self.position.y + sign * (half_extents.y + particle_radius);
                        vel.y = -vel.y * restitution;
                    } else {
                        let sign = diff.z.signum();
                        pos.z = self.position.z + sign * (half_extents.z + particle_radius);
                        vel.z = -vel.z * restitution;
                    }
                    return true;
                }
                false
            }
        }
    }
}
