use glam::{Mat4, Vec2, Vec3};
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CameraUniforms {
    pub view_proj: [f32; 16],
    pub inv_view_proj: [f32; 16],
    pub view: [f32; 16],
    pub proj: [f32; 16],
    pub eye_pos: [f32; 4], // xyz, w = 1.0
    pub light_dir: [f32; 4], // xyz = normalized direction to light, w = 0.0
    pub viewport_size: [f32; 2],
    pub near_far: [f32; 2], // x = near, y = far
}

#[derive(Debug, Clone)]
pub struct Camera {
    pub target: Vec3,
    pub distance: f32,
    pub yaw: f32,   // in radians
    pub pitch: f32, // in radians
    pub fov_y: f32, // in radians
    pub z_near: f32,
    pub z_far: f32,
    pub aspect_ratio: f32,
    pub auto_rotate: bool,
    pub auto_rotate_speed: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            target: Vec3::new(0.0, 0.4, 0.0),
            distance: 3.2,
            yaw: -0.785, // ~-45 degrees
            pitch: 0.45,  // ~25 degrees
            fov_y: 55.0_f32.to_radians(),
            z_near: 0.05,
            z_far: 100.0,
            aspect_ratio: 16.0 / 9.0,
            auto_rotate: false,
            auto_rotate_speed: 0.2,
        }
    }
}

impl Camera {
    pub fn eye_position(&self) -> Vec3 {
        let cos_pitch = self.pitch.cos();
        let sin_pitch = self.pitch.sin();
        let cos_yaw = self.yaw.cos();
        let sin_yaw = self.yaw.sin();

        let offset = Vec3::new(
            self.distance * cos_pitch * sin_yaw,
            self.distance * sin_pitch,
            self.distance * cos_pitch * cos_yaw,
        );
        self.target + offset
    }

    #[allow(deprecated)]
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye_position(), self.target, Vec3::Y)
    }

    #[allow(deprecated)]
    pub fn proj_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, self.aspect_ratio, self.z_near, self.z_far)
    }

    pub fn update(&mut self, dt: f32) {
        if self.auto_rotate {
            self.yaw += self.auto_rotate_speed * dt;
            if self.yaw > std::f32::consts::TAU {
                self.yaw -= std::f32::consts::TAU;
            }
        }
    }

    pub fn orbit(&mut self, delta_x: f32, delta_y: f32) {
        let sensitivity = 0.005;
        self.yaw -= delta_x * sensitivity;
        self.pitch += delta_y * sensitivity;
        self.pitch = self.pitch.clamp(-1.45, 1.45);
    }

    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        let sensitivity = self.distance * 0.0012;
        let forward = (self.target - self.eye_position()).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward).normalize();

        self.target += (-right * delta_x + up * delta_y) * sensitivity;
    }

    pub fn zoom(&mut self, delta: f32) {
        self.distance = (self.distance * (1.0 - delta * 0.08)).clamp(0.5, 20.0);
    }

    pub fn raycast_from_screen(&self, mouse_pos: Vec2, viewport_size: Vec2) -> (Vec3, Vec3) {
        // Returns (ray_origin, ray_direction)
        let ndc_x = (2.0 * mouse_pos.x / viewport_size.x) - 1.0;
        let ndc_y = 1.0 - (2.0 * mouse_pos.y / viewport_size.y);

        let inv_view_proj = (self.proj_matrix() * self.view_matrix()).inverse();
        let near_point = inv_view_proj.project_point3(Vec3::new(ndc_x, ndc_y, 0.0));
        let far_point = inv_view_proj.project_point3(Vec3::new(ndc_x, ndc_y, 1.0));

        let ray_dir = (far_point - near_point).normalize();
        (near_point, ray_dir)
    }

    pub fn build_uniforms(&self, viewport_size: Vec2) -> CameraUniforms {
        let view = self.view_matrix();
        let proj = self.proj_matrix();
        let view_proj = proj * view;
        let inv_view_proj = view_proj.inverse();
        let eye = self.eye_position();

        let light_dir = Vec3::new(0.6, 1.0, 0.8).normalize();

        CameraUniforms {
            view_proj: view_proj.to_cols_array(),
            inv_view_proj: inv_view_proj.to_cols_array(),
            view: view.to_cols_array(),
            proj: proj.to_cols_array(),
            eye_pos: [eye.x, eye.y, eye.z, 1.0],
            light_dir: [light_dir.x, light_dir.y, light_dir.z, 0.0],
            viewport_size: [viewport_size.x, viewport_size.y],
            near_far: [self.z_near, self.z_far],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_orbit() {
        let mut cam = Camera::default();
        let initial_eye = cam.eye_position();
        let initial_dist = (initial_eye - cam.target).length();
        cam.orbit(std::f32::consts::PI * 0.5, 0.0);
        let rotated_eye = cam.eye_position();
        let rotated_dist = (rotated_eye - cam.target).length();
        assert!((initial_dist - rotated_dist).abs() < 1e-4);
        assert!((initial_eye.y - rotated_eye.y).abs() < 1e-4);
    }

    #[test]
    fn test_camera_zoom_clamping() {
        let mut cam = Camera::default();
        cam.zoom(-100.0);
        assert!(cam.distance >= 0.5);
        cam.zoom(100.0);
        assert!(cam.distance <= 15.0);
    }

    #[test]
    fn test_camera_uniforms_invertibility() {
        let cam = Camera::default();
        let uniforms = cam.build_uniforms(Vec2::new(1920.0, 1080.0));
        let vp = glam::Mat4::from_cols_array(&uniforms.view_proj);
        let inv_vp = glam::Mat4::from_cols_array(&uniforms.inv_view_proj);
        let ident = vp * inv_vp;
        assert!((ident.col(0).x - 1.0).abs() < 1e-3);
        assert!((ident.col(1).y - 1.0).abs() < 1e-3);
        assert!((ident.col(2).z - 1.0).abs() < 1e-3);
    }
}

