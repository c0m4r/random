use crate::worker::Snapshot;
use eframe::{
    egui,
    glow::{self, HasContext},
};
use glam::{Mat4, Vec3};

#[derive(Clone)]
pub struct View {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub exposure: f32,
    pub field: i32,
    pub cut: f32,
    pub beaming: bool,
    pub samples: i32,
    pub range: [f32; 2],
    pub logarithmic: bool,
}
impl Default for View {
    fn default() -> Self {
        Self {
            yaw: 0.45,
            pitch: 0.27,
            distance: 5.3,
            exposure: 1.2,
            field: 1,
            cut: 1.0,
            beaming: false,
            samples: 144,
            range: [0.02, 0.7],
            logarithmic: true,
        }
    }
}
impl View {
    pub fn eye(&self) -> Vec3 {
        Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.cos() * self.pitch.cos(),
        ) * self.distance
    }
    pub fn matrix(&self, aspect: f32) -> Mat4 {
        glam::camera::rh::proj::opengl::perspective(43.0_f32.to_radians(), aspect, 0.05, 50.0)
            * glam::camera::rh::view::look_at_mat4(self.eye(), Vec3::ZERO, Vec3::Y)
    }
    pub fn project(&self, p: Vec3, rect: egui::Rect) -> Option<egui::Pos2> {
        let clip = self.matrix(rect.aspect_ratio()) * p.extend(1.0);
        if clip.w <= 0.0 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        Some(egui::pos2(
            rect.center().x + ndc.x * rect.width() * 0.5,
            rect.center().y - ndc.y * rect.height() * 0.5,
        ))
    }
    pub fn hit_plane(&self, pos: egui::Pos2, rect: egui::Rect) -> Option<[f64; 3]> {
        let x = (pos.x - rect.center().x) * 2.0 / rect.width();
        let y = -(pos.y - rect.center().y) * 2.0 / rect.height();
        let inv = self.matrix(rect.aspect_ratio()).inverse();
        let q = inv * glam::Vec4::new(x, y, 1.0, 1.0);
        let dir = (q.truncate() / q.w - self.eye()).normalize();
        if dir.z.abs() < 1e-5 {
            return None;
        }
        let t = -self.eye().z / dir.z;
        let p = self.eye() + dir * t;
        if t <= 0.0 || p.x.abs() >= 1.5 || p.y.abs() >= 1.0 {
            return None;
        }
        Some([p.x as f64, p.y as f64, 0.0])
    }
}

pub struct Renderer {
    program: glow::Program,
    composite: glow::Program,
    vao: glow::VertexArray,
    volume: glow::Texture,
    velocity: glow::Texture,
    framebuffer: glow::Framebuffer,
    color: glow::Texture,
    target_size: [i32; 2],
    revision: u64,
}
impl Renderer {
    pub fn new(gl: &glow::Context) -> Result<Self, String> {
        unsafe {
            let program = compile(gl, include_str!("shaders/volume.frag"))?;
            let composite = compile(gl, include_str!("shaders/composite.frag"))?;
            let vao = gl.create_vertex_array()?;
            let volume = gl.create_texture()?;
            let velocity = gl.create_texture()?;
            let framebuffer = gl.create_framebuffer()?;
            let color = gl.create_texture()?;
            gl.bind_texture(glow::TEXTURE_2D, Some(color));
            for p in [glow::TEXTURE_MIN_FILTER, glow::TEXTURE_MAG_FILTER] {
                gl.tex_parameter_i32(glow::TEXTURE_2D, p, glow::LINEAR as i32);
            }
            for p in [glow::TEXTURE_WRAP_S, glow::TEXTURE_WRAP_T] {
                gl.tex_parameter_i32(glow::TEXTURE_2D, p, glow::CLAMP_TO_EDGE as i32);
            }
            for texture in [volume, velocity] {
                gl.bind_texture(glow::TEXTURE_3D, Some(texture));
                for p in [glow::TEXTURE_MIN_FILTER, glow::TEXTURE_MAG_FILTER] {
                    gl.tex_parameter_i32(glow::TEXTURE_3D, p, glow::LINEAR as i32);
                }
                for p in [
                    glow::TEXTURE_WRAP_S,
                    glow::TEXTURE_WRAP_T,
                    glow::TEXTURE_WRAP_R,
                ] {
                    gl.tex_parameter_i32(glow::TEXTURE_3D, p, glow::CLAMP_TO_EDGE as i32);
                }
            }
            Ok(Self {
                program,
                composite,
                vao,
                volume,
                velocity,
                framebuffer,
                color,
                target_size: [0, 0],
                revision: 0,
            })
        }
    }
    pub fn upload(&mut self, gl: &glow::Context, s: &Snapshot) {
        if self.revision == s.revision {
            return;
        }
        unsafe {
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            for (texture, data) in [
                (self.volume, s.texture()),
                (self.velocity, s.velocity_texture()),
            ] {
                gl.bind_texture(glow::TEXTURE_3D, Some(texture));
                gl.tex_image_3d(
                    glow::TEXTURE_3D,
                    0,
                    glow::RGBA32F as i32,
                    s.grid.n[0] as i32,
                    s.grid.n[1] as i32,
                    s.grid.n[2] as i32,
                    0,
                    glow::RGBA,
                    glow::FLOAT,
                    glow::PixelUnpackData::Slice(Some(bytemuck::cast_slice(&data))),
                );
            }
            self.revision = s.revision;
        }
    }
    pub fn paint(&mut self, gl: &glow::Context, v: &View, aspect: f32) {
        unsafe {
            let previous = gl.get_parameter_framebuffer(glow::DRAW_FRAMEBUFFER_BINDING);
            let mut viewport = [0; 4];
            gl.get_parameter_i32_slice(glow::VIEWPORT, &mut viewport);
            let scale = if v.samples > 160 { 1.0 } else { 0.65 };
            let size = [
                (viewport[2] as f32 * scale).max(1.0) as i32,
                (viewport[3] as f32 * scale).max(1.0) as i32,
            ];
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(self.color));
            if size != self.target_size {
                gl.tex_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    glow::RGBA16F as i32,
                    size[0],
                    size[1],
                    0,
                    glow::RGBA,
                    glow::FLOAT,
                    glow::PixelUnpackData::Slice(None),
                );
                self.target_size = size;
            }
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.framebuffer));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(self.color),
                0,
            );
            gl.viewport(0, 0, size[0], size[1]);
            gl.disable(glow::SCISSOR_TEST);
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::CULL_FACE);
            gl.disable(glow::BLEND);
            gl.use_program(Some(self.program));
            gl.bind_vertex_array(Some(self.vao));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_3D, Some(self.volume));
            gl.uniform_1_i32(gl.get_uniform_location(self.program, "volume").as_ref(), 0);
            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_3D, Some(self.velocity));
            gl.uniform_1_i32(
                gl.get_uniform_location(self.program, "velocity").as_ref(),
                1,
            );
            let inv = v.matrix(aspect).inverse().to_cols_array();
            gl.uniform_matrix_4_f32_slice(
                gl.get_uniform_location(self.program, "inv_vp").as_ref(),
                false,
                &inv,
            );
            let eye = v.eye();
            gl.uniform_3_f32(
                gl.get_uniform_location(self.program, "eye").as_ref(),
                eye.x,
                eye.y,
                eye.z,
            );
            for (name, value) in [("exposure", v.exposure), ("cut", v.cut)] {
                gl.uniform_1_f32(gl.get_uniform_location(self.program, name).as_ref(), value);
            }
            gl.uniform_2_f32(
                gl.get_uniform_location(self.program, "value_range")
                    .as_ref(),
                v.range[0],
                v.range[1],
            );
            for (name, value) in [
                ("field", v.field),
                ("beaming", v.beaming as i32),
                ("samples", v.samples),
                ("logarithmic", v.logarithmic as i32),
            ] {
                gl.uniform_1_i32(gl.get_uniform_location(self.program, name).as_ref(), value);
            }
            gl.draw_arrays(glow::TRIANGLES, 0, 3);
            gl.active_texture(glow::TEXTURE0);
            gl.bind_framebuffer(glow::FRAMEBUFFER, previous);
            gl.viewport(viewport[0], viewport[1], viewport[2], viewport[3]);
            gl.enable(glow::SCISSOR_TEST);
            gl.use_program(Some(self.composite));
            gl.bind_texture(glow::TEXTURE_2D, Some(self.color));
            gl.uniform_1_i32(gl.get_uniform_location(self.composite, "scene").as_ref(), 0);
            gl.uniform_2_f32(
                gl.get_uniform_location(self.composite, "pixel").as_ref(),
                1.0 / size[0] as f32,
                1.0 / size[1] as f32,
            );
            gl.draw_arrays(glow::TRIANGLES, 0, 3);
        }
    }
    pub fn destroy(&self, gl: &glow::Context) {
        unsafe {
            gl.delete_texture(self.volume);
            gl.delete_texture(self.velocity);
            gl.delete_texture(self.color);
            gl.delete_framebuffer(self.framebuffer);
            gl.delete_vertex_array(self.vao);
            gl.delete_program(self.program);
            gl.delete_program(self.composite);
        }
    }
}

fn compile(gl: &glow::Context, fragment: &str) -> Result<glow::Program, String> {
    // All graphics calls occur on eframe's context-owning render thread.
    unsafe {
        let program = gl.create_program()?;
        for (kind, src) in [
            (glow::VERTEX_SHADER, include_str!("shaders/volume.vert")),
            (glow::FRAGMENT_SHADER, fragment),
        ] {
            let shader = gl.create_shader(kind)?;
            gl.shader_source(shader, src);
            gl.compile_shader(shader);
            if !gl.get_shader_compile_status(shader) {
                let error = gl.get_shader_info_log(shader);
                gl.delete_shader(shader);
                gl.delete_program(program);
                return Err(error);
            }
            gl.attach_shader(program, shader);
            gl.delete_shader(shader);
        }
        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            let error = gl.get_program_info_log(program);
            gl.delete_program(program);
            return Err(error);
        }
        Ok(program)
    }
}
