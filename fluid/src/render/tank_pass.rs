use glam::Vec3;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

use crate::physics::obstacle::{Obstacle, ObstacleKind};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct TankVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
}

pub struct TankPass {
    pub pipeline: wgpu::RenderPipeline,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    pub obstacle_vertex_buffer: wgpu::Buffer,
    pub obstacle_index_buffer: wgpu::Buffer,
    pub obstacle_index_count: u32,
}

impl TankPass {
    pub fn new(
        device: &wgpu::Device,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        surface_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tank Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/tank.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Tank Pipeline Layout"),
            bind_group_layouts: &[Some(camera_bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Tank Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<TankVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: 12,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: 24,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                    ],
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let (vertices, indices) = Self::generate_tank_and_floor();
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tank Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tank Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let max_obs_verts = 10000;
        let obstacle_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Obstacle Vertex Buffer"),
            size: (max_obs_verts * std::mem::size_of::<TankVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let obstacle_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Obstacle Index Buffer"),
            size: (max_obs_verts * 3 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            obstacle_vertex_buffer,
            obstacle_index_buffer,
            obstacle_index_count: 0,
        }
    }

    fn generate_tank_and_floor() -> (Vec<TankVertex>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let floor_size = 8.0;
        let floor_y = 0.0;
        let base_idx = vertices.len() as u32;
        let floor_col = [0.15, 0.17, 0.22, 1.0];
        vertices.push(TankVertex { position: [-floor_size, floor_y, -floor_size], normal: [0.0, 1.0, 0.0], color: floor_col });
        vertices.push(TankVertex { position: [ floor_size, floor_y, -floor_size], normal: [0.0, 1.0, 0.0], color: floor_col });
        vertices.push(TankVertex { position: [ floor_size, floor_y,  floor_size], normal: [0.0, 1.0, 0.0], color: floor_col });
        vertices.push(TankVertex { position: [-floor_size, floor_y,  floor_size], normal: [0.0, 1.0, 0.0], color: floor_col });
        indices.extend_from_slice(&[base_idx, base_idx + 1, base_idx + 2, base_idx, base_idx + 2, base_idx + 3]);

        let min = Vec3::new(-1.0, 0.0, -0.6);
        let max = Vec3::new(1.0, 1.4, 0.6);
        let frame_col = [0.45, 0.55, 0.70, 0.85];
        let glass_col = [0.15, 0.35, 0.55, 0.12];

        Self::add_quad(&mut vertices, &mut indices,
            Vec3::new(min.x, min.y + 0.001, min.z),
            Vec3::new(max.x, min.y + 0.001, min.z),
            Vec3::new(max.x, min.y + 0.001, max.z),
            Vec3::new(min.x, min.y + 0.001, max.z),
            Vec3::Y, [0.2, 0.4, 0.6, 0.25]
        );

        Self::add_quad(&mut vertices, &mut indices,
            Vec3::new(min.x, min.y, min.z),
            Vec3::new(max.x, min.y, min.z),
            Vec3::new(max.x, max.y, min.z),
            Vec3::new(min.x, max.y, min.z),
            Vec3::Z, glass_col
        );

        Self::add_quad(&mut vertices, &mut indices,
            Vec3::new(min.x, min.y, max.z),
            Vec3::new(min.x, min.y, min.z),
            Vec3::new(min.x, max.y, min.z),
            Vec3::new(min.x, max.y, max.z),
            Vec3::X, glass_col
        );

        Self::add_quad(&mut vertices, &mut indices,
            Vec3::new(max.x, min.y, min.z),
            Vec3::new(max.x, min.y, max.z),
            Vec3::new(max.x, max.y, max.z),
            Vec3::new(max.x, max.y, min.z),
            -Vec3::X, glass_col
        );

        Self::add_quad(&mut vertices, &mut indices,
            Vec3::new(max.x, min.y, max.z),
            Vec3::new(min.x, min.y, max.z),
            Vec3::new(min.x, max.y, max.z),
            Vec3::new(max.x, max.y, max.z),
            -Vec3::Z, [0.15, 0.35, 0.55, 0.08]
        );

        let r = 0.015;
        let corners = [
            (Vec3::new(min.x, min.y, min.z), Vec3::new(max.x, min.y, min.z)),
            (Vec3::new(max.x, min.y, min.z), Vec3::new(max.x, min.y, max.z)),
            (Vec3::new(max.x, min.y, max.z), Vec3::new(min.x, min.y, max.z)),
            (Vec3::new(min.x, min.y, max.z), Vec3::new(min.x, min.y, min.z)),
            (Vec3::new(min.x, max.y, min.z), Vec3::new(max.x, max.y, min.z)),
            (Vec3::new(max.x, max.y, min.z), Vec3::new(max.x, max.y, max.z)),
            (Vec3::new(max.x, max.y, max.z), Vec3::new(min.x, max.y, max.z)),
            (Vec3::new(min.x, max.y, max.z), Vec3::new(min.x, max.y, min.z)),
            (Vec3::new(min.x, min.y, min.z), Vec3::new(min.x, max.y, min.z)),
            (Vec3::new(max.x, min.y, min.z), Vec3::new(max.x, max.y, min.z)),
            (Vec3::new(max.x, min.y, max.z), Vec3::new(max.x, max.y, max.z)),
            (Vec3::new(min.x, min.y, max.z), Vec3::new(min.x, max.y, max.z)),
        ];

        for (p0, p1) in corners {
            Self::add_beam(&mut vertices, &mut indices, p0, p1, r, frame_col);
        }

        (vertices, indices)
    }

    #[allow(clippy::too_many_arguments)]
    fn add_quad(
        vertices: &mut Vec<TankVertex>,
        indices: &mut Vec<u32>,
        p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3,
        normal: Vec3,
        color: [f32; 4],
    ) {
        let base = vertices.len() as u32;
        let n = [normal.x, normal.y, normal.z];
        vertices.push(TankVertex { position: [p0.x, p0.y, p0.z], normal: n, color });
        vertices.push(TankVertex { position: [p1.x, p1.y, p1.z], normal: n, color });
        vertices.push(TankVertex { position: [p2.x, p2.y, p2.z], normal: n, color });
        vertices.push(TankVertex { position: [p3.x, p3.y, p3.z], normal: n, color });
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    fn add_beam(
        vertices: &mut Vec<TankVertex>,
        indices: &mut Vec<u32>,
        p0: Vec3, p1: Vec3,
        radius: f32,
        color: [f32; 4],
    ) {
        let dir = (p1 - p0).normalize();
        let up = if dir.y.abs() > 0.99 { Vec3::X } else { Vec3::Y };
        let side = dir.cross(up).normalize() * radius;
        let norm_up = side.cross(dir).normalize() * radius;

        let corners = [
            p0 - side - norm_up,
            p0 + side - norm_up,
            p0 + side + norm_up,
            p0 - side + norm_up,
            p1 - side - norm_up,
            p1 + side - norm_up,
            p1 + side + norm_up,
            p1 - side + norm_up,
        ];

        Self::add_quad(vertices, indices, corners[0], corners[1], corners[5], corners[4], -norm_up, color);
        Self::add_quad(vertices, indices, corners[1], corners[2], corners[6], corners[5], side, color);
        Self::add_quad(vertices, indices, corners[2], corners[3], corners[7], corners[6], norm_up, color);
        Self::add_quad(vertices, indices, corners[3], corners[0], corners[4], corners[7], -side, color);
    }

    pub fn update_obstacles(&mut self, queue: &wgpu::Queue, obstacles: &[Obstacle]) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for obs in obstacles {
            match obs.kind {
                ObstacleKind::Sphere { radius } => {
                    let col = if obs.is_grabbed {
                        [1.0, 0.8, 0.2, 1.0]
                    } else {
                        [0.85, 0.45, 0.25, 1.0]
                    };
                    Self::generate_sphere(&mut vertices, &mut indices, obs.position, radius, col, 24, 16);
                }
                ObstacleKind::Rotor {
                    radius,
                    height,
                    current_angle,
                    num_blades,
                    ..
                } => {
                    let col = [0.3, 0.7, 0.9, 1.0];
                    Self::generate_cylinder(&mut vertices, &mut indices, obs.position, 0.05, height, [0.2, 0.2, 0.3, 1.0], 16);
                    let blade_step = std::f32::consts::TAU / num_blades as f32;
                    for b in 0..num_blades {
                        let ang = current_angle + b as f32 * blade_step;
                        let dir = Vec3::new(ang.cos(), 0.0, ang.sin());
                        let blade_end = obs.position + dir * radius;
                        Self::add_beam(&mut vertices, &mut indices, obs.position, blade_end, 0.02, col);
                    }
                }
                ObstacleKind::BoxCollider { half_extents } => {
                    let col = [0.6, 0.6, 0.7, 1.0];
                    let min = obs.position - half_extents;
                    let max = obs.position + half_extents;
                    Self::add_quad(&mut vertices, &mut indices, Vec3::new(min.x, min.y, min.z), Vec3::new(max.x, min.y, min.z), Vec3::new(max.x, max.y, min.z), Vec3::new(min.x, max.y, min.z), -Vec3::Z, col);
                    Self::add_quad(&mut vertices, &mut indices, Vec3::new(max.x, min.y, min.z), Vec3::new(max.x, min.y, max.z), Vec3::new(max.x, max.y, max.z), Vec3::new(max.x, max.y, min.z), Vec3::X, col);
                    Self::add_quad(&mut vertices, &mut indices, Vec3::new(max.x, min.y, max.z), Vec3::new(min.x, min.y, max.z), Vec3::new(min.x, max.y, max.z), Vec3::new(max.x, max.y, max.z), Vec3::Z, col);
                    Self::add_quad(&mut vertices, &mut indices, Vec3::new(min.x, min.y, max.z), Vec3::new(min.x, min.y, min.z), Vec3::new(min.x, max.y, min.z), Vec3::new(min.x, max.y, max.z), -Vec3::X, col);
                    Self::add_quad(&mut vertices, &mut indices, Vec3::new(min.x, max.y, min.z), Vec3::new(max.x, max.y, min.z), Vec3::new(max.x, max.y, max.z), Vec3::new(min.x, max.y, max.z), Vec3::Y, col);
                }
            }
        }

        self.obstacle_index_count = indices.len() as u32;
        if !vertices.is_empty() {
            queue.write_buffer(&self.obstacle_vertex_buffer, 0, bytemuck::cast_slice(&vertices));
            queue.write_buffer(&self.obstacle_index_buffer, 0, bytemuck::cast_slice(&indices));
        }
    }

    fn generate_sphere(
        vertices: &mut Vec<TankVertex>,
        indices: &mut Vec<u32>,
        center: Vec3,
        radius: f32,
        color: [f32; 4],
        segments: u32,
        rings: u32,
    ) {
        let base_idx = vertices.len() as u32;
        for r in 0..=rings {
            let theta = r as f32 * std::f32::consts::PI / rings as f32;
            let sin_t = theta.sin();
            let cos_t = theta.cos();
            for s in 0..=segments {
                let phi = s as f32 * std::f32::consts::TAU / segments as f32;
                let sin_p = phi.sin();
                let cos_p = phi.cos();

                let n = Vec3::new(sin_t * cos_p, cos_t, sin_t * sin_p);
                let p = center + n * radius;
                vertices.push(TankVertex {
                    position: [p.x, p.y, p.z],
                    normal: [n.x, n.y, n.z],
                    color,
                });
            }
        }

        for r in 0..rings {
            for s in 0..segments {
                let first = base_idx + (r * (segments + 1) + s);
                let second = first + segments + 1;
                indices.extend_from_slice(&[first, second, first + 1, second, second + 1, first + 1]);
            }
        }
    }

    fn generate_cylinder(
        vertices: &mut Vec<TankVertex>,
        indices: &mut Vec<u32>,
        center: Vec3,
        radius: f32,
        height: f32,
        color: [f32; 4],
        segments: u32,
    ) {
        let base_idx = vertices.len() as u32;
        let half_h = height * 0.5;
        for s in 0..=segments {
            let phi = s as f32 * std::f32::consts::TAU / segments as f32;
            let n = Vec3::new(phi.cos(), 0.0, phi.sin());
            let b = center + n * radius - Vec3::new(0.0, half_h, 0.0);
            let t = center + n * radius + Vec3::new(0.0, half_h, 0.0);
            vertices.push(TankVertex { position: [b.x, b.y, b.z], normal: [n.x, 0.0, n.z], color });
            vertices.push(TankVertex { position: [t.x, t.y, t.z], normal: [n.x, 0.0, n.z], color });
        }
        for s in 0..segments {
            let i = base_idx + s * 2;
            indices.extend_from_slice(&[i, i + 1, i + 2, i + 1, i + 3, i + 2]);
        }
    }

    pub fn render<'a>(
        &'a self,
        rpass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
    ) {
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, camera_bind_group, &[]);

        rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        rpass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(0..self.index_count, 0, 0..1);

        if self.obstacle_index_count > 0 {
            rpass.set_vertex_buffer(0, self.obstacle_vertex_buffer.slice(..));
            rpass.set_index_buffer(self.obstacle_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..self.obstacle_index_count, 0, 0..1);
        }
    }
}
