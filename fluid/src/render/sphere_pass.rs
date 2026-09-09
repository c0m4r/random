use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use crate::physics::particle::ParticleGpu;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct RenderParamsGpu {
    pub particle_radius: f32,
    pub render_mode: u32,
    pub color_theme: u32,
    pub max_velocity_color: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct QuadVertex {
    pub pos: [f32; 2],
}

pub struct SpherePass {
    pub pipeline: wgpu::RenderPipeline,
    pub quad_vertex_buffer: wgpu::Buffer,
    pub render_params_buffer: wgpu::Buffer,
    pub render_params_bind_group: wgpu::BindGroup,
    pub render_params_layout: wgpu::BindGroupLayout,
}

impl SpherePass {
    pub fn new(
        device: &wgpu::Device,
        _camera_bind_group_layout: &wgpu::BindGroupLayout,
        surface_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sphere Particle Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sphere.wgsl").into()),
        });

        let render_params_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Render Params Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let initial_params = RenderParamsGpu {
            particle_radius: 0.022,
            render_mode: 0,
            color_theme: 0,
            max_velocity_color: 4.5,
        };

        let render_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Render Params Buffer"),
            contents: bytemuck::bytes_of(&initial_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sphere Pipeline Layout"),
            bind_group_layouts: &[Some(&render_params_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sphere Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<QuadVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 0,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                        ],
                    }),
                    Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<ParticleGpu>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 1,
                                format: wgpu::VertexFormat::Float32x3,
                            },
                            wgpu::VertexAttribute {
                                offset: 12,
                                shader_location: 2,
                                format: wgpu::VertexFormat::Float32,
                            },
                            wgpu::VertexAttribute {
                                offset: 16,
                                shader_location: 3,
                                format: wgpu::VertexFormat::Float32x3,
                            },
                            wgpu::VertexAttribute {
                                offset: 28,
                                shader_location: 4,
                                format: wgpu::VertexFormat::Float32,
                            },
                            wgpu::VertexAttribute {
                                offset: 32,
                                shader_location: 5,
                                format: wgpu::VertexFormat::Float32x4,
                            },
                        ],
                    }),
                ],
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
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let quad_vertices = [
            QuadVertex { pos: [-1.0, -1.0] },
            QuadVertex { pos: [ 1.0, -1.0] },
            QuadVertex { pos: [ 1.0,  1.0] },
            QuadVertex { pos: [-1.0, -1.0] },
            QuadVertex { pos: [ 1.0,  1.0] },
            QuadVertex { pos: [-1.0,  1.0] },
        ];

        let quad_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Quad Vertex Buffer"),
            contents: bytemuck::cast_slice(&quad_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let render_params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Params Bind Group"),
            layout: &render_params_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: render_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: render_params_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            pipeline,
            quad_vertex_buffer,
            render_params_buffer,
            render_params_bind_group,
            render_params_layout,
        }
    }

    pub fn update_params(
        &self,
        queue: &wgpu::Queue,
        particle_radius: f32,
        render_mode: u32,
        color_theme: u32,
        max_vel: f32,
    ) {
        let params = RenderParamsGpu {
            particle_radius,
            render_mode,
            color_theme,
            max_velocity_color: max_vel,
        };
        queue.write_buffer(&self.render_params_buffer, 0, bytemuck::bytes_of(&params));
    }

    pub fn create_combined_bind_group(
        &mut self,
        device: &wgpu::Device,
        camera_buffer: &wgpu::Buffer,
    ) {
        self.render_params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Combined Camera & Render Params Bind Group"),
            layout: &self.render_params_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.render_params_buffer.as_entire_binding(),
                },
            ],
        });
    }

    pub fn render<'a>(
        &'a self,
        rpass: &mut wgpu::RenderPass<'a>,
        particle_buffer_slice: wgpu::BufferSlice<'a>,
        particle_count: u32,
    ) {
        if particle_count == 0 {
            return;
        }
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.render_params_bind_group, &[]);
        rpass.set_vertex_buffer(0, self.quad_vertex_buffer.slice(..));
        rpass.set_vertex_buffer(1, particle_buffer_slice);
        rpass.draw(0..6, 0..particle_count);
    }
}
