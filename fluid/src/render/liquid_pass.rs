use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use crate::physics::particle::ParticleGpu;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct LiquidMaterialGpu {
    pub liquid_color: [f32; 4],
    pub refraction_ior: f32,
    pub roughness: f32,
    pub fresnel_power: f32,
    pub specular_intensity: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct BlurUniformsGpu {
    pub blur_dir: [f32; 2],
    pub filter_radius: i32,
    pub depth_threshold: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct QuadVertex {
    pub pos: [f32; 2],
}

pub struct LiquidPass {
    pub depth_pipeline: wgpu::RenderPipeline,
    pub blur_pipeline: wgpu::RenderPipeline,
    pub composite_pipeline: wgpu::RenderPipeline,

    pub quad_vertex_buffer: wgpu::Buffer,
    pub material_buffer: wgpu::Buffer,
    pub blur_h_buffer: wgpu::Buffer,
    pub blur_v_buffer: wgpu::Buffer,

    pub linear_sampler: wgpu::Sampler,
    pub nearest_sampler: wgpu::Sampler,

    pub depth_tex: Option<wgpu::Texture>,
    pub depth_view: Option<wgpu::TextureView>,
    pub blur_temp_tex: Option<wgpu::Texture>,
    pub blur_temp_view: Option<wgpu::TextureView>,
    pub blur_final_tex: Option<wgpu::Texture>,
    pub blur_final_view: Option<wgpu::TextureView>,

    pub blur_h_bind_group: Option<wgpu::BindGroup>,
    pub blur_v_bind_group: Option<wgpu::BindGroup>,
    pub composite_tex_bind_group: Option<wgpu::BindGroup>,

    pub blur_bind_group_layout: wgpu::BindGroupLayout,
    pub composite_tex_layout: wgpu::BindGroupLayout,
    pub composite_camera_layout: wgpu::BindGroupLayout,
    pub composite_camera_bind_group: Option<wgpu::BindGroup>,

    pub width: u32,
    pub height: u32,
}

impl LiquidPass {
    pub fn new(
        device: &wgpu::Device,
        depth_bind_group_layout: &wgpu::BindGroupLayout,
        surface_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> Self {
        let depth_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SS Depth Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ss_depth.wgsl").into()),
        });
        let blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SS Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ss_blur.wgsl").into()),
        });
        let comp_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SS Composite Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ss_composite.wgsl").into()),
        });

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Linear Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Nearest Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // 1. Depth Pipeline
        let depth_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SS Depth Pipeline Layout"),
            bind_group_layouts: &[Some(depth_bind_group_layout)],
            immediate_size: 0,
        });

        let depth_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SS Depth Pipeline"),
            layout: Some(&depth_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &depth_shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<QuadVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        }],
                    }),
                    Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<ParticleGpu>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute { offset: 0, shader_location: 1, format: wgpu::VertexFormat::Float32x3 },
                            wgpu::VertexAttribute { offset: 12, shader_location: 2, format: wgpu::VertexFormat::Float32 },
                            wgpu::VertexAttribute { offset: 16, shader_location: 3, format: wgpu::VertexFormat::Float32x3 },
                            wgpu::VertexAttribute { offset: 28, shader_location: 4, format: wgpu::VertexFormat::Float32 },
                            wgpu::VertexAttribute { offset: 32, shader_location: 5, format: wgpu::VertexFormat::Float32x4 },
                        ],
                    }),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &depth_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // 2. Bilateral Blur Pipeline
        let blur_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Blur Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let blur_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Blur Pipeline Layout"),
            bind_group_layouts: &[Some(&blur_bind_group_layout)],
            immediate_size: 0,
        });

        let blur_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blur Pipeline"),
            layout: Some(&blur_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blur_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blur_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // 3. Composite Pipeline
        let composite_camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Composite Camera Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let composite_tex_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Composite Tex Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let composite_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Composite Pipeline Layout"),
            bind_group_layouts: &[Some(&composite_camera_layout), Some(&composite_tex_layout)],
            immediate_size: 0,
        });

        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Composite Pipeline"),
            layout: Some(&composite_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &comp_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &comp_shader,
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
                ..Default::default()
            },
            depth_stencil: None,
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
            label: Some("SS Liquid Quad Buffer"),
            contents: bytemuck::cast_slice(&quad_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let initial_mat = LiquidMaterialGpu {
            liquid_color: [0.15, 0.65, 0.95, 1.0],
            refraction_ior: 1.333,
            roughness: 0.05,
            fresnel_power: 4.0,
            specular_intensity: 0.8,
        };
        let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Liquid Material Buffer"),
            contents: bytemuck::bytes_of(&initial_mat),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let blur_h = BlurUniformsGpu { blur_dir: [1.0, 0.0], filter_radius: 5, depth_threshold: 0.08 };
        let blur_v = BlurUniformsGpu { blur_dir: [0.0, 1.0], filter_radius: 5, depth_threshold: 0.08 };

        let blur_h_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Blur H Buffer"),
            contents: bytemuck::bytes_of(&blur_h),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let blur_v_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Blur V Buffer"),
            contents: bytemuck::bytes_of(&blur_v),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        Self {
            depth_pipeline,
            blur_pipeline,
            composite_pipeline,
            quad_vertex_buffer,
            material_buffer,
            blur_h_buffer,
            blur_v_buffer,
            linear_sampler,
            nearest_sampler,
            depth_tex: None,
            depth_view: None,
            blur_temp_tex: None,
            blur_temp_view: None,
            blur_final_tex: None,
            blur_final_view: None,
            blur_h_bind_group: None,
            blur_v_bind_group: None,
            composite_tex_bind_group: None,
            blur_bind_group_layout,
            composite_tex_layout,
            composite_camera_layout,
            composite_camera_bind_group: None,
            width: 0,
            height: 0,
        }
    }

    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        scene_color_view: &wgpu::TextureView,
        camera_buffer: &wgpu::Buffer,
    ) {
        if width == 0 || height == 0 {
            return;
        }
        self.width = width;
        self.height = height;

        let depth_desc = wgpu::TextureDescriptor {
            label: Some("SS Liquid Depth Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };

        let depth_tex = device.create_texture(&depth_desc);
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let blur_temp_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SS Blur Temp Texture"),
            ..depth_desc
        });
        let blur_temp_view = blur_temp_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let blur_final_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SS Blur Final Texture"),
            ..depth_desc
        });
        let blur_final_view = blur_final_tex.create_view(&wgpu::TextureViewDescriptor::default());

        self.blur_h_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blur H Bind Group"),
            layout: &self.blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&depth_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.linear_sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: self.blur_h_buffer.as_entire_binding() },
            ],
        }));

        self.blur_v_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blur V Bind Group"),
            layout: &self.blur_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&blur_temp_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.linear_sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: self.blur_v_buffer.as_entire_binding() },
            ],
        }));

        self.composite_tex_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Composite Tex Bind Group"),
            layout: &self.composite_tex_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&blur_final_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(scene_color_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.linear_sampler) },
            ],
        }));

        self.composite_camera_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Composite Camera Bind Group"),
            layout: &self.composite_camera_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.material_buffer.as_entire_binding() },
            ],
        }));

        self.depth_tex = Some(depth_tex);
        self.depth_view = Some(depth_view);
        self.blur_temp_tex = Some(blur_temp_tex);
        self.blur_temp_view = Some(blur_temp_view);
        self.blur_final_tex = Some(blur_final_tex);
        self.blur_final_view = Some(blur_final_view);
    }

    pub fn update_material(
        &self,
        queue: &wgpu::Queue,
        color: [f32; 4],
        refraction_ior: f32,
        roughness: f32,
        fresnel_power: f32,
        specular_intensity: f32,
    ) {
        let mat = LiquidMaterialGpu {
            liquid_color: color,
            refraction_ior,
            roughness,
            fresnel_power,
            specular_intensity,
        };
        queue.write_buffer(&self.material_buffer, 0, bytemuck::bytes_of(&mat));
    }
}
