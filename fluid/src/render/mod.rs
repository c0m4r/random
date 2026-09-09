pub mod tank_pass;
pub mod sphere_pass;
pub mod liquid_pass;

use std::sync::Arc;
use wgpu::util::DeviceExt;
use glam::Vec2;

use crate::math::Camera;
use crate::physics::obstacle::Obstacle;
use self::tank_pass::TankPass;
use self::sphere_pass::SpherePass;
use self::liquid_pass::LiquidPass;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RenderStyle {
    RealisticLiquid,
    ShadedSpheres,
    VelocityHeatmap,
    PressureHeatmap,
}

impl RenderStyle {
    pub const ALL: [RenderStyle; 4] = [
        RenderStyle::RealisticLiquid,
        RenderStyle::ShadedSpheres,
        RenderStyle::VelocityHeatmap,
        RenderStyle::PressureHeatmap,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            RenderStyle::RealisticLiquid => "Realistic Glassy Fluid (Screen-Space)",
            RenderStyle::ShadedSpheres => "3D Glossy Spheres (Raytraced)",
            RenderStyle::VelocityHeatmap => "Velocity Heatmap (Scientific)",
            RenderStyle::PressureHeatmap => "Pressure / Density Field",
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ColorTheme {
    OceanAzure,
    EmeraldToxic,
    GoldenHoney,
    LavaAmber,
    MercurySilver,
    PlasmaPurple,
}

impl ColorTheme {
    pub const ALL: [ColorTheme; 6] = [
        ColorTheme::OceanAzure,
        ColorTheme::EmeraldToxic,
        ColorTheme::GoldenHoney,
        ColorTheme::LavaAmber,
        ColorTheme::MercurySilver,
        ColorTheme::PlasmaPurple,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            ColorTheme::OceanAzure => "Ocean Azure Water",
            ColorTheme::EmeraldToxic => "Biohazard Emerald",
            ColorTheme::GoldenHoney => "Golden Viscous Honey",
            ColorTheme::LavaAmber => "Molten Lava Amber",
            ColorTheme::MercurySilver => "Quicksilver (Liquid Metal)",
            ColorTheme::PlasmaPurple => "Neon Plasma Violet",
        }
    }

    pub fn rgba(&self) -> [f32; 4] {
        match self {
            ColorTheme::OceanAzure => [0.15, 0.60, 0.95, 1.0],
            ColorTheme::EmeraldToxic => [0.10, 0.92, 0.45, 1.0],
            ColorTheme::GoldenHoney => [0.95, 0.68, 0.05, 1.0],
            ColorTheme::LavaAmber => [0.98, 0.28, 0.08, 1.0],
            ColorTheme::MercurySilver => [0.85, 0.88, 0.92, 1.0],
            ColorTheme::PlasmaPurple => [0.75, 0.20, 0.92, 1.0],
        }
    }

    pub fn ior(&self) -> f32 {
        match self {
            ColorTheme::OceanAzure => 1.333,
            ColorTheme::EmeraldToxic => 1.38,
            ColorTheme::GoldenHoney => 1.52,
            ColorTheme::LavaAmber => 1.45,
            ColorTheme::MercurySilver => 2.40,
            ColorTheme::PlasmaPurple => 1.25,
        }
    }
}

pub struct Renderer {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub surface_format: wgpu::TextureFormat,
    pub depth_format: wgpu::TextureFormat,

    pub camera_buffer: wgpu::Buffer,
    pub camera_bind_group: wgpu::BindGroup,
    pub camera_bind_group_layout: wgpu::BindGroupLayout,

    pub tank_pass: TankPass,
    pub sphere_pass: SpherePass,
    pub liquid_pass: LiquidPass,

    pub depth_texture: Option<wgpu::Texture>,
    pub depth_view: Option<wgpu::TextureView>,
    pub scene_color_tex: Option<wgpu::Texture>,
    pub scene_color_view: Option<wgpu::TextureView>,

    pub render_style: RenderStyle,
    pub color_theme: ColorTheme,
    pub particle_radius_scale: f32,
    pub max_velocity_color: f32,
    pub width: u32,
    pub height: u32,
}

impl Renderer {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        surface_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Camera Bind Group Layout"),
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
            ],
        });

        let dummy_cam = Camera::default().build_uniforms(Vec2::new(width as f32, height as f32));
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Uniform Buffer"),
            contents: bytemuck::bytes_of(&dummy_cam),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
            ],
        });

        let tank_pass = TankPass::new(&device, &camera_bind_group_layout, surface_format, depth_format);
        let mut sphere_pass = SpherePass::new(&device, &camera_bind_group_layout, surface_format, depth_format);
        sphere_pass.create_combined_bind_group(&device, &camera_buffer);

        let liquid_pass = LiquidPass::new(&device, &sphere_pass.render_params_layout, surface_format, depth_format);

        let mut renderer = Self {
            device,
            queue,
            surface_format,
            depth_format,
            camera_buffer,
            camera_bind_group,
            camera_bind_group_layout,
            tank_pass,
            sphere_pass,
            liquid_pass,
            depth_texture: None,
            depth_view: None,
            scene_color_tex: None,
            scene_color_view: None,
            render_style: RenderStyle::RealisticLiquid,
            color_theme: ColorTheme::OceanAzure,
            particle_radius_scale: 1.0,
            max_velocity_color: 4.5,
            width: 0,
            height: 0,
        };

        renderer.resize(width, height);
        renderer
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.width = width;
        self.height = height;

        let depth_desc = wgpu::TextureDescriptor {
            label: Some("Scene Depth Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.depth_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let depth_tex = self.device.create_texture(&depth_desc);
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let color_desc = wgpu::TextureDescriptor {
            label: Some("Scene Color Offscreen Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let color_tex = self.device.create_texture(&color_desc);
        let color_view = color_tex.create_view(&wgpu::TextureViewDescriptor::default());

        self.liquid_pass.resize(&self.device, width, height, &color_view, &self.camera_buffer);

        self.depth_texture = Some(depth_tex);
        self.depth_view = Some(depth_view);
        self.scene_color_tex = Some(color_tex);
        self.scene_color_view = Some(color_view);
    }

    pub fn update_camera(&self, camera: &Camera) {
        let uniforms = camera.build_uniforms(Vec2::new(self.width as f32, self.height as f32));
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    pub fn update_obstacles(&mut self, obstacles: &[Obstacle]) {
        self.tank_pass.update_obstacles(&self.queue, obstacles);
    }

    pub fn render(
        &mut self,
        target_view: &wgpu::TextureView,
        particle_buffer: &wgpu::Buffer,
        particle_count: u32,
        particle_base_radius: f32,
    ) {
        let depth_view = match &self.depth_view {
            Some(v) => v,
            None => return,
        };
        let scene_color_view = match &self.scene_color_view {
            Some(v) => v,
            None => return,
        };

        let radius = particle_base_radius * self.particle_radius_scale;
        let mode_id = match self.render_style {
            RenderStyle::RealisticLiquid => 0,
            RenderStyle::ShadedSpheres => 0,
            RenderStyle::VelocityHeatmap => 1,
            RenderStyle::PressureHeatmap => 2,
        };

        self.sphere_pass.update_params(&self.queue, radius, mode_id, 0, self.max_velocity_color);
        self.liquid_pass.update_material(
            &self.queue,
            self.color_theme.rgba(),
            self.color_theme.ior(),
            0.05,
            4.0,
            0.85,
        );

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Fluid Master Render Encoder"),
        });

        match self.render_style {
            RenderStyle::ShadedSpheres | RenderStyle::VelocityHeatmap | RenderStyle::PressureHeatmap => {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Forward Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.06, g: 0.07, b: 0.09, a: 1.0 }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });

                self.sphere_pass.render(&mut rpass, particle_buffer.slice(..), particle_count);
                self.tank_pass.render(&mut rpass, &self.camera_bind_group);
            }
            RenderStyle::RealisticLiquid => {
                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Scene Offscreen Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: scene_color_view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.06, g: 0.07, b: 0.09, a: 1.0 }),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    self.tank_pass.render(&mut rpass, &self.camera_bind_group);
                }

                if let (Some(l_depth_view), Some(blur_temp_view), Some(blur_final_view), Some(blur_h_bg), Some(blur_v_bg), Some(comp_tex_bg), Some(comp_cam_bg)) = (
                    &self.liquid_pass.depth_view,
                    &self.liquid_pass.blur_temp_view,
                    &self.liquid_pass.blur_final_view,
                    &self.liquid_pass.blur_h_bind_group,
                    &self.liquid_pass.blur_v_bind_group,
                    &self.liquid_pass.composite_tex_bind_group,
                    &self.liquid_pass.composite_camera_bind_group,
                ) {
                    {
                        let mut dpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("Liquid Depth Pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: l_depth_view,
                                depth_slice: None,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                view: depth_view,
                                depth_ops: Some(wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(1.0),
                                    store: wgpu::StoreOp::Store,
                                }),
                                stencil_ops: None,
                            }),
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        });
                        dpass.set_pipeline(&self.liquid_pass.depth_pipeline);
                        dpass.set_bind_group(0, &self.sphere_pass.render_params_bind_group, &[]);
                        dpass.set_vertex_buffer(0, self.liquid_pass.quad_vertex_buffer.slice(..));
                        dpass.set_vertex_buffer(1, particle_buffer.slice(..));
                        dpass.draw(0..6, 0..particle_count);
                    }

                    {
                        let mut bpass_h = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("Bilateral Blur H Pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: blur_temp_view,
                                depth_slice: None,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        });
                        bpass_h.set_pipeline(&self.liquid_pass.blur_pipeline);
                        bpass_h.set_bind_group(0, blur_h_bg, &[]);
                        bpass_h.draw(0..3, 0..1);
                    }

                    {
                        let mut bpass_v = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("Bilateral Blur V Pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: blur_final_view,
                                depth_slice: None,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        });
                        bpass_v.set_pipeline(&self.liquid_pass.blur_pipeline);
                        bpass_v.set_bind_group(0, blur_v_bg, &[]);
                        bpass_v.draw(0..3, 0..1);
                    }

                    {
                        let mut cpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("Final Composite Pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: target_view,
                                depth_slice: None,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.06, g: 0.07, b: 0.09, a: 1.0 }),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        });
                        cpass.set_pipeline(&self.liquid_pass.composite_pipeline);
                        cpass.set_bind_group(0, comp_cam_bg, &[]);
                        cpass.set_bind_group(1, comp_tex_bg, &[]);
                        cpass.draw(0..3, 0..1);
                    }
                }
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    pub fn render_to_image(
        &mut self,
        particle_buffer: &wgpu::Buffer,
        particle_count: u32,
        particle_base_radius: f32,
        output_path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let width = self.width;
        let height = self.height;

        let render_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Screenshot Target"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let render_view = render_tex.create_view(&wgpu::TextureViewDescriptor::default());

        self.render(&render_view, particle_buffer, particle_count, particle_base_radius);

        let bytes_per_pixel = 4u32;
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;

        let staging_size = (padded_bytes_per_row * height) as u64;
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Screenshot Staging Buffer"),
            size: staging_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Screenshot Copy Encoder"),
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &render_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()??;

        let mapped_range = buffer_slice.get_mapped_range();
        let padded_data = mapped_range.as_deref().map_err(|e| format!("MapRangeError: {:?}", e))?;
        let mut image_data = Vec::with_capacity((width * height * 4) as usize);

        for row in 0..height {
            let start = (row * padded_bytes_per_row) as usize;
            let end = start + (unpadded_bytes_per_row) as usize;
            let row_bytes = &padded_data[start..end];
            // Swap BGRA to RGBA if format is Bgra8
            if self.surface_format == wgpu::TextureFormat::Bgra8Unorm || self.surface_format == wgpu::TextureFormat::Bgra8UnormSrgb {
                for chunk in row_bytes.as_chunks::<4>().0 {
                    image_data.push(chunk[2]);
                    image_data.push(chunk[1]);
                    image_data.push(chunk[0]);
                    image_data.push(chunk[3]);
                }
            } else {
                image_data.extend_from_slice(row_bytes);
            }
        }

        drop(mapped_range);
        staging_buffer.unmap();

        let img = image::RgbaImage::from_raw(width, height, image_data)
            .ok_or("Failed to create image from raw buffer")?;
        img.save(output_path)?;
        println!("Screenshot successfully saved to: {}", output_path);
        Ok(())
    }
}
