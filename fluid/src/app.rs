use std::sync::Arc;
use std::time::Instant;
use glam::{Vec2, Vec3};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

use crate::math::Camera;
use crate::physics::{
    cpu_sph::CpuSph,
    gpu_sph::GpuSph,
    obstacle::Obstacle,
    scenarios::ScenarioType,
    EngineType, PhysicsEngine, SimParams, StepMetrics,
};
use crate::render::Renderer;
use crate::ui::{GuiState, MouseInteractionMode};

pub struct AppState {
    pub window: Arc<Window>,
    pub surface: wgpu::Surface<'static>,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub renderer: Renderer,

    pub gpu_engine: Option<GpuSph>,
    pub cpu_engine: Option<CpuSph>,
    pub current_engine: EngineType,
    pub params: SimParams,
    pub obstacles: Vec<Obstacle>,

    pub camera: Camera,
    pub gui_state: GuiState,
    pub egui_ctx: egui::Context,
    pub egui_state: egui_winit::State,
    pub egui_renderer: egui_wgpu::Renderer,

    pub last_frame_time: Instant,
    pub mouse_pos: Vec2,
    pub is_mouse_left_down: bool,
    pub is_mouse_right_down: bool,
    pub is_mouse_middle_down: bool,
    pub is_shift_down: bool,
    pub is_ctrl_down: bool,

    pub grabbed_obstacle_idx: Option<usize>,
}

impl AppState {
    pub fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).expect("Failed to create window surface");

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            ..Default::default()
        })).expect("No suitable GPU adapter found");

        println!("Selected GPU: {} ({:?})", adapter.get_info().name, adapter.get_info().backend);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Aether Device"),
                ..Default::default()
            },
        )).expect("Failed to create wgpu device");

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width,
            height,
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &surface_config);

        let depth_format = wgpu::TextureFormat::Depth32Float;
        let renderer = Renderer::new(
            device.clone(),
            queue.clone(),
            surface_format,
            depth_format,
            width,
            height,
        );

        let camera = Camera {
            aspect_ratio: width as f32 / height as f32,
            ..Default::default()
        };

        let initial_scenario = ScenarioType::DamBreak;
        let initial_particles = 25_000;

        let (gpu_engine, params, obstacles) = GpuSph::new(
            device.clone(),
            queue.clone(),
            initial_scenario,
            initial_particles,
        );

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        let egui_renderer = egui_wgpu::Renderer::new(
            &device,
            surface_format,
            egui_wgpu::RendererOptions::default(),
        );

        Self {
            window,
            surface,
            surface_config,
            device,
            queue,
            renderer,
            gpu_engine: Some(gpu_engine),
            cpu_engine: None,
            current_engine: EngineType::GpuCompute,
            params,
            obstacles,
            camera,
            gui_state: GuiState::default(),
            egui_ctx,
            egui_state,
            egui_renderer,
            last_frame_time: Instant::now(),
            mouse_pos: Vec2::ZERO,
            is_mouse_left_down: false,
            is_mouse_right_down: false,
            is_mouse_middle_down: false,
            is_shift_down: false,
            is_ctrl_down: false,
            grabbed_obstacle_idx: None,
        }
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.surface_config.width = new_size.width;
        self.surface_config.height = new_size.height;
        self.surface.configure(&self.device, &self.surface_config);

        self.renderer.resize(new_size.width, new_size.height);
        self.camera.aspect_ratio = new_size.width as f32 / new_size.height as f32;
    }

    pub fn switch_engine(&mut self, target_engine: EngineType) {
        if self.current_engine == target_engine {
            return;
        }
        self.current_engine = target_engine;
        let p_count = self.gui_state.target_particles;
        let scenario = self.gui_state.scenario;

        match target_engine {
            EngineType::GpuCompute => {
                let (gpu, params, obs) = GpuSph::new(
                    self.device.clone(),
                    self.queue.clone(),
                    scenario,
                    p_count,
                );
                self.gpu_engine = Some(gpu);
                self.params = params;
                self.obstacles = obs;
                self.cpu_engine = None;
            }
            EngineType::CpuMultiThread => {
                let (cpu, params, obs) = CpuSph::new(scenario, p_count);
                self.cpu_engine = Some(cpu);
                self.params = params;
                self.obstacles = obs;
            }
        }
    }

    pub fn reset_simulation(&mut self) {
        let scenario = self.gui_state.scenario;
        let p_count = self.gui_state.target_particles;

        match self.current_engine {
            EngineType::GpuCompute => {
                if let Some(gpu) = &mut self.gpu_engine {
                    gpu.reset(scenario, p_count, &mut self.params, &mut self.obstacles);
                }
            }
            EngineType::CpuMultiThread => {
                if let Some(cpu) = &mut self.cpu_engine {
                    cpu.reset(scenario, p_count, &mut self.params, &mut self.obstacles);
                }
            }
        }
    }

    pub fn update_and_render(&mut self) {
        let now = Instant::now();
        let dt_sec = now.duration_since(self.last_frame_time).as_secs_f32().min(0.1);
        self.last_frame_time = now;
        let dt_ms = dt_sec * 1000.0;

        self.camera.update(dt_sec);

        // Update dynamic obstacles (e.g. rotor rotation)
        for obs in &mut self.obstacles {
            obs.update(dt_sec);
        }

        // Raycast mouse to world plane for 3D interaction
        let (ray_origin, ray_dir) = self.camera.raycast_from_screen(
            self.mouse_pos,
            Vec2::new(self.surface_config.width as f32, self.surface_config.height as f32),
        );

        // Intersect with horizontal plane Y = 0.4
        let t_plane = (0.4 - ray_origin.y) / ray_dir.y;
        let world_mouse_target = if t_plane > 0.0 {
            ray_origin + ray_dir * t_plane
        } else {
            ray_origin + ray_dir * 3.0
        };

        // Handle interactive obstacle drag
        if self.is_shift_down && self.is_mouse_left_down {
            if let Some(idx) = self.grabbed_obstacle_idx {
                if idx < self.obstacles.len() {
                    let old_pos = self.obstacles[idx].position;
                    let target_pos = Vec3::new(
                        world_mouse_target.x.clamp(-0.8, 0.8),
                        world_mouse_target.y.clamp(0.15, 1.2),
                        world_mouse_target.z.clamp(-0.45, 0.45),
                    );
                    self.obstacles[idx].position = target_pos;
                    self.obstacles[idx].velocity = (target_pos - old_pos) / dt_sec.max(0.001);
                }
            } else {
                // Find closest obstacle to pick
                let mut best_idx = None;
                let mut min_d = 0.5;
                for (i, obs) in self.obstacles.iter().enumerate() {
                    let d = obs.position.distance(world_mouse_target);
                    if d < min_d {
                        min_d = d;
                        best_idx = Some(i);
                    }
                }
                if let Some(idx) = best_idx {
                    self.grabbed_obstacle_idx = Some(idx);
                    self.obstacles[idx].is_grabbed = true;
                }
            }
        } else {
            if let Some(idx) = self.grabbed_obstacle_idx
                && idx < self.obstacles.len() {
                    self.obstacles[idx].is_grabbed = false;
                    self.obstacles[idx].velocity = Vec3::ZERO;
                }
            self.grabbed_obstacle_idx = None;
        }

        // Apply mouse interaction forces to fluid
        if self.is_mouse_left_down && !self.is_shift_down {
            self.params.mouse_pos = world_mouse_target;
            self.params.mouse_active = true;
            match self.gui_state.mouse_interaction_mode {
                MouseInteractionMode::Attract => {
                    self.params.mouse_strength = -45.0;
                }
                MouseInteractionMode::Repel => {
                    self.params.mouse_strength = 45.0;
                }
                _ => {
                    self.params.mouse_strength = 45.0;
                }
            }

            // Ctrl + Left Click: Spray high-velocity water jet!
            if self.is_ctrl_down {
                let jet_dir = (world_mouse_target - self.camera.eye_position()).normalize();
                let spawn_pos = self.camera.eye_position() + jet_dir * 1.5;
                let new_particles = crate::physics::scenarios::spawn_sphere(
                    spawn_pos,
                    0.08,
                    0.035,
                    jet_dir * 8.0,
                    [0.2, 0.7, 1.0, 1.0],
                );
                match self.current_engine {
                    EngineType::GpuCompute => {
                        if let Some(gpu) = &mut self.gpu_engine {
                            gpu.add_particles(&new_particles);
                        }
                    }
                    EngineType::CpuMultiThread => {
                        if let Some(cpu) = &mut self.cpu_engine {
                            cpu.add_particles(&new_particles);
                        }
                    }
                }
            }
        } else {
            self.params.mouse_active = false;
        }

        // Physics step
        let sim_dt = dt_sec * self.gui_state.time_scale;
        let mut step_metrics = StepMetrics::default();

        let should_step = !self.gui_state.is_paused || self.gui_state.step_once;
        if should_step {
            self.gui_state.step_once = false;
            match self.current_engine {
                EngineType::GpuCompute => {
                    if let Some(gpu) = &mut self.gpu_engine {
                        step_metrics = gpu.step(sim_dt, &self.params, &self.obstacles);
                    }
                }
                EngineType::CpuMultiThread => {
                    if let Some(cpu) = &mut self.cpu_engine {
                        step_metrics = cpu.step(sim_dt, &self.params, &self.obstacles);
                        // Write CPU particles to GPU buffer for rendering
                        if let Some(gpu) = &self.gpu_engine {
                            let bytes: &[u8] = bytemuck::cast_slice(&cpu.gpu_particles_cache);
                            self.queue.write_buffer(&gpu.particle_buffer, 0, bytes);
                        }
                    }
                }
            }
        }

        let p_count = match self.current_engine {
            EngineType::GpuCompute => self.gpu_engine.as_ref().map(|g| g.particle_count()).unwrap_or(0),
            EngineType::CpuMultiThread => self.cpu_engine.as_ref().map(|c| c.particle_count()).unwrap_or(0),
        };

        self.gui_state.update_metrics(dt_ms, &step_metrics, p_count, self.params.substeps);

        // Update renderer settings
        self.renderer.render_style = self.gui_state.render_style;
        self.renderer.color_theme = self.gui_state.color_theme;
        self.renderer.particle_radius_scale = self.gui_state.particle_radius_scale;
        self.renderer.max_velocity_color = self.gui_state.max_velocity_color;
        self.renderer.update_camera(&self.camera);
        self.renderer.update_obstacles(&self.obstacles);

        // Prepare egui frame
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let mut trigger_reset = false;
        let mut trigger_engine_switch = None;

        let full_output = self.egui_ctx.run_ui(raw_input, |ctx| {
            self.gui_state.render_ui(
                ctx,
                &mut self.params,
                &mut self.obstacles,
                p_count,
                &mut trigger_reset,
                &mut trigger_engine_switch,
            );
        });

        self.egui_state.handle_platform_output(&self.window, full_output.platform_output);

        if trigger_reset {
            self.reset_simulation();
        }
        if let Some(engine) = trigger_engine_switch {
            self.switch_engine(engine);
        }

        let clipped_primitives = self.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        for (id, deltas) in &full_output.textures_delta.set {
            for delta in deltas {
                self.egui_renderer.update_texture(&self.device, &self.queue, *id, delta);
            }
        }

        // Render Frame
        let output_surface = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.surface_config);
                return;
            }
            _ => return,
        };

        let view = output_surface.texture.create_view(&wgpu::TextureViewDescriptor::default());

        // 1. Fluid & 3D Tank Rendering
        if let Some(gpu) = &self.gpu_engine {
            self.renderer.render(
                &view,
                &gpu.particle_buffer,
                p_count as u32,
                self.params.particle_radius,
            );
        }

        // 2. Egui GUI Rendering
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.surface_config.width, self.surface_config.height],
            pixels_per_point: self.window.scale_factor() as f32,
        };

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Egui Encoder"),
        });

        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &clipped_primitives,
            &screen_descriptor,
        );

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Egui Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            }).forget_lifetime();
            self.egui_renderer.render(&mut rpass, &clipped_primitives, &screen_descriptor);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        self.queue.present(output_surface);
    }
}

#[derive(Default)]
pub struct FluidApp {
    pub state: Option<AppState>,
}


impl ApplicationHandler for FluidApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            let window_attributes = Window::default_attributes()
                .with_title("Aether Fluid 3D - Real-Time Simulation & Benchmark")
                .with_inner_size(PhysicalSize::new(1440, 900));

            let window = Arc::new(event_loop.create_window(window_attributes).expect("Failed to create window"));
            self.state = Some(AppState::new(window));
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        let state = match &mut self.state {
            Some(s) => s,
            None => return,
        };

        let egui_consumed = state.egui_state.on_window_event(&state.window, &event).consumed;

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                state.resize(physical_size);
            }
            WindowEvent::CursorMoved { position, .. } => {
                let new_pos = Vec2::new(position.x as f32, position.y as f32);
                let delta = new_pos - state.mouse_pos;
                state.mouse_pos = new_pos;

                if !egui_consumed {
                    if state.is_mouse_right_down && !state.is_shift_down {
                        // Orbit camera
                        state.camera.orbit(delta.x, delta.y);
                    } else if state.is_mouse_middle_down || (state.is_mouse_right_down && state.is_shift_down) {
                        // Pan camera
                        state.camera.pan(delta.x, delta.y);
                    }
                }
            }
            WindowEvent::MouseInput { state: elem_state, button, .. } => {
                let is_down = elem_state == ElementState::Pressed;
                match button {
                    MouseButton::Left => state.is_mouse_left_down = is_down && !egui_consumed,
                    MouseButton::Right => state.is_mouse_right_down = is_down,
                    MouseButton::Middle => state.is_mouse_middle_down = is_down,
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if !egui_consumed {
                    let scroll = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.05,
                    };
                    state.camera.zoom(scroll);
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state: elem_state,
                        ..
                    },
                ..
            } => {
                let is_down = elem_state == ElementState::Pressed;
                match key {
                    KeyCode::ShiftLeft | KeyCode::ShiftRight => state.is_shift_down = is_down,
                    KeyCode::ControlLeft | KeyCode::ControlRight => state.is_ctrl_down = is_down,
                    KeyCode::Space if is_down => state.gui_state.is_paused = !state.gui_state.is_paused,
                    KeyCode::Period if is_down => state.gui_state.step_once = true,
                    KeyCode::KeyR if is_down => state.reset_simulation(),
                    KeyCode::Digit1 if is_down => state.gui_state.render_style = crate::render::RenderStyle::RealisticLiquid,
                    KeyCode::Digit2 if is_down => state.gui_state.render_style = crate::render::RenderStyle::ShadedSpheres,
                    KeyCode::Digit3 if is_down => state.gui_state.render_style = crate::render::RenderStyle::VelocityHeatmap,
                    KeyCode::Digit4 if is_down => state.gui_state.render_style = crate::render::RenderStyle::PressureHeatmap,
                    KeyCode::F11 if is_down => {
                        let is_fullscreen = state.window.fullscreen().is_some();
                        state.window.set_fullscreen(if is_fullscreen {
                            None
                        } else {
                            Some(winit::window::Fullscreen::Borderless(None))
                        });
                    }
                    _ => {}
                }
            }
            WindowEvent::RedrawRequested => {
                state.update_and_render();
                state.window.request_redraw();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}
