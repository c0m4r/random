use std::time::Instant;
use wgpu::util::DeviceExt;

use crate::physics::{
    particle::{ParticleCpu, ParticleGpu},
    obstacle::Obstacle,
    scenarios::{create_scenario, ScenarioType},
    PhysicsEngine, SimParams, StepMetrics, EngineType,
};

pub struct GpuSph {
    pub particle_count: usize,
    pub particle_buffer: wgpu::Buffer,
    pub params_buffer: wgpu::Buffer,
    pub grid_heads_buffer: wgpu::Buffer,
    pub grid_links_buffer: wgpu::Buffer,

    pub clear_grid_pipeline: wgpu::ComputePipeline,
    pub build_grid_pipeline: wgpu::ComputePipeline,
    pub density_pipeline: wgpu::ComputePipeline,
    pub forces_pipeline: wgpu::ComputePipeline,

    pub bind_group: wgpu::BindGroup,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub device: std::sync::Arc<wgpu::Device>,
    pub queue: std::sync::Arc<wgpu::Queue>,
    pub initial_particles: Vec<ParticleGpu>,
}

impl GpuSph {
    pub fn new(
        device: std::sync::Arc<wgpu::Device>,
        queue: std::sync::Arc<wgpu::Queue>,
        scenario: ScenarioType,
        target_particles: usize,
    ) -> (Self, SimParams, Vec<Obstacle>) {
        let mut params = SimParams::default();
        let config = create_scenario(scenario, target_particles);
        params.viscosity = config.viscosity;
        params.surface_tension = config.surface_tension;
        params.gravity = config.gravity;
        params.stiffness = config.stiffness;
        params.rest_density = config.rest_density;

        let particles_gpu: Vec<ParticleGpu> = config.particles.iter().map(|p| p.to_gpu()).collect();
        let particle_count = particles_gpu.len();

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SPH Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/compute_sph.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SPH Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SPH Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let clear_grid_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SPH Clear Grid Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_clear_grid"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let build_grid_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SPH Build Grid Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_build_grid"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let density_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SPH Density Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_density"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let forces_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SPH Forces Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_forces"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let max_particles = target_particles.max(300_000);
        let buffer_size = (max_particles * std::mem::size_of::<ParticleGpu>()) as u64;

        let particle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluid Particles Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        queue.write_buffer(&particle_buffer, 0, bytemuck::cast_slice(&particles_gpu));

        let initial_gpu_params = params.to_gpu(particle_count as u32, 0.005, &config.obstacles);
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SPH Params Buffer"),
            contents: bytemuck::bytes_of(&initial_gpu_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Total grid cells = 48 * 36 * 32 = 55,296
        let total_cells = 55296;
        let grid_heads_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SPH Grid Heads Buffer"),
            size: (total_cells * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grid_links_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SPH Grid Links Buffer"),
            size: (max_particles * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SPH Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grid_heads_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: grid_links_buffer.as_entire_binding(),
                },
            ],
        });

        let engine = Self {
            particle_count,
            particle_buffer,
            params_buffer,
            grid_heads_buffer,
            grid_links_buffer,
            clear_grid_pipeline,
            build_grid_pipeline,
            density_pipeline,
            forces_pipeline,
            bind_group,
            bind_group_layout,
            device,
            queue,
            initial_particles: particles_gpu,
        };

        (engine, params, config.obstacles)
    }

    pub fn dispatch(&mut self, dt: f32, params: &SimParams, obstacles: &[Obstacle]) -> StepMetrics {
        let start = Instant::now();
        let substeps = params.substeps.max(1);
        let sub_dt = dt / substeps as f32;

        let workgroups = (self.particle_count as u32).div_ceil(64);
        let grid_workgroups = 55296u32.div_ceil(64);

        for _ in 0..substeps {
            let gpu_params = params.to_gpu(self.particle_count as u32, sub_dt, obstacles);
            self.queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&gpu_params));

            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("SPH Step Encoder"),
            });

            // 1. Clear Grid
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SPH Clear Grid Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.clear_grid_pipeline);
                cpass.set_bind_group(0, &self.bind_group, &[]);
                cpass.dispatch_workgroups(grid_workgroups, 1, 1);
            }

            // 2. Build Grid
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SPH Build Grid Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.build_grid_pipeline);
                cpass.set_bind_group(0, &self.bind_group, &[]);
                cpass.dispatch_workgroups(workgroups, 1, 1);
            }

            // 3. Density Pass
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SPH Density Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.density_pipeline);
                cpass.set_bind_group(0, &self.bind_group, &[]);
                cpass.dispatch_workgroups(workgroups, 1, 1);
            }

            // 4. Forces Pass
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SPH Forces Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.forces_pipeline);
                cpass.set_bind_group(0, &self.bind_group, &[]);
                cpass.dispatch_workgroups(workgroups, 1, 1);
            }

            self.queue.submit(std::iter::once(encoder.finish()));
        }

        StepMetrics {
            step_duration_ms: start.elapsed().as_secs_f64() * 1000.0,
            substep_count: substeps,
            particle_updates: (self.particle_count as u64) * (substeps as u64),
            density_ms: 0.0,
            force_ms: 0.0,
            integration_ms: 0.0,
        }
    }
}

impl PhysicsEngine for GpuSph {
    fn step(&mut self, dt: f32, params: &SimParams, obstacles: &[Obstacle]) -> StepMetrics {
        self.dispatch(dt, params, obstacles)
    }

    fn reset(
        &mut self,
        scenario: ScenarioType,
        target_particles: usize,
        params: &mut SimParams,
        obstacles: &mut Vec<Obstacle>,
    ) {
        let config = create_scenario(scenario, target_particles);
        params.viscosity = config.viscosity;
        params.surface_tension = config.surface_tension;
        params.gravity = config.gravity;
        params.stiffness = config.stiffness;
        params.rest_density = config.rest_density;

        let particles_gpu: Vec<ParticleGpu> = config.particles.iter().map(|p| p.to_gpu()).collect();
        self.particle_count = particles_gpu.len();
        self.queue.write_buffer(&self.particle_buffer, 0, bytemuck::cast_slice(&particles_gpu));
        *obstacles = config.obstacles;
    }

    fn particle_count(&self) -> usize {
        self.particle_count
    }

    fn engine_type(&self) -> EngineType {
        EngineType::GpuCompute
    }

    fn add_particles(&mut self, new_particles: &[ParticleCpu]) {
        let new_gpu: Vec<ParticleGpu> = new_particles.iter().map(|p| p.to_gpu()).collect();
        let offset = (self.particle_count * std::mem::size_of::<ParticleGpu>()) as u64;
        self.queue.write_buffer(&self.particle_buffer, offset, bytemuck::cast_slice(&new_gpu));
        self.particle_count += new_gpu.len();
    }
}
