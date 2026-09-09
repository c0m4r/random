pub mod math;
pub mod physics;
pub mod render;
pub mod benchmark;
pub mod ui;
pub mod app;

use std::sync::Arc;
use clap::Parser;
use winit::event_loop::{ControlFlow, EventLoop};

use crate::app::FluidApp;
use crate::benchmark::run_headless_benchmark;

#[derive(Parser, Debug)]
#[command(name = "fluid", version = "0.1.0", about = "3D Real-Time Fluid Physics Simulation & Benchmark")]
pub struct CliArgs {
    /// Run automated headless benchmark suite and export reports to JSON/CSV
    #[arg(short, long)]
    pub benchmark: bool,

    /// Number of fluid particles to simulate in benchmark
    #[arg(short, long)]
    pub particles: Option<usize>,

    /// Duration of each benchmark tier in seconds
    #[arg(short, long)]
    pub duration: Option<u64>,

    /// Physics engine to benchmark ('gpu' or 'cpu')
    #[arg(short, long)]
    pub engine: Option<String>,

    /// Target discrete GPU device index (if multiple GPUs present)
    #[arg(long)]
    pub gpu_index: Option<usize>,

    /// Render a high-resolution frame headlessly and save to PNG image file
    #[arg(long)]
    pub screenshot: Option<String>,

    /// Rendering style for screenshot ('realistic', 'spheres', 'velocity', 'pressure')
    #[arg(long, default_value = "realistic")]
    pub render_style: String,

    /// Scenario preset ('dambreak', 'doublewave', 'splash', 'honey', 'vortex', 'zerog')
    #[arg(long, default_value = "dambreak")]
    pub scenario: String,

    /// Liquid color theme ('azure', 'emerald', 'honey', 'lava', 'silver', 'plasma')
    #[arg(long, default_value = "azure")]
    pub theme: String,

    /// Number of warm-up simulation steps before capturing screenshot
    #[arg(long, default_value_t = 90)]
    pub steps: usize,
}

fn main() {
    let args = CliArgs::parse();

    if let Some(path) = &args.screenshot {
        run_screenshot_cli(&args, path);
    } else if args.benchmark {
        run_benchmark_cli(&args);
    } else {
        run_windowed_app();
    }
}

fn run_benchmark_cli(args: &CliArgs) {
    println!("══════════════════════════════════════════════════════════════════");
    println!("        AETHER FLUID 3D — STANDALONE HEADLESS BENCHMARK           ");
    println!("══════════════════════════════════════════════════════════════════");

    let instance = wgpu::Instance::default();
    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    if adapters.is_empty() {
        eprintln!("Error: No wgpu graphics adapters detected on this system.");
        std::process::exit(1);
    }

    println!("Detected {} GPU adapter(s):", adapters.len());
    for (i, a) in adapters.iter().enumerate() {
        let info = a.get_info();
        println!("  [{}] {} ({:?}, backend: {:?})", i, info.name, info.device_type, info.backend);
    }

    // Pick discrete GPU if available, or selected index
    let chosen_adapter = if let Some(idx) = args.gpu_index {
        adapters.get(idx).expect("Invalid GPU index specified")
    } else {
        adapters.iter().find(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
            .unwrap_or(&adapters[0])
    };

    let adapter_info = chosen_adapter.get_info();
    println!("Using GPU: {} ({:?})", adapter_info.name, adapter_info.backend);

    let (device, queue) = pollster::block_on(chosen_adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Benchmark Device"),
            ..Default::default()
        },
    )).expect("Failed to initialize GPU compute device");

    let device = Arc::new(device);
    let queue = Arc::new(queue);

    run_headless_benchmark(
        device,
        queue,
        &adapter_info,
        args.particles,
        args.duration,
        args.engine.as_deref(),
    );
}

fn run_windowed_app() {
    println!("Launching Aether Fluid 3D Interactive Window...");
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = FluidApp::default();
    event_loop.run_app(&mut app).expect("Application event loop failed");
}

fn run_screenshot_cli(args: &CliArgs, output_path: &str) {
    use crate::physics::PhysicsEngine;

    println!("══════════════════════════════════════════════════════════════════");
    println!("        AETHER FLUID 3D — HEADLESS SCREENSHOT RENDERER             ");
    println!("══════════════════════════════════════════════════════════════════");

    let instance = wgpu::Instance::default();
    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    if adapters.is_empty() {
        eprintln!("Error: No wgpu graphics adapters detected on this system.");
        std::process::exit(1);
    }

    let chosen_adapter = if let Some(idx) = args.gpu_index {
        adapters.get(idx).expect("Invalid GPU index specified")
    } else {
        adapters.iter().find(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
            .unwrap_or(&adapters[0])
    };

    let adapter_info = chosen_adapter.get_info();
    println!("Using GPU: {} ({:?})", adapter_info.name, adapter_info.backend);

    let (device, queue) = pollster::block_on(chosen_adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Screenshot Device"),
            ..Default::default()
        },
    )).expect("Failed to initialize GPU compute device");

    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let width = 1920;
    let height = 1080;
    let surface_format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let depth_format = wgpu::TextureFormat::Depth32Float;

    let mut renderer = crate::render::Renderer::new(
        device.clone(),
        queue.clone(),
        surface_format,
        depth_format,
        width,
        height,
    );

    let particle_count = args.particles.unwrap_or(25_000);
    let scenario = match args.scenario.to_lowercase().as_str() {
        "doublewave" => crate::physics::scenarios::ScenarioType::DoubleWave,
        "splash" => crate::physics::scenarios::ScenarioType::DropletSplash,
        "honey" => crate::physics::scenarios::ScenarioType::ViscousHoney,
        "vortex" => crate::physics::scenarios::ScenarioType::VortexMixer,
        "zerog" => crate::physics::scenarios::ScenarioType::ZeroG,
        _ => crate::physics::scenarios::ScenarioType::DamBreak,
    };

    let style = match args.render_style.to_lowercase().as_str() {
        "spheres" | "shaded" => crate::render::RenderStyle::ShadedSpheres,
        "velocity" => crate::render::RenderStyle::VelocityHeatmap,
        "pressure" => crate::render::RenderStyle::PressureHeatmap,
        _ => crate::render::RenderStyle::RealisticLiquid,
    };

    let theme = match args.theme.to_lowercase().as_str() {
        "emerald" => crate::render::ColorTheme::EmeraldToxic,
        "honey" => crate::render::ColorTheme::GoldenHoney,
        "lava" => crate::render::ColorTheme::LavaAmber,
        "silver" => crate::render::ColorTheme::MercurySilver,
        "plasma" => crate::render::ColorTheme::PlasmaPurple,
        _ => crate::render::ColorTheme::OceanAzure,
    };

    renderer.render_style = style;
    renderer.color_theme = theme;

    let (mut engine, params, obstacles) = crate::physics::gpu_sph::GpuSph::new(
        device.clone(),
        queue.clone(),
        scenario,
        particle_count,
    );

    println!(
        "Simulating {} particles (scenario: {:?}) across {} warm-up physics steps...",
        engine.particle_count, scenario, args.steps
    );
    let dt = 0.003;
    for _ in 0..args.steps {
        engine.step(dt, &params, &obstacles);
    }

    let camera = crate::math::Camera {
        distance: 2.7,
        yaw: 0.55,
        pitch: 0.38,
        aspect_ratio: width as f32 / height as f32,
        ..Default::default()
    };

    renderer.update_camera(&camera);
    renderer.update_obstacles(&obstacles);

    println!(
        "Rendering 1920x1080 fluid frame (style: {:?}, theme: {:?}) to {}...",
        style, theme, output_path
    );
    if let Err(e) = renderer.render_to_image(
        &engine.particle_buffer,
        engine.particle_count as u32,
        params.particle_radius,
        output_path,
    ) {
        eprintln!("Error rendering screenshot: {:?}", e);
        std::process::exit(1);
    }

    println!("Screenshot successfully rendered to: {}", output_path);
}

