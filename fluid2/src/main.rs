mod app;
mod renderer;
mod worker;

use fluid_observatory::{
    cases::{self, Scene},
    physics::Model,
    validation,
};
use worker::Config;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = Config::default();
    let mut validate = false;
    let mut headless = false;
    let mut steps = 100;
    let mut paused = false;
    let mut capture = None;
    let mut capture_after = 0.3;
    let mut smoke = false;
    let mut export = false;
    let mut args = std::env::args().skip(1);
    let mut explicit_model = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                println!(
                    "Fluid Observatory — native 3D Newtonian & relativistic hydrodynamics\n\n  cargo run --release -- [OPTIONS]\n\n  --scene jet|blast|shear|shock\n  --model newtonian|relativistic\n  --resolution N      transverse cells (8–64); x cells = 3N/2\n  --jet-speed V       inlet velocity, 0 < V < 0.995\n  --paused            start paused\n  --validate          run analytical and 3D conservation suite\n  --headless          run physics without a display\n  --steps N           headless steps (default 100)\n  --export            export headless result as VTK\n  --capture PATH      save a PNG of the app and exit\n  --capture-after T   simulation time before capture (default 0.3)\n  --smoke             evolve and render all four scenes, capture and exit\n\n  Space pause · N step · 1–4 scenes · R reset · F immerse · H guide"
                );
                return Ok(());
            }
            "--validate" => validate = true,
            "--headless" => headless = true,
            "--paused" => paused = true,
            "--export" => export = true,
            "--smoke" => smoke = true,
            "--scene" => {
                config.scene =
                    Scene::parse(&args.next().ok_or("missing scene")?).ok_or("unknown scene")?;
            }
            "--model" => {
                config.model = match args.next().ok_or("missing model")?.as_str() {
                    "newtonian" => Model::Newtonian,
                    "relativistic" => Model::Relativistic,
                    _ => return Err("unknown model".into()),
                };
                explicit_model = true;
            }
            "--resolution" => {
                config.resolution = args.next().ok_or("missing resolution")?.parse()?;
                if !(8..=64).contains(&config.resolution) {
                    return Err("resolution must be 8–64".into());
                }
            }
            "--jet-speed" => {
                config.jet_speed = args.next().ok_or("missing speed")?.parse()?;
                if !(0.0..0.995).contains(&config.jet_speed) || config.jet_speed == 0.0 {
                    return Err("jet speed must be 0 < v < 0.995".into());
                }
            }
            "--steps" => steps = args.next().ok_or("missing steps")?.parse::<u64>()?,
            "--capture" => capture = Some(args.next().ok_or("missing capture path")?),
            "--capture-after" => {
                capture_after = args.next().ok_or("missing capture time")?.parse::<f64>()?;
                if !capture_after.is_finite() || capture_after < 0.0 {
                    return Err("capture time must be finite and nonnegative".into());
                }
            }
            _ => return Err(format!("Unknown option: {arg}; use --help").into()),
        }
    }
    if !explicit_model {
        config.model = config.scene.default_model();
    }
    if smoke {
        config.scene = Scene::Jet;
        config.model = Model::Relativistic;
        paused = false;
    }
    if validate {
        if !validation::report()? {
            std::process::exit(1);
        }
        return Ok(());
    }
    if headless {
        let mut grid = cases::build(
            config.scene,
            config.model,
            config.resolution,
            config.jet_speed,
        );
        let start = std::time::Instant::now();
        for _ in 0..steps {
            grid.step(1.0)?;
        }
        let seconds = start.elapsed().as_secs_f64();
        println!(
            "scene={} model={:?} cells={} steps={} t={:.6} wall_s={:.3} ms/step={:.3} simulated_time/wall_time={:.4}",
            config.scene.id(),
            config.model,
            grid.u.len(),
            grid.steps,
            grid.time,
            seconds,
            seconds * 1000.0 / steps.max(1) as f64,
            grid.time / seconds
        );
        println!(
            "conservation_residual={:?} rejected={} first_order={}",
            grid.residual(),
            grid.rejected,
            grid.first_order_steps
        );
        if export {
            println!("Export: {}", worker::export_vtk(&grid)?);
        }
        return Ok(());
    }
    if paused && capture.is_some() && capture_after > 0.0 {
        return Err("--paused with --capture requires --capture-after 0".into());
    }
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Glow,
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 920.0])
            .with_min_inner_size([960.0, 600.0])
            .with_maximized(true)
            .with_title("Fluid Observatory | A living atlas of motion"),
        ..Default::default()
    };
    let launch = app::Launch {
        config,
        paused,
        capture,
        capture_after,
        smoke,
        completion: std::sync::Arc::new(std::sync::Mutex::new(None)),
    };
    let completion = launch.completion.clone();
    let must_complete = launch.capture.is_some() || launch.smoke;
    eframe::run_native(
        "Fluid Observatory",
        options,
        Box::new(move |cc| Ok(Box::new(app::Observatory::new(cc, launch)?))),
    )?;
    if must_complete {
        match completion.lock().unwrap().take() {
            Some(Ok(())) => println!("Graphics verification completed."),
            Some(Err(e)) => return Err(e.into()),
            None => return Err("Window closed before graphics verification completed".into()),
        }
    }
    Ok(())
}
