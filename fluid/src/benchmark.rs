use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

use crate::physics::{
    cpu_sph::CpuSph,
    gpu_sph::GpuSph,
    scenarios::ScenarioType,
    EngineType, PhysicsEngine, SimParams,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub os: String,
    pub cpu_cores: usize,
    pub gpu_name: String,
    pub gpu_backend: String,
    pub gpu_driver: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkTierResult {
    pub particle_count: usize,
    pub engine: String,
    pub frames_sampled: usize,
    pub duration_secs: f64,
    pub fps_min: f64,
    pub fps_avg: f64,
    pub fps_max: f64,
    pub fps_1_percent_low: f64,
    pub frametime_p50_ms: f64,
    pub frametime_p95_ms: f64,
    pub frametime_p99_ms: f64,
    pub physics_avg_ms: f64,
    pub pups_avg: f64,      // Particle Updates Per Second
    pub gflops_est: f64,
    pub score: u64,
    pub grade: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub title: String,
    pub timestamp: String,
    pub hardware: HardwareInfo,
    pub results: Vec<BenchmarkTierResult>,
    pub overall_score: u64,
    pub overall_grade: String,
}

impl BenchmarkReport {
    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        let mut file = File::create(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    pub fn save_csv<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let mut file = File::create(path)?;
        writeln!(
            file,
            "ParticleCount,Engine,Frames,DurationSec,FPS_Min,FPS_Avg,FPS_Max,FPS_1PctLow,FrametimeP50_ms,FrametimeP99_ms,PhysicsAvg_ms,PUPs_Avg,GFLOPS_Est,Score,Grade"
        )?;
        for r in &self.results {
            writeln!(
                file,
                "{},{},{},{:.3},{:.2},{:.2},{:.2},{:.2},{:.3},{:.3},{:.3},{:.0},{:.2},{},{}",
                r.particle_count,
                r.engine,
                r.frames_sampled,
                r.duration_secs,
                r.fps_min,
                r.fps_avg,
                r.fps_max,
                r.fps_1_percent_low,
                r.frametime_p50_ms,
                r.frametime_p99_ms,
                r.physics_avg_ms,
                r.pups_avg,
                r.gflops_est,
                r.score,
                r.grade
            )?;
        }
        Ok(())
    }

    pub fn print_terminal_summary(&self) {
        println!("\n╔════════════════════════════════════════════════════════════════════════════════╗");
        println!("║                      3D FLUID PHYSICS BENCHMARK REPORT                         ║");
        println!("╠════════════════════════════════════════════════════════════════════════════════╣");
        println!("║ Hardware: {:<68} ║", self.hardware.gpu_name);
        println!("║ Backend:  {:<18} Driver: {:<40} ║", self.hardware.gpu_backend, self.hardware.gpu_driver);
        println!("║ CPU:      {:<18} Cores:  {:<40} ║", self.hardware.os, self.hardware.cpu_cores);
        println!("╠════════════════════════════════════════════════════════════════════════════════╣");
        println!("║ Particles │ Engine │   FPS (Avg/1% Low)  │ Step Time │    PUP/s    │ Score │ Grade ║");
        println!("╟───────────┼────────┼─────────────────────┼───────────┼─────────────┼───────┼───────╢");
        for r in &self.results {
            println!(
                "║ {:>9} │ {:<6} │ {:>6.1} / {:>5.1} fps │ {:>7.2}ms │ {:>9.2}M │ {:>5} │ {:>5} ║",
                r.particle_count,
                r.engine,
                r.fps_avg,
                r.fps_1_percent_low,
                r.physics_avg_ms,
                r.pups_avg / 1_000_000.0,
                r.score,
                r.grade
            );
        }
        println!("╠════════════════════════════════════════════════════════════════════════════════╣");
        println!(
            "║ OVERALL FLUID BENCHMARK SCORE: {:>8}                      GRADE: [{:>4}]   ║",
            self.overall_score, self.overall_grade
        );
        println!("╚════════════════════════════════════════════════════════════════════════════════╝\n");
    }
}

pub fn calculate_grade(score: u64) -> String {
    if score >= 100_000 {
        "S+".to_string()
    } else if score >= 60_000 {
        "S".to_string()
    } else if score >= 35_000 {
        "A".to_string()
    } else if score >= 18_000 {
        "B".to_string()
    } else if score >= 8_000 {
        "C".to_string()
    } else {
        "D".to_string()
    }
}

pub fn run_benchmark_tier(
    engine: &mut dyn PhysicsEngine,
    params: &SimParams,
    obstacles: &[crate::physics::obstacle::Obstacle],
    warmup_frames: usize,
    sample_duration: Duration,
) -> BenchmarkTierResult {
    let dt = 1.0 / 60.0;

    // Warmup
    for _ in 0..warmup_frames {
        engine.step(dt, params, obstacles);
    }

    let mut frametimes_ms = Vec::new();
    let mut step_times_ms = Vec::new();
    let start_time = Instant::now();

    while start_time.elapsed() < sample_duration {
        let frame_start = Instant::now();
        let metrics = engine.step(dt, params, obstacles);
        let frame_time = frame_start.elapsed().as_secs_f64() * 1000.0;

        frametimes_ms.push(frame_time);
        step_times_ms.push(metrics.step_duration_ms);
    }

    let duration_secs = start_time.elapsed().as_secs_f64();
    let frames_sampled = frametimes_ms.len();

    frametimes_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    step_times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let frametime_p50_ms = frametimes_ms[frames_sampled / 2];
    let frametime_p95_ms = frametimes_ms[(frames_sampled as f64 * 0.95) as usize];
    let frametime_p99_ms = frametimes_ms[(frames_sampled as f64 * 0.99) as usize];

    let total_frametime: f64 = frametimes_ms.iter().sum();
    let fps_avg = (frames_sampled as f64) / (total_frametime / 1000.0);
    let fps_min = 1000.0 / frametimes_ms.last().copied().unwrap_or(1.0);
    let fps_max = 1000.0 / frametimes_ms.first().copied().unwrap_or(1.0);
    let fps_1_percent_low = 1000.0 / frametime_p99_ms.max(0.001);

    let physics_avg_ms = step_times_ms.iter().sum::<f64>() / (frames_sampled as f64);
    let particles = engine.particle_count();
    let pups_avg = (particles as f64) * (params.substeps as f64) * fps_avg;

    // ~80 FLOPs per particle neighbor interaction
    let gflops_est = (pups_avg * 80.0) / 1e9;

    let stability_factor = (fps_1_percent_low / fps_avg.max(1.0)).clamp(0.2, 1.0);
    let raw_score = (pups_avg * 0.0008 * (fps_avg / 60.0).sqrt() * stability_factor) as u64;
    let score = raw_score.max(1);
    let grade = calculate_grade(score);

    BenchmarkTierResult {
        particle_count: particles,
        engine: match engine.engine_type() {
            EngineType::GpuCompute => "GPU".to_string(),
            EngineType::CpuMultiThread => "CPU".to_string(),
        },
        frames_sampled,
        duration_secs,
        fps_min,
        fps_avg,
        fps_max,
        fps_1_percent_low,
        frametime_p50_ms,
        frametime_p95_ms,
        frametime_p99_ms,
        physics_avg_ms,
        pups_avg,
        gflops_est,
        score,
        grade,
    }
}

pub fn run_headless_benchmark(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    adapter_info: &wgpu::AdapterInfo,
    custom_particles: Option<usize>,
    custom_duration_secs: Option<u64>,
    engine_choice: Option<&str>,
) -> BenchmarkReport {
    let hw = HardwareInfo {
        os: "Linux x86_64".to_string(),
        cpu_cores: rayon::current_num_threads(),
        gpu_name: adapter_info.name.clone(),
        gpu_backend: format!("{:?}", adapter_info.backend),
        gpu_driver: format!("{:?}", adapter_info.driver),
    };

    println!("\nStarting Fluid Physics Real-Time Benchmark Suite...");
    println!("Device: {} ({:?})", hw.gpu_name, hw.gpu_backend);

    let particle_tiers: Vec<usize> = if let Some(p) = custom_particles {
        vec![p]
    } else {
        vec![10_000, 25_000, 50_000, 100_000]
    };

    let sample_duration = Duration::from_secs(custom_duration_secs.unwrap_or(4));
    let mut results = Vec::new();

    let test_gpu = engine_choice.is_none() || engine_choice == Some("gpu");
    let test_cpu = engine_choice.is_none() || engine_choice == Some("cpu");

    if test_gpu {
        println!("\n--- Testing GPU Compute Engine ---");
        for &particles in &particle_tiers {
            print!("  Running GPU Tier: {:>6} particles... ", particles);
            std::io::stdout().flush().unwrap();
            let (mut engine, params, obstacles) = GpuSph::new(
                device.clone(),
                queue.clone(),
                ScenarioType::DamBreak,
                particles,
            );
            let res = run_benchmark_tier(&mut engine, &params, &obstacles, 40, sample_duration);
            println!("{:.1} FPS | {:.2}M PUP/s | Score: {} [{}]", res.fps_avg, res.pups_avg / 1e6, res.score, res.grade);
            results.push(res);
        }
    }

    if test_cpu {
        println!("\n--- Testing CPU Multi-Threaded Engine (Rayon) ---");
        let cpu_tiers: Vec<usize> = if let Some(p) = custom_particles {
            vec![p]
        } else {
            vec![5_000, 10_000, 20_000]
        };
        for &particles in &cpu_tiers {
            print!("  Running CPU Tier: {:>6} particles... ", particles);
            std::io::stdout().flush().unwrap();
            let (mut engine, params, obstacles) = CpuSph::new(ScenarioType::DamBreak, particles);
            let res = run_benchmark_tier(&mut engine, &params, &obstacles, 20, sample_duration);
            println!("{:.1} FPS | {:.2}M PUP/s | Score: {} [{}]", res.fps_avg, res.pups_avg / 1e6, res.score, res.grade);
            results.push(res);
        }
    }

    let total_score: u64 = results.iter().map(|r| r.score).sum::<u64>();
    let overall_score = if !results.is_empty() { total_score / results.len() as u64 } else { 0 };
    let overall_grade = calculate_grade(overall_score);

    let report = BenchmarkReport {
        title: "Fluid Dynamics 3D Benchmark".to_string(),
        timestamp: chrono_like_timestamp(),
        hardware: hw,
        results,
        overall_score,
        overall_grade,
    };

    report.print_terminal_summary();
    report.save_json("benchmark_results.json").ok();
    report.save_csv("benchmark_results.csv").ok();
    println!("Benchmark reports exported to benchmark_results.json and benchmark_results.csv\n");

    report
}

fn chrono_like_timestamp() -> String {
    let now = std::time::SystemTime::now();
    let secs = now.duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
    format!("UNIX-{}", secs)
}
