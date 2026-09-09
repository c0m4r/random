use std::time::Instant;
use egui::{Color32, RichText, Ui};

use crate::physics::{
    scenarios::ScenarioType,
    obstacle::{Obstacle, ObstacleKind},
    EngineType, SimParams, StepMetrics,
};
use crate::render::{ColorTheme, RenderStyle};
use crate::benchmark::{BenchmarkTierResult, calculate_grade};

pub struct GuiState {
    pub selected_tab: Tab,
    pub scenario: ScenarioType,
    pub target_particles: usize,
    pub engine_type: EngineType,
    pub render_style: RenderStyle,
    pub color_theme: ColorTheme,
    pub particle_radius_scale: f32,
    pub max_velocity_color: f32,

    pub is_paused: bool,
    pub step_once: bool,
    pub time_scale: f32,

    // Telemetry
    pub frametimes: Vec<f32>,
    pub physics_times: Vec<f32>,
    pub fps_current: f32,
    pub fps_avg: f32,
    pub fps_1_pct_low: f32,
    pub pups: f64,
    pub gflops: f64,

    // In-app benchmark
    pub is_benchmarking: bool,
    pub benchmark_start: Option<Instant>,
    pub benchmark_duration_secs: f32,
    pub benchmark_samples: Vec<f32>,
    pub benchmark_result: Option<BenchmarkTierResult>,

    // Interactivity
    pub mouse_interaction_mode: MouseInteractionMode,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Tab {
    Simulation,
    Rendering,
    Obstacles,
    Benchmark,
    Controls,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MouseInteractionMode {
    Attract,
    Repel,
    Vortex,
    FluidJet,
}

impl Default for GuiState {
    fn default() -> Self {
        Self {
            selected_tab: Tab::Simulation,
            scenario: ScenarioType::DamBreak,
            target_particles: 25_000,
            engine_type: EngineType::GpuCompute,
            render_style: RenderStyle::RealisticLiquid,
            color_theme: ColorTheme::OceanAzure,
            particle_radius_scale: 1.0,
            max_velocity_color: 4.5,
            is_paused: false,
            step_once: false,
            time_scale: 1.0,
            frametimes: Vec::with_capacity(120),
            physics_times: Vec::with_capacity(120),
            fps_current: 60.0,
            fps_avg: 60.0,
            fps_1_pct_low: 55.0,
            pups: 0.0,
            gflops: 0.0,
            is_benchmarking: false,
            benchmark_start: None,
            benchmark_duration_secs: 15.0,
            benchmark_samples: Vec::new(),
            benchmark_result: None,
            mouse_interaction_mode: MouseInteractionMode::Repel,
        }
    }
}

impl GuiState {
    pub fn update_metrics(&mut self, dt_ms: f32, step_metrics: &StepMetrics, particle_count: usize, substeps: u32) {
        if self.frametimes.len() >= 120 {
            self.frametimes.remove(0);
        }
        self.frametimes.push(dt_ms);

        if self.physics_times.len() >= 120 {
            self.physics_times.remove(0);
        }
        self.physics_times.push(step_metrics.step_duration_ms as f32);

        if dt_ms > 0.001 {
            self.fps_current = 1000.0 / dt_ms;
        }

        let count = self.frametimes.len() as f32;
        let sum: f32 = self.frametimes.iter().sum();
        self.fps_avg = 1000.0 / (sum / count).max(0.001);

        let mut sorted = self.frametimes.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p99_idx = (sorted.len() as f32 * 0.99) as usize;
        let p99 = sorted[p99_idx.min(sorted.len() - 1)];
        self.fps_1_pct_low = 1000.0 / p99.max(0.001);

        self.pups = (particle_count as f64) * (substeps as f64) * (self.fps_avg as f64);
        self.gflops = (self.pups * 80.0) / 1e9;

        // In-app benchmark progress
        if self.is_benchmarking
            && let Some(start) = self.benchmark_start {
                self.benchmark_samples.push(dt_ms);
                if start.elapsed().as_secs_f32() >= self.benchmark_duration_secs {
                    self.finish_in_app_benchmark(particle_count, substeps);
                }
            }
    }

    pub fn start_in_app_benchmark(&mut self) {
        self.is_benchmarking = true;
        self.benchmark_start = Some(Instant::now());
        self.benchmark_samples.clear();
        self.benchmark_result = None;
    }

    fn finish_in_app_benchmark(&mut self, particle_count: usize, substeps: u32) {
        self.is_benchmarking = false;
        self.benchmark_start = None;

        if self.benchmark_samples.is_empty() {
            return;
        }

        let mut samples = self.benchmark_samples.clone();
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = samples.len();
        let total_ms: f32 = samples.iter().sum();
        let duration_secs = total_ms / 1000.0;
        let fps_avg = (n as f64) / (duration_secs as f64);
        let fps_min = (1000.0 / samples.last().copied().unwrap_or(1.0)) as f64;
        let fps_max = (1000.0 / samples.first().copied().unwrap_or(1.0)) as f64;
        let p99_ms = samples[(n as f32 * 0.99) as usize] as f64;
        let p50_ms = samples[n / 2] as f64;
        let p95_ms = samples[(n as f32 * 0.95) as usize] as f64;
        let fps_1_pct_low = 1000.0 / p99_ms;

        let pups_avg = (particle_count as f64) * (substeps as f64) * fps_avg;
        let gflops_est = (pups_avg * 80.0) / 1e9;
        let stability = (fps_1_pct_low / fps_avg.max(1.0)).clamp(0.2, 1.0);
        let score = (pups_avg * 0.0008 * (fps_avg / 60.0).sqrt() * stability) as u64;
        let grade = calculate_grade(score);

        self.benchmark_result = Some(BenchmarkTierResult {
            particle_count,
            engine: match self.engine_type {
                EngineType::GpuCompute => "GPU".to_string(),
                EngineType::CpuMultiThread => "CPU".to_string(),
            },
            frames_sampled: n,
            duration_secs: duration_secs as f64,
            fps_min,
            fps_avg,
            fps_max,
            fps_1_percent_low: fps_1_pct_low,
            frametime_p50_ms: p50_ms,
            frametime_p95_ms: p95_ms,
            frametime_p99_ms: p99_ms,
            physics_avg_ms: p50_ms * 0.7,
            pups_avg,
            gflops_est,
            score,
            grade,
        });
    }

    pub fn render_ui(
        &mut self,
        ctx: &egui::Context,
        params: &mut SimParams,
        obstacles: &mut Vec<Obstacle>,
        particle_count: usize,
        trigger_reset: &mut bool,
        trigger_engine_switch: &mut Option<EngineType>,
    ) {
        egui::Window::new("🌊 AETHER FLUID 3D")
            .default_width(360.0)
            .default_pos([16.0, 16.0])
            .resizable(true)
            .collapsible(true)
            .show(ctx, |ui| {
                ui.add_space(8.0);
                ui.heading(RichText::new("🌊 AETHER FLUID 3D").size(22.0).strong().color(Color32::from_rgb(80, 200, 255)));
                ui.label(RichText::new("Real-time GPU/CPU Physics & Benchmark").size(12.0).color(Color32::from_rgb(160, 180, 200)));
                ui.separator();

                // Live Performance HUD
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(format!("{:.0} FPS", self.fps_current)).size(18.0).strong().color(
                            if self.fps_current >= 55.0 { Color32::GREEN } else if self.fps_current >= 30.0 { Color32::YELLOW } else { Color32::RED }
                        ));
                        ui.label(format!("(Avg: {:.1} | 1% Low: {:.1})", self.fps_avg, self.fps_1_pct_low));
                    });

                    ui.horizontal(|ui| {
                        ui.label(format!("Particles: {}", particle_count));
                        ui.separator();
                        ui.label(format!("{:.2}M PUP/s", self.pups / 1e6));
                    });

                    // Mini Frametime Graph
                    if self.frametimes.len() > 10 {
                        let points: Vec<egui::Pos2> = self.frametimes.iter().enumerate().map(|(i, &ms)| {
                            egui::pos2(i as f32 * 2.6, (35.0 - ms.min(33.3)).max(0.0))
                        }).collect();
                        let (response, painter) = ui.allocate_painter(egui::vec2(310.0, 40.0), egui::Sense::hover());
                        let rect = response.rect;
                        painter.rect_filled(rect, 4.0, Color32::from_rgb(25, 28, 35));
                        for w in points.windows(2) {
                            let p1 = egui::pos2(rect.min.x + w[0].x, rect.min.y + w[0].y);
                            let p2 = egui::pos2(rect.min.x + w[1].x, rect.min.y + w[1].y);
                            painter.line_segment([p1, p2], egui::Stroke::new(1.5, Color32::from_rgb(60, 180, 240)));
                        }
                    }
                });

                ui.add_space(6.0);

                // Simulation Control Bar
                ui.horizontal(|ui| {
                    if ui.button(if self.is_paused { "▶ Resume" } else { "⏸ Pause" }).clicked() {
                        self.is_paused = !self.is_paused;
                    }
                    if self.is_paused
                        && ui.button("⏭ Step").clicked() {
                            self.step_once = true;
                        }
                    if ui.button("🔄 Reset").clicked() {
                        *trigger_reset = true;
                    }
                });

                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label("Speed:");
                    ui.add(egui::Slider::new(&mut self.time_scale, 0.1..=2.0).text("x"));
                });

                ui.separator();

                // Navigation Tabs
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.selected_tab, Tab::Simulation, "Physics");
                    ui.selectable_value(&mut self.selected_tab, Tab::Rendering, "Visuals");
                    ui.selectable_value(&mut self.selected_tab, Tab::Obstacles, "Obstacles");
                    ui.selectable_value(&mut self.selected_tab, Tab::Benchmark, "Benchmark");
                    ui.selectable_value(&mut self.selected_tab, Tab::Controls, "Controls");
                });
                ui.separator();

                egui::ScrollArea::vertical().show(ui, |ui| {
                    match self.selected_tab {
                        Tab::Simulation => {
                            self.ui_simulation_tab(ui, params, trigger_reset, trigger_engine_switch);
                        }
                        Tab::Rendering => {
                            self.ui_rendering_tab(ui);
                        }
                        Tab::Obstacles => {
                            self.ui_obstacles_tab(ui, obstacles);
                        }
                        Tab::Benchmark => {
                            self.ui_benchmark_tab(ui, particle_count);
                        }
                        Tab::Controls => {
                            self.ui_controls_tab(ui);
                        }
                    }
                });
            });
    }

    fn ui_simulation_tab(
        &mut self,
        ui: &mut Ui,
        params: &mut SimParams,
        trigger_reset: &mut bool,
        trigger_engine_switch: &mut Option<EngineType>,
    ) {
        ui.heading("Preset Scenarios");
        for s in ScenarioType::ALL {
            if ui.selectable_label(self.scenario == s, s.name()).clicked() {
                self.scenario = s;
                *trigger_reset = true;
            }
        }

        ui.add_space(10.0);
        ui.heading("Physics Engine");
        ui.horizontal(|ui| {
            if ui.radio_value(&mut self.engine_type, EngineType::GpuCompute, "GPU Compute (WGSL)").clicked() {
                *trigger_engine_switch = Some(EngineType::GpuCompute);
            }
            if ui.radio_value(&mut self.engine_type, EngineType::CpuMultiThread, "CPU (Rayon)").clicked() {
                *trigger_engine_switch = Some(EngineType::CpuMultiThread);
            }
        });

        ui.add_space(10.0);
        ui.heading("Particle Count");
        let prev_count = self.target_particles;
        ui.add(egui::Slider::new(&mut self.target_particles, 2_000..=150_000).logarithmic(true).text("particles"));
        if self.target_particles != prev_count && ui.button("Apply New Count").clicked() {
            *trigger_reset = true;
        }

        ui.add_space(10.0);
        ui.heading("Physical Parameters");
        ui.add(egui::Slider::new(&mut params.viscosity, 0.005..=1.5).logarithmic(true).text("Viscosity"));
        ui.add(egui::Slider::new(&mut params.surface_tension, 0.0..=0.15).text("Surface Tension"));
        ui.add(egui::Slider::new(&mut params.stiffness, 50.0..=600.0).text("Stiffness (Bulk Modulus)"));
        ui.add(egui::Slider::new(&mut params.rest_density, 500.0..=2000.0).text("Rest Density (kg/m³)"));
        ui.add(egui::Slider::new(&mut params.damping, 0.05..=0.85).text("Boundary Damping"));
        ui.add(egui::Slider::new(&mut params.substeps, 1..=6).text("Substeps per frame"));

        ui.add_space(8.0);
        ui.heading("Gravity Vector");
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut params.gravity.x).speed(0.1).prefix("X: "));
            ui.add(egui::DragValue::new(&mut params.gravity.y).speed(0.1).prefix("Y: "));
            ui.add(egui::DragValue::new(&mut params.gravity.z).speed(0.1).prefix("Z: "));
        });
        ui.horizontal(|ui| {
            if ui.button("Earth Normal (-9.81)").clicked() {
                params.gravity = glam::Vec3::new(0.0, -9.81, 0.0);
            }
            if ui.button("Zero-G (0.0)").clicked() {
                params.gravity = glam::Vec3::ZERO;
            }
            if ui.button("Invert (+9.81)").clicked() {
                params.gravity = glam::Vec3::new(0.0, 9.81, 0.0);
            }
        });
    }

    fn ui_rendering_tab(&mut self, ui: &mut Ui) {
        ui.heading("Render Style");
        for style in RenderStyle::ALL {
            ui.radio_value(&mut self.render_style, style, style.name());
        }

        ui.add_space(10.0);
        ui.heading("Liquid Palette & Theme");
        for theme in ColorTheme::ALL {
            ui.radio_value(&mut self.color_theme, theme, theme.name());
        }

        ui.add_space(10.0);
        ui.heading("Visual Adjustments");
        ui.add(egui::Slider::new(&mut self.particle_radius_scale, 0.5..=2.5).text("Particle Radius Scale"));
        ui.add(egui::Slider::new(&mut self.max_velocity_color, 1.0..=12.0).text("Heatmap Max Speed"));
    }

    fn ui_obstacles_tab(&mut self, ui: &mut Ui, obstacles: &mut Vec<Obstacle>) {
        ui.heading("Dynamic Colliders");
        ui.label("Interact with obstacles using the mouse in 3D view!");

        let mut remove_idx = None;
        for (i, obs) in obstacles.iter_mut().enumerate() {
            ui.group(|ui| {
                match &mut obs.kind {
                    ObstacleKind::Sphere { radius } => {
                        ui.label(RichText::new(format!("Sphere Collider #{}", i + 1)).strong());
                        ui.add(egui::Slider::new(radius, 0.05..=0.5).text("Radius"));
                        ui.horizontal(|ui| {
                            ui.label("Pos:");
                            ui.add(egui::DragValue::new(&mut obs.position.x).speed(0.02).prefix("X: "));
                            ui.add(egui::DragValue::new(&mut obs.position.y).speed(0.02).prefix("Y: "));
                            ui.add(egui::DragValue::new(&mut obs.position.z).speed(0.02).prefix("Z: "));
                        });
                    }
                    ObstacleKind::Rotor {
                        radius,
                        height,
                        angular_velocity,
                        num_blades,
                        ..
                    } => {
                        ui.label(RichText::new(format!("Turbine Rotor #{}", i + 1)).strong());
                        ui.add(egui::Slider::new(radius, 0.1..=0.6).text("Blade Radius"));
                        ui.add(egui::Slider::new(height, 0.1..=0.8).text("Height"));
                        let mut rpm = *angular_velocity * 60.0 / std::f32::consts::TAU;
                        if ui.add(egui::Slider::new(&mut rpm, -300.0..=300.0).text("RPM")).changed() {
                            *angular_velocity = rpm * std::f32::consts::TAU / 60.0;
                        }
                        ui.add(egui::Slider::new(num_blades, 2..=8).text("Blade Count"));
                    }
                    ObstacleKind::BoxCollider { half_extents } => {
                        ui.label(RichText::new(format!("Box Obstacle #{}", i + 1)).strong());
                        ui.horizontal(|ui| {
                            ui.add(egui::DragValue::new(&mut half_extents.x).speed(0.02).prefix("W: "));
                            ui.add(egui::DragValue::new(&mut half_extents.y).speed(0.02).prefix("H: "));
                            ui.add(egui::DragValue::new(&mut half_extents.z).speed(0.02).prefix("D: "));
                        });
                    }
                }
                if ui.button("🗑 Remove").clicked() {
                    remove_idx = Some(i);
                }
            });
        }

        if let Some(idx) = remove_idx {
            obstacles.remove(idx);
        }

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if ui.button("+ Add Sphere").clicked() {
                obstacles.push(Obstacle::new_sphere(glam::Vec3::new(0.0, 0.4, 0.0), 0.22));
            }
            if ui.button("+ Add Turbine").clicked() {
                obstacles.push(Obstacle::new_rotor(glam::Vec3::new(0.0, 0.25, 0.0), 0.35, 0.4, 120.0));
            }
        });
    }

    fn ui_benchmark_tab(&mut self, ui: &mut Ui, _particle_count: usize) {
        ui.heading("Performance Benchmark");
        ui.label("Run an automated standard stress test and get a certified benchmark score and tier rating.");

        ui.add_space(8.0);
        if self.is_benchmarking {
            let elapsed = self.benchmark_start.map(|s| s.elapsed().as_secs_f32()).unwrap_or(0.0);
            let progress = (elapsed / self.benchmark_duration_secs).clamp(0.0, 1.0);
            ui.label(RichText::new(format!("Benchmarking: {:.1}s / {:.0}s", elapsed, self.benchmark_duration_secs)).strong());
            ui.add(egui::ProgressBar::new(progress).show_percentage());
        } else {
            ui.horizontal(|ui| {
                ui.label("Duration:");
                ui.add(egui::Slider::new(&mut self.benchmark_duration_secs, 5.0..=30.0).text("seconds"));
            });

            if ui.button(RichText::new("⚡ START BENCHMARK").size(16.0).strong().color(Color32::from_rgb(50, 220, 120))).clicked() {
                self.start_in_app_benchmark();
            }
        }

        if let Some(res) = &self.benchmark_result {
            ui.add_space(12.0);
            ui.group(|ui| {
                ui.heading("Benchmark Results");
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("SCORE: {}", res.score)).size(24.0).strong().color(Color32::from_rgb(255, 215, 0)));
                    ui.label(RichText::new(format!("GRADE: [{}]", res.grade)).size(24.0).strong().color(Color32::from_rgb(80, 220, 255)));
                });
                ui.separator();
                ui.label(format!("Engine: {} | Particles: {}", res.engine, res.particle_count));
                ui.label(format!("Average FPS: {:.1} (Min: {:.1} | Max: {:.1})", res.fps_avg, res.fps_min, res.fps_max));
                ui.label(format!("1% Low FPS: {:.1} fps", res.fps_1_percent_low));
                ui.label(format!("P50 Frametime: {:.2} ms | P99: {:.2} ms", res.frametime_p50_ms, res.frametime_p99_ms));
                ui.label(format!("Fluid Throughput: {:.2} Million PUP/s", res.pups_avg / 1e6));
                ui.label(format!("Estimated Compute: {:.1} GFLOPS", res.gflops_est));
            });
        }
    }

    fn ui_controls_tab(&mut self, ui: &mut Ui) {
        ui.heading("Interactive Controls");
        ui.label(RichText::new("Camera Controls:").strong());
        ui.label("• Orbit: Right-Click + Drag (or Left-Click + Drag on empty space)");
        ui.label("• Pan: Middle-Click + Drag (or Shift + Right-Click)");
        ui.label("• Zoom: Mouse Scroll Wheel");

        ui.add_space(8.0);
        ui.label(RichText::new("Fluid Mouse Tool:").strong());
        ui.horizontal(|ui| {
            ui.radio_value(&mut self.mouse_interaction_mode, MouseInteractionMode::Repel, "Push / Repel");
            ui.radio_value(&mut self.mouse_interaction_mode, MouseInteractionMode::Attract, "Pull / Attract");
        });
        ui.label("• Left-Click + Drag on fluid exerts 3D physical force!");
        ui.label("• Ctrl + Left-Click sprays continuous high-speed fluid particles!");

        ui.add_space(8.0);
        ui.label(RichText::new("Keyboard Shortcuts:").strong());
        ui.label("• Space: Pause / Resume simulation");
        ui.label("• Period (.): Single physics step");
        ui.label("• R: Reset current scenario");
        ui.label("• 1 - 4: Switch Render Styles");
    }
}
