use crate::{
    renderer::{Renderer, View},
    worker::{Command, Config, Snapshot, Worker},
};
use eframe::{
    egui::{self, Color32, RichText, Stroke, Vec2},
    glow,
};
use fluid_observatory::{
    cases::Scene,
    physics::{Boundary, Model},
};
use glam::Vec3;
use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

const INK: Color32 = Color32::from_rgb(9, 16, 25);
const PANEL: Color32 = Color32::from_rgb(13, 22, 32);
const LINE: Color32 = Color32::from_rgb(34, 53, 66);
const MUTED: Color32 = Color32::from_rgb(130, 152, 166);
const TEXT: Color32 = Color32::from_rgb(224, 233, 234);
const CYAN: Color32 = Color32::from_rgb(90, 224, 209);
const AMBER: Color32 = Color32::from_rgb(246, 187, 114);

pub struct Observatory {
    worker: Worker,
    snapshot: Option<Snapshot>,
    renderer: Arc<Mutex<Renderer>>,
    config: Config,
    view: View,
    paused: bool,
    rate: f64,
    orbit: bool,
    streamlines: bool,
    help: bool,
    focus: bool,
    fps: f32,
    last_frame: Instant,
    capture: Option<String>,
    capture_after: f64,
    capture_requested: bool,
    quit_after_capture: bool,
    frames: u64,
    status: Option<String>,
    tour: bool,
    tour_started: Instant,
    smoke: bool,
    smoke_stage: usize,
    start: Instant,
    probe: Option<[f64; 3]>,
    completion: Arc<Mutex<Option<Result<(), String>>>>,
}

pub struct Launch {
    pub config: Config,
    pub paused: bool,
    pub capture: Option<String>,
    pub capture_after: f64,
    pub smoke: bool,
    pub completion: Arc<Mutex<Option<Result<(), String>>>>,
}

impl Observatory {
    pub fn new(cc: &eframe::CreationContext<'_>, launch: Launch) -> Result<Self, String> {
        let mut style = (*cc.egui_ctx.style_of(egui::Theme::Dark)).clone();
        style.visuals = egui::Visuals::dark();
        style.visuals.panel_fill = INK;
        style.visuals.window_fill = PANEL;
        style.visuals.override_text_color = Some(TEXT);
        style.visuals.selection.bg_fill = Color32::from_rgb(30, 89, 89);
        style.visuals.selection.stroke = Stroke::new(1.0, CYAN);
        style.visuals.widgets.inactive.bg_fill = Color32::from_rgb(21, 34, 46);
        style.visuals.widgets.inactive.weak_bg_fill = Color32::from_rgb(21, 34, 46);
        style.visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, LINE);
        style.visuals.widgets.hovered.bg_fill = Color32::from_rgb(31, 66, 73);
        style.visuals.widgets.active.bg_fill = Color32::from_rgb(36, 94, 93);
        style.spacing.item_spacing = Vec2::new(10.0, 10.0);
        style.spacing.button_padding = Vec2::new(12.0, 9.0);
        style.spacing.slider_width = 104.0;
        style
            .text_styles
            .insert(egui::TextStyle::Body, egui::FontId::proportional(14.0));
        style
            .text_styles
            .insert(egui::TextStyle::Button, egui::FontId::proportional(14.0));
        style
            .text_styles
            .insert(egui::TextStyle::Small, egui::FontId::proportional(11.0));
        cc.egui_ctx.set_theme(egui::Theme::Dark);
        cc.egui_ctx.set_style_of(egui::Theme::Dark, style);
        let gl = cc.gl.as_ref().ok_or("OpenGL context unavailable")?;
        let renderer = Arc::new(Mutex::new(Renderer::new(gl)?));
        let worker = Worker::new(launch.config, launch.paused);
        let quit_after_capture = launch.capture.is_some();
        let mut app = Self {
            worker,
            snapshot: None,
            renderer,
            config: launch.config,
            view: View::default(),
            paused: launch.paused,
            rate: 0.3,
            orbit: false,
            streamlines: false,
            help: false,
            focus: false,
            fps: 60.0,
            last_frame: Instant::now(),
            capture: launch.capture,
            capture_after: launch.capture_after,
            capture_requested: false,
            quit_after_capture,
            frames: 0,
            status: None,
            tour: false,
            tour_started: Instant::now(),
            smoke: launch.smoke,
            smoke_stage: 0,
            start: Instant::now(),
            probe: None,
            completion: launch.completion,
        };
        app.view.field = match app.config.scene {
            Scene::Shear | Scene::Shock => 0,
            _ => 1,
        };
        app.set_range();
        Ok(app)
    }
    fn reset(&mut self) {
        self.worker.send(Command::Reset(self.config));
        self.set_range();
        self.tour_started = Instant::now();
        self.status = None;
    }
    fn select(&mut self, scene: Scene) {
        self.config.scene = scene;
        self.config.model = scene.default_model();
        self.view.field = match scene {
            Scene::Shear | Scene::Shock => 0,
            _ => 1,
        };
        self.view.beaming = false;
        self.view.cut = 1.0;
        self.reset();
    }
    fn set_range(&mut self) {
        if self.view.field == 3 && self.config.model == Model::Newtonian {
            self.view.field = 2;
        }
        self.view.range = match self.view.field {
            0 => match self.config.scene {
                Scene::Shear => [1.0, 2.0],
                Scene::Shock => [0.125, 1.0],
                _ => [0.1, 2.5],
            },
            1 => match self.config.scene {
                Scene::Jet => [0.02, 0.6],
                Scene::Blast => [0.015, 2.0],
                Scene::Shear => [0.26, 0.4],
                Scene::Shock => [0.1, 1.0],
            },
            2 => [
                0.0,
                if self.config.model == Model::Relativistic {
                    1.0
                } else {
                    1.5
                },
            ],
            _ => [1.0, 3.0],
        };
        self.view.logarithmic = self.view.field < 2;
    }
    fn toggle_pause(&mut self) {
        self.paused = !self.paused;
        self.worker.send(Command::Pause(self.paused));
    }
    fn top(&mut self, ui: &mut egui::Ui) {
        egui::Panel::top("masthead")
            .exact_size(78.0)
            .frame(
                egui::Frame::new()
                    .fill(INK)
                    .inner_margin(egui::Margin::symmetric(25, 17)),
            )
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    let (r, _) = ui.allocate_exact_size(Vec2::splat(39.0), egui::Sense::hover());
                    let p = ui.painter();
                    for i in 0..4 {
                        let y = r.top() + 8.0 + i as f32 * 7.0;
                        let pts = (0..26)
                            .map(|j| {
                                let x = j as f32 / 25.0;
                                egui::pos2(
                                    r.left() + x * 34.0,
                                    y + (x * 6.2 + i as f32 * 0.55).sin() * 3.0,
                                )
                            })
                            .collect();
                        p.add(egui::Shape::line(pts, Stroke::new(1.5, CYAN)));
                    }
                    ui.vertical(|ui| {
                        ui.label(RichText::new("FLUID OBSERVATORY").size(19.0).strong());
                        ui.label(
                            RichText::new("A living atlas of motion")
                                .size(12.0)
                                .color(MUTED),
                        );
                    });
                    ui.add_space(24.0);
                    ui.label(
                        RichText::new("CLASSICAL  /  RELATIVISTIC")
                            .size(10.0)
                            .color(CYAN),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("?   Field guide").clicked() {
                            self.help = !self.help;
                        }
                        if ui.selectable_label(self.tour, "Guided tour").clicked() {
                            self.tour = !self.tour;
                            self.tour_started = Instant::now();
                            if self.tour && self.paused {
                                self.toggle_pause();
                            }
                        }
                        ui.label(RichText::new("RUST  /  3D LAB").size(10.0).color(MUTED));
                    });
                });
            });
    }
    fn sidebar(&mut self, ui: &mut egui::Ui) {
        egui::Panel::left("controls").exact_size(292.0).resizable(false).frame(egui::Frame::new().fill(PANEL).inner_margin(20)).show(ui,|ui|{
            egui::ScrollArea::vertical().show(ui,|ui|{
                eyebrow(ui,"CHOOSE AN EXPERIMENT");ui.add_space(5.0);
                for scene in Scene::ALL{
                    let active=self.config.scene==scene;
                    let response=ui.add_sized([ui.available_width(),43.0],egui::Button::new(RichText::new(scene.label()).color(if active{CYAN}else{TEXT})).fill(if active{Color32::from_rgb(24,57,61)}else{PANEL}).stroke(Stroke::new(1.0,if active{Color32::from_rgb(53,113,110)}else{LINE})).corner_radius(7));
                    if response.clicked(){self.select(scene);}
                }
                ui.add_space(12.0);eyebrow(ui,"LAWS OF MOTION");
                ui.horizontal(|ui|{let mut model=self.config.model;ui.selectable_value(&mut model,Model::Newtonian,"Newtonian");ui.selectable_value(&mut model,Model::Relativistic,"Relativistic");if model!=self.config.model{self.config.model=model;self.view.beaming=false;self.reset();}});
                ui.label(RichText::new(if self.config.model==Model::Relativistic{"c = 1  ·  flat spacetime  ·  causal sound speed"}else{"Dimensionless units  ·  compressible ideal gas"}).size(11.0).color(MUTED));
                ui.add_space(12.0);eyebrow(ui,"LOOK INSIDE");
                let old=self.view.field;
                egui::ComboBox::from_id_salt("field").selected_text(field_name(self.view.field)).width(ui.available_width()-12.0).show_ui(ui,|ui|{for k in 0..4{ui.add_enabled_ui(k!=3||self.config.model==Model::Relativistic,|ui|{ui.selectable_value(&mut self.view.field,k,field_name(k));});}});
                if old!=self.view.field{self.set_range();}
                ui.add(egui::Slider::new(&mut self.view.exposure,0.2..=3.0).text("Opacity"));
                ui.add(egui::Slider::new(&mut self.view.cut,-0.95..=1.0).text("Slice z"));
                ui.horizontal(|ui|{ui.checkbox(&mut self.orbit,"Orbit");ui.checkbox(&mut self.streamlines,"Streamlines");});
                ui.horizontal(|ui|{ui.label(RichText::new("Render").size(11.0).color(MUTED));ui.selectable_value(&mut self.view.samples,144,"Balanced");ui.selectable_value(&mut self.view.samples,208,"Cinematic");});
                if self.config.model==Model::Relativistic{ui.checkbox(&mut self.view.beaming,"Doppler beaming").on_hover_text("Illustrative emission: brightness × δ³, δ = 1 / [W (1 − v · n)]. No radiation transport or light-travel delay.");}
                ui.add_space(10.0);eyebrow(ui,"TIME & RESOLUTION");
                if ui.add(egui::Slider::new(&mut self.rate,0.05..=1.0).logarithmic(true).text("Time / s")).changed(){self.worker.send(Command::Rate(self.rate));}
                ui.label(RichText::new("Requested simulation units per wall second; compute speed may limit it.").size(11.0).color(MUTED));
                ui.horizontal(|ui|{
                    let mut resolution=self.config.resolution;
                    egui::ComboBox::from_id_salt("resolution").selected_text(match resolution{20=>"Fast · 12k",28=>"Balanced · 33k",40=>"Fine · 96k",_=>"Custom"}).show_ui(ui,|ui|{for(n,s)in[(20,"Fast · 12,000 cells"),(28,"Balanced · 32,928 cells"),(40,"Fine · 96,000 cells")]{ui.selectable_value(&mut resolution,n,s);}});
                    if resolution!=self.config.resolution{self.config.resolution=resolution;self.reset();}
                });
                if self.config.scene==Scene::Jet{
                    ui.add_space(8.0);let old=self.config.jet_speed;
                    let response=ui.add(egui::Slider::new(&mut self.config.jet_speed,0.2..=0.98).text(if self.config.model==Model::Relativistic{"Jet v / c"}else{"Jet speed"}));
                    if response.drag_stopped()||(!response.dragged()&&old!=self.config.jet_speed){self.reset();}
                }
                ui.add_space(10.0);ui.separator();
                if ui.add_sized([ui.available_width(),35.0],egui::Button::new("+   Deposit a pulse of energy")).clicked(){self.worker.send(Command::Heat([0.0;3]));}
                ui.label(RichText::new("Or double-click the volume to heat the z = 0 plane.").size(11.0).color(MUTED));
                ui.horizontal(|ui|{
                    if ui.button("Save PNG").clicked(){self.capture=Some(capture_name());self.capture_requested=false;self.capture_after=0.0;self.quit_after_capture=false;}
                    if ui.button("Export VTK").clicked(){self.worker.send(Command::Export);}
                });
            });
        });
    }
    fn bottom(&mut self, ui: &mut egui::Ui) {
        egui::Panel::bottom("telemetry")
            .exact_size(150.0)
            .frame(
                egui::Frame::new()
                    .fill(INK)
                    .inner_margin(egui::Margin::symmetric(24, 15)),
            )
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    if ui
                        .add_sized(
                            [97.0, 34.0],
                            egui::Button::new(
                                RichText::new(if self.paused {
                                    "▶  Resume"
                                } else {
                                    "Ⅱ  Pause"
                                })
                                .color(CYAN),
                            ),
                        )
                        .clicked()
                    {
                        self.toggle_pause();
                    }
                    if ui.button("Step").clicked() {
                        self.paused = true;
                        self.worker.send(Command::Step);
                    }
                    if ui.button("↺  Reset").clicked() {
                        self.reset();
                    }
                    ui.add_space(12.0);
                    if let Some(s) = &self.snapshot {
                        ui.label(
                            RichText::new(format!("t = {:06.3}", s.grid.time))
                                .monospace()
                                .size(18.0),
                        );
                        ui.label(
                            RichText::new(if s.grid.model == Model::Relativistic {
                                "L₀ / c"
                            } else {
                                "L₀ / v₀"
                            })
                            .color(MUTED),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(
                                RichText::new(format!(
                                    "{:.0} FPS   /   {:.1} ms solve   /   step {}",
                                    self.fps, s.ms, s.grid.steps
                                ))
                                .monospace()
                                .size(11.0)
                                .color(MUTED),
                            );
                        });
                    }
                });
                ui.add_space(10.0);
                if let Some(s) = &self.snapshot {
                    let (v, w, _, _) = s.extrema();
                    let residual = s.grid.residual();
                    ui.columns(4, |c| {
                        metric(
                            &mut c[0],
                            if s.grid.model == Model::Relativistic {
                                "PEAK SPEED / c"
                            } else {
                                "PEAK SPEED / v₀"
                            },
                            format!("{v:.4}"),
                            "computed from all cells",
                        );
                        metric(
                            &mut c[1],
                            if s.grid.model == Model::Relativistic {
                                "PEAK LORENTZ FACTOR"
                            } else {
                                "ADIABATIC INDEX"
                            },
                            format!(
                                "{:.4}",
                                if s.grid.model == Model::Relativistic {
                                    w
                                } else {
                                    s.grid.gamma
                                }
                            ),
                            if s.grid.model == Model::Relativistic {
                                "local clock rate = 1 / W"
                            } else {
                                "p = (Γ − 1) ρ ε"
                            },
                        );
                        metric(
                            &mut c[2],
                            "MASS BUDGET RESIDUAL",
                            format!("{:.1e}", residual[0].abs()),
                            "accounts for boundary flux",
                        );
                        metric(
                            &mut c[3],
                            "ENERGY BUDGET RESIDUAL",
                            format!("{:.1e}", residual[4].abs()),
                            "accounts for flux & heat pulses",
                        );
                    });
                }
            });
    }
    fn viewport(&mut self, ui: &mut egui::Ui, gl: &glow::Context) {
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(INK).inner_margin(0))
            .show(ui, |ui| {
                let rect = ui.max_rect();
                let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());
                if response.dragged() {
                    let delta = response.drag_delta();
                    self.view.yaw -= delta.x * 0.007;
                    self.view.pitch = (self.view.pitch + delta.y * 0.007).clamp(-0.8, 1.2);
                    self.orbit = false;
                }
                if response.hovered() {
                    let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                    self.view.distance =
                        (self.view.distance * (-scroll * 0.0015).exp()).clamp(2.5, 10.0);
                }
                if response.clicked()
                    && let Some(pos) = response.interact_pointer_pos()
                {
                    self.probe = self.view.hit_plane(pos, rect);
                }
                if response.double_clicked()
                    && let Some(pos) = response.interact_pointer_pos()
                    && let Some(point) = self.view.hit_plane(pos, rect)
                {
                    self.worker.send(Command::Heat(point));
                    self.probe = Some(point);
                }
                if self.orbit {
                    self.view.yaw += ui.input(|i| i.stable_dt).min(0.05) * 0.11;
                }
                if let Some(s) = &self.snapshot {
                    self.renderer.lock().unwrap().upload(gl, s);
                    let renderer = self.renderer.clone();
                    let view = self.view.clone();
                    let aspect = rect.aspect_ratio();
                    let callback = eframe::egui_glow::CallbackFn::new(move |_, painter| {
                        renderer.lock().unwrap().paint(painter.gl(), &view, aspect);
                    });
                    ui.painter().add(egui::PaintCallback {
                        rect,
                        callback: Arc::new(callback),
                    });
                    self.cage(ui, rect);
                    if self.streamlines {
                        self.lines(ui, rect, s);
                    }
                    self.overlay(ui, rect, s);
                    if let Some(point) = self.probe {
                        self.draw_probe(ui, rect, s, point);
                    }
                } else {
                    ui.painter().text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        "Preparing the fluid…",
                        egui::FontId::proportional(20.0),
                        MUTED,
                    );
                }
            });
    }
    fn cage(&self, ui: &egui::Ui, rect: egui::Rect) {
        for axis in 0..3 {
            for a in [-1.0, 1.0] {
                for b in [-1.0, 1.0] {
                    let mut l = [0.0; 3];
                    let mut r = [0.0; 3];
                    l[axis] = -1.0;
                    r[axis] = 1.0;
                    l[(axis + 1) % 3] = a;
                    r[(axis + 1) % 3] = a;
                    l[(axis + 2) % 3] = b;
                    r[(axis + 2) % 3] = b;
                    let scale = Vec3::new(1.5, 1.0, 1.0);
                    if let (Some(p), Some(q)) = (
                        self.view.project(Vec3::from_array(l) * scale, rect),
                        self.view.project(Vec3::from_array(r) * scale, rect),
                    ) {
                        ui.painter().line_segment(
                            [p, q],
                            Stroke::new(0.65, Color32::from_rgba_unmultiplied(112, 181, 195, 85)),
                        );
                    }
                }
            }
        }
        for (p, label, color) in [
            (Vec3::new(1.6, -1.0, 1.0), "x", AMBER),
            (Vec3::new(-1.5, 1.13, 1.0), "y", CYAN),
            (Vec3::new(-1.5, -1.0, 1.15), "z", MUTED),
        ] {
            if let Some(p) = self.view.project(p, rect) {
                ui.painter().text(
                    p,
                    egui::Align2::CENTER_CENTER,
                    label,
                    egui::FontId::monospace(12.0),
                    color,
                );
            }
        }
    }
    fn draw_probe(&self, ui: &egui::Ui, rect: egui::Rect, s: &Snapshot, point: [f64; 3]) {
        let c = std::array::from_fn(|a| {
            ((point[a] / s.grid.length[a] + 0.5) * s.grid.n[a] as f64)
                .floor()
                .clamp(0.0, (s.grid.n[a] - 1) as f64) as usize
        });
        let cell = s.fields[s.grid.index(c)];
        if let Some(p) = self
            .view
            .project(Vec3::from_array(point.map(|v| v as f32)), rect)
        {
            ui.painter().circle_stroke(p, 5.0, Stroke::new(1.5, AMBER));
        }
        let text = format!(
            "PROBE  ({:.2}, {:.2}, 0)\nρ = {:.4}   p = {:.4}\n|v| = {:.4}{}",
            point[0],
            point[1],
            cell.rho,
            cell.p,
            cell.speed2().sqrt(),
            if s.grid.model == Model::Relativistic {
                format!(
                    " c\nW = {:.4}   dτ/dt = {:.4}",
                    cell.lorentz(),
                    1.0 / cell.lorentz()
                )
            } else {
                " v₀".into()
            }
        );
        let galley = ui
            .painter()
            .layout_no_wrap(text, egui::FontId::monospace(11.0), AMBER);
        let pos = egui::pos2(rect.right() - galley.size().x - 25.0, rect.top() + 142.0);
        ui.painter().rect_filled(
            egui::Rect::from_min_size(pos, galley.size()).expand(12.0),
            7,
            Color32::from_rgba_unmultiplied(8, 18, 28, 230),
        );
        ui.painter().galley(pos, galley, AMBER);
    }
    fn lines(&self, ui: &egui::Ui, rect: egui::Rect, s: &Snapshot) {
        // Instantaneous streamlines, integrated in the actual sampled velocity field.
        for j in 0..7 {
            for k in 0..5 {
                let mut pos = Vec3::new(-1.40, -0.6 + j as f32 * 0.2, -0.45 + k as f32 * 0.225);
                let mut points = Vec::new();
                for _ in 0..95 {
                    if pos.x.abs() > 1.5
                        || pos.y.abs() > 1.0
                        || pos.z.abs() > 1.0
                        || pos.z > self.view.cut
                    {
                        break;
                    }
                    if let Some(p) = self.view.project(pos, rect) {
                        points.push(p);
                    }
                    let v = sample_velocity(s, pos);
                    if v.length() < 0.005 {
                        break;
                    }
                    let mid = pos + v.normalize() * 0.0175;
                    let vm = sample_velocity(s, mid);
                    if vm.length() < 0.005 {
                        break;
                    }
                    pos += vm.normalize() * 0.035;
                }
                if points.len() > 2 {
                    ui.painter().add(egui::Shape::line(
                        points,
                        Stroke::new(0.8, Color32::from_rgba_unmultiplied(161, 248, 233, 95)),
                    ));
                }
            }
        }
    }
    fn overlay(&self, ui: &egui::Ui, rect: egui::Rect, s: &Snapshot) {
        let painter = ui.painter();
        let x = rect.left() + 28.0;
        let y = rect.top() + 24.0;
        painter.text(
            egui::pos2(x, y),
            egui::Align2::LEFT_TOP,
            format!(
                "EXPERIMENT {}   /   {}",
                self.config.scene.id().to_uppercase(),
                if self.paused { "PAUSED" } else { "LIVE" }
            ),
            egui::FontId::monospace(11.0),
            CYAN,
        );
        painter.text(
            egui::pos2(x, y + 26.0),
            egui::Align2::LEFT_TOP,
            self.config.scene.title(),
            egui::FontId::proportional(29.0),
            TEXT,
        );
        if !self.focus {
            let galley = painter.layout(
                self.config.scene.description().into(),
                egui::FontId::proportional(13.0),
                MUTED,
                (rect.width() - 65.0).min(555.0),
            );
            painter.galley(egui::pos2(x, y + 66.0), galley, TEXT);
        }
        let label = format!(
            "{} × {} × {}   /   {}",
            s.grid.n[0],
            s.grid.n[1],
            s.grid.n[2],
            match s.grid.boundary {
                Boundary::Periodic => "PERIODIC",
                Boundary::Outflow => "OPEN BOUNDARIES",
                Boundary::Jet => "INFLOW / OUTFLOW",
            }
        );
        painter.text(
            egui::pos2(rect.right() - 24.0, rect.bottom() - 20.0),
            egui::Align2::RIGHT_BOTTOM,
            label,
            egui::FontId::monospace(10.0),
            MUTED,
        );
        let legend =
            egui::Rect::from_min_size(egui::pos2(x, rect.bottom() - 79.0), Vec2::new(210.0, 7.0));
        for i in 0..100 {
            let t = i as f32 / 99.0;
            painter.rect_filled(
                egui::Rect::from_min_size(
                    egui::pos2(legend.left() + t * 208.0, legend.top()),
                    Vec2::new(3.0, 7.0),
                ),
                0,
                palette(t),
            );
        }
        painter.text(
            legend.left_top() - Vec2::new(0.0, 10.0),
            egui::Align2::LEFT_BOTTOM,
            format!(
                "{}  ·  {}",
                field_name(self.view.field),
                if self.view.logarithmic {
                    "log scale"
                } else {
                    "linear"
                }
            ),
            egui::FontId::proportional(11.0),
            TEXT,
        );
        painter.text(
            legend.left_bottom() + Vec2::new(0.0, 7.0),
            egui::Align2::LEFT_TOP,
            format!("{:.3}", self.view.range[0]),
            egui::FontId::monospace(10.0),
            MUTED,
        );
        painter.text(
            legend.right_bottom() + Vec2::new(0.0, 7.0),
            egui::Align2::RIGHT_TOP,
            format!("{:.3}", self.view.range[1]),
            egui::FontId::monospace(10.0),
            MUTED,
        );
        painter.text(
            egui::pos2(x, rect.bottom() - 20.0),
            egui::Align2::LEFT_BOTTOM,
            "DRAG  orbit     SCROLL  zoom     SPACE  pause     F  immerse",
            egui::FontId::monospace(10.0),
            MUTED,
        );
        if rect.width() > 680.0 && !self.focus {
            self.profile(
                ui,
                egui::Rect::from_min_size(
                    egui::pos2(rect.right() - 280.0, rect.bottom() - 139.0),
                    Vec2::new(250.0, 85.0),
                ),
                s,
            );
        }
        if let Some(e) = &s.error {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                e,
                egui::FontId::proportional(16.0),
                AMBER,
            );
        }
        if self.view.beaming {
            painter.text(
                egui::pos2(rect.right() - 25.0, y),
                egui::Align2::RIGHT_TOP,
                "δ³  ILLUSTRATIVE BEAMING",
                egui::FontId::monospace(10.0),
                AMBER,
            );
        }
        if let Some(notice) = self.status.as_ref().or(s.notice.as_ref()) {
            painter.text(
                egui::pos2(x, y + 112.0),
                egui::Align2::LEFT_TOP,
                notice,
                egui::FontId::proportional(11.0),
                AMBER,
            );
        }
    }
    fn profile(&self, ui: &egui::Ui, rect: egui::Rect, s: &Snapshot) {
        let p = ui.painter();
        p.rect_filled(
            rect.expand(10.0),
            7,
            Color32::from_rgba_unmultiplied(8, 18, 28, 210),
        );
        p.text(
            rect.left_top(),
            egui::Align2::LEFT_TOP,
            "DENSITY ALONG x  /  y ≈ z ≈ 0",
            egui::FontId::monospace(9.0),
            MUTED,
        );
        let vals: Vec<f64> = (0..s.grid.n[0])
            .map(|x| s.fields[s.grid.index([x, s.grid.n[1] / 2, s.grid.n[2] / 2])].rho)
            .collect();
        let max = vals.iter().copied().fold(0.0, f64::max).max(0.01);
        let plot =
            egui::Rect::from_min_max(rect.left_top() + Vec2::new(0.0, 23.0), rect.right_bottom());
        for j in 0..3 {
            let y = plot.top() + j as f32 * plot.height() / 2.0;
            p.line_segment(
                [egui::pos2(plot.left(), y), egui::pos2(plot.right(), y)],
                Stroke::new(0.5, LINE),
            );
        }
        let pts = vals
            .iter()
            .enumerate()
            .map(|(i, v)| {
                egui::pos2(
                    plot.left() + i as f32 / (vals.len() - 1) as f32 * plot.width(),
                    plot.bottom() - (*v / max) as f32 * plot.height(),
                )
            })
            .collect();
        p.add(egui::Shape::line(pts, Stroke::new(1.5, CYAN)));
        p.text(
            rect.right_top(),
            egui::Align2::RIGHT_TOP,
            format!("0–{max:.2}"),
            egui::FontId::monospace(9.0),
            CYAN,
        );
    }
    fn guide(&mut self, ctx: &egui::Context) {
        egui::Window::new("Field guide").open(&mut self.help).default_width(560.0).resizable(true).show(ctx,|ui|{
            egui::ScrollArea::vertical().max_height(610.0).show(ui,|ui|{
                ui.heading("One fluid. Two descriptions of motion.");
                ui.label("Every voxel evolves five conserved quantities on a three-dimensional Cartesian grid. Pressure waves, shocks and vortices arise from the equations; the image samples those computed fields.");
                ui.add_space(10.0);eyebrow(ui,"NEWTONIAN EULER");ui.monospace("U = (ρ, ρv, E),   E = p/(Γ−1) + ½ρ|v|²\n∂t U + ∇·F(U) = 0");
                ui.label("The inviscid limit of compressible fluid dynamics. Γ describes an ideal gas, not incompressible water. Both modes use the same initial primitive fields when you switch the laws.");
                ui.add_space(10.0);eyebrow(ui,"SPECIAL RELATIVISTIC EULER  ·  c = 1");ui.monospace("W = 1 / √(1−|v|²),   h = 1 + Γp/[(Γ−1)ρ]\nD = ρW,   S = ρhW²v,   E = ρhW² − p\ndτ / dt = 1/W");
                ui.label("Rest mass and energy–momentum are conserved. Pressure also contributes to inertia. Characteristic speeds include transverse motion and stay inside the light cone. Total E includes rest mass energy in this mode.");
                ui.add_space(10.0);eyebrow(ui,"READING THE PICTURE");ui.label("Color is a scalar-field visualization, not a photograph of gas. The fixed legend shows clipping limits. Volume opacity and lighting emphasize structure. The density trace is a direct centerline sample; streamlines are instantaneous paths through the velocity field, not material particles. Slice z removes the near part of the box.");
                ui.label("Optional Doppler beaming multiplies illustrative brightness by δ³ for a fixed snapshot, with δ = 1/[W(1−v·n)]. It omits light-travel delays, spectral transport, absorption physics and radiation feedback. It does not change the dynamics.");
                ui.add_space(10.0);eyebrow(ui,"NUMERICAL CONTRACT");ui.label("Finite volumes; HLLC Newtonian / HLLE relativistic fluxes; minmod MUSCL reconstruction; unsplit SSP-RK2; multidimensional CFL 0.35. Invalid stages retry at smaller steps, with first-order fallback. No hidden density, pressure or velocity clamps. Open-boundary flux and user-deposited heat are included in conservation budgets.");
                ui.label("This is a finite-resolution educational laboratory. Numerical diffusion broadens shocks and damps small vortices; increase the grid resolution to compare. There is no physical viscosity, surface tension, magnetic field, self-gravity, curved spacetime or general relativity.");
                ui.add_space(10.0);eyebrow(ui,"CONTROLS");ui.label("1–4 select experiment · Space pause · N single step · R reset\nDrag orbit · Scroll zoom · F immersive view · H field guide\nDouble-click deposit heat at z = 0 · Esc leave immersive view");
                ui.label("The guided tour moves through the four experiments every 35 seconds. Model, grid and nozzle changes restart the experiment. VTK exports retain the physical scalar fields and integral conservation budgets for external analysis.");
                ui.add_space(10.0);ui.label("Reference: Martí & Müller (2003), Numerical Hydrodynamics in Special Relativity, Living Reviews in Relativity 6, 7. Full equations, verification instructions and scope are in docs/SCIENCE.md.");
            });
        });
    }
    fn captures(&mut self, ctx: &egui::Context) {
        for event in ctx.input(|i| i.events.clone()) {
            if let egui::Event::Screenshot { image, .. } = event
                && self.capture_requested
            {
                if let Some(path) = self.capture.take() {
                    match save_png(&path, &image) {
                        Ok(()) => {
                            self.status = Some(format!("Saved {path}"));
                            println!("CAPTURE {path}");
                            if let Some(s) = &self.snapshot {
                                println!(
                                    "RENDER scene={} fps={:.1} solve_ms={:.2} cells={} t={:.4}",
                                    self.config.scene.id(),
                                    self.fps,
                                    s.ms,
                                    s.grid.u.len(),
                                    s.grid.time
                                );
                            }
                            if self.quit_after_capture {
                                *self.completion.lock().unwrap() = Some(Ok(()));
                            }
                        }
                        Err(e) => {
                            self.status = Some(format!("Capture failed: {e}"));
                            eprintln!("Capture failed: {e}");
                            if self.quit_after_capture {
                                *self.completion.lock().unwrap() = Some(Err(e.to_string()));
                            }
                        }
                    }
                }
                self.capture_requested = false;
                if self.quit_after_capture {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            }
        }
        if self.capture.is_some()
            && !self.capture_requested
            && self.frames > 8
            && self
                .snapshot
                .as_ref()
                .is_some_and(|s| s.grid.time >= self.capture_after)
        {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
            self.capture_requested = true;
        }
    }
    fn smoke(&mut self, ctx: &egui::Context) {
        if !self.smoke {
            return;
        }
        if self.start.elapsed().as_secs() > 100 {
            eprintln!("SMOKE FAILED: timed out");
            *self.completion.lock().unwrap() = Some(Err("smoke test timed out".into()));
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }
        if let Some(s) = &self.snapshot
            && s.grid.steps >= 12
            && self.smoke_stage < 4
        {
            println!(
                "SMOKE scene={} t={:.5} steps={} solve_ms={:.2} mass_residual={:.2e}",
                self.config.scene.id(),
                s.grid.time,
                s.grid.steps,
                s.ms,
                s.grid.residual()[0]
            );
            self.smoke_stage += 1;
            if self.smoke_stage < 4 {
                self.select(Scene::ALL[self.smoke_stage]);
                self.snapshot = None;
            } else {
                self.paused = true;
                self.worker.send(Command::Pause(true));
                self.view.cut = 0.0;
                self.streamlines = true;
                self.help = true;
                self.capture = Some(capture_name().replace("observatory-", "smoke-"));
                self.capture_after = 0.0;
                self.quit_after_capture = true;
                println!("SMOKE: four scenes evolved and rendered; capture pending");
            }
        }
    }
}

impl eframe::App for Observatory {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;
        self.frames += 1;
        if dt > 0.0 {
            self.fps = self.fps * 0.94 + 0.06 / dt;
        }
        if let Some(s) = self.worker.take() {
            if s.error.is_some() {
                self.paused = true;
                if self.quit_after_capture || self.smoke {
                    *self.completion.lock().unwrap() = Some(Err(s.error.clone().unwrap()));
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            }
            self.snapshot = Some(s);
        }
        if !ctx.egui_wants_keyboard_input() {
            for (key, scene) in [
                (egui::Key::Num1, Scene::Jet),
                (egui::Key::Num2, Scene::Blast),
                (egui::Key::Num3, Scene::Shear),
                (egui::Key::Num4, Scene::Shock),
            ] {
                if ctx.input(|i| i.key_pressed(key)) {
                    self.select(scene);
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Space)) {
                self.toggle_pause();
            }
            if ctx.input(|i| i.key_pressed(egui::Key::N)) {
                self.paused = true;
                self.worker.send(Command::Step);
            }
            if ctx.input(|i| i.key_pressed(egui::Key::R)) {
                self.reset();
            }
            if ctx.input(|i| i.key_pressed(egui::Key::F)) {
                self.focus = !self.focus;
            }
            if ctx.input(|i| i.key_pressed(egui::Key::H)) {
                self.help = !self.help;
            }
            if ctx.input(|i| i.key_pressed(egui::Key::P)) {
                self.capture = Some(capture_name());
                self.capture_requested = false;
                self.capture_after = 0.0;
                self.quit_after_capture = false;
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
                self.focus = false;
                self.help = false;
            }
        }
        if self.tour && !self.paused && self.tour_started.elapsed().as_secs() >= 35 {
            let i = Scene::ALL
                .iter()
                .position(|s| *s == self.config.scene)
                .unwrap();
            self.select(Scene::ALL[(i + 1) % 4]);
        }
        if !self.focus {
            self.top(ui);
            self.sidebar(ui);
            self.bottom(ui);
        }
        if let Some(gl) = frame.gl() {
            self.viewport(ui, gl);
        }
        self.guide(&ctx);
        self.smoke(&ctx);
        self.captures(&ctx);
        ctx.request_repaint();
    }
    fn on_exit(&mut self, gl: Option<&glow::Context>) {
        if let Some(gl) = gl {
            self.renderer.lock().unwrap().destroy(gl);
        }
    }
}

fn eyebrow(ui: &mut egui::Ui, s: &str) {
    ui.label(RichText::new(s).size(10.0).color(MUTED));
}
fn metric(ui: &mut egui::Ui, label: &str, value: String, note: &str) {
    ui.label(RichText::new(label).size(9.0).color(MUTED));
    ui.label(RichText::new(value).monospace().size(20.0).color(CYAN));
    ui.label(RichText::new(note).size(10.0).color(MUTED));
}
fn field_name(k: i32) -> &'static str {
    match k {
        0 => "Proper density ρ",
        1 => "Pressure p",
        2 => "Speed |v|",
        _ => "Lorentz factor W",
    }
}
fn palette(t: f32) -> Color32 {
    let (a, b, u) = if t < 0.38 {
        ([0.035, 0.14, 0.24], [0.06, 0.77, 0.81], t / 0.38)
    } else if t < 0.7 {
        ([0.06, 0.77, 0.81], [0.56, 0.35, 0.86], (t - 0.38) / 0.32)
    } else {
        ([0.56, 0.35, 0.86], [1.0, 0.68, 0.32], (t - 0.7) / 0.3)
    };
    let u = u * u * (3.0 - 2.0 * u);
    let c: [u8; 3] = std::array::from_fn(|i| ((a[i] + u * (b[i] - a[i])) * 255.0) as u8);
    Color32::from_rgb(c[0], c[1], c[2])
}
fn sample_velocity(s: &Snapshot, pos: Vec3) -> Vec3 {
    let n = s.grid.n;
    let p = (pos / Vec3::new(3.0, 2.0, 2.0) + Vec3::splat(0.5))
        * Vec3::new(n[0] as f32, n[1] as f32, n[2] as f32)
        - Vec3::splat(0.5);
    let base = p.floor();
    let f = p - base;
    let mut v = Vec3::ZERO;
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                let d = [x, y, z];
                let c = std::array::from_fn(|a| {
                    (base[a] as isize + d[a]).clamp(0, n[a] as isize - 1) as usize
                });
                let weight: f32 = (0..3)
                    .map(|a| if d[a] == 0 { 1.0 - f[a] } else { f[a] })
                    .product();
                let wi = s.fields[s.grid.index(c)];
                v += Vec3::from_array(wi.v.map(|x| x as f32)) * weight;
            }
        }
    }
    v
}
fn capture_name() -> String {
    format!(
        "captures/observatory-{}.png",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    )
}
fn save_png(path: &str, image: &egui::ColorImage) -> Result<(), Box<dyn std::error::Error>> {
    let path = std::path::Path::new(path);
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    let mut encoder = png::Encoder::new(
        std::io::BufWriter::new(file),
        image.size[0] as u32,
        image.size[1] as u32,
    );
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    let bytes: Vec<u8> = image.pixels.iter().flat_map(|p| p.to_array()).collect();
    writer.write_image_data(&bytes)?;
    Ok(())
}
