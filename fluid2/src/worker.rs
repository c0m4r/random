use fluid_observatory::{
    cases::{self, Scene},
    physics::{Grid, Model, Primitive, State},
};
use std::{
    sync::{Arc, Mutex, mpsc},
    thread,
    time::{Duration, Instant},
};

#[derive(Clone, Copy)]
pub struct Config {
    pub scene: Scene,
    pub model: Model,
    pub resolution: usize,
    pub jet_speed: f64,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            scene: Scene::Jet,
            model: Model::Relativistic,
            resolution: 28,
            jet_speed: 0.92,
        }
    }
}

pub enum Command {
    Reset(Config),
    Pause(bool),
    Step,
    Rate(f64),
    Heat([f64; 3]),
    Export,
    Quit,
}
pub struct Snapshot {
    pub grid: Grid,
    pub fields: Vec<Primitive>,
    pub ms: f64,
    pub error: Option<String>,
    pub notice: Option<String>,
    pub revision: u64,
}
impl Snapshot {
    pub fn extrema(&self) -> (f64, f64, f64, f64) {
        let mut v = 0.0_f64;
        let mut w = 1.0_f64;
        let mut p = 0.0_f64;
        let mut rho = 0.0_f64;
        for f in &self.fields {
            v = v.max(f.speed2().sqrt());
            if self.grid.model == Model::Relativistic {
                w = w.max(f.lorentz());
            }
            p = p.max(f.p);
            rho = rho.max(f.rho);
        }
        (v, w, p, rho)
    }
    pub fn texture(&self) -> Vec<[f32; 4]> {
        self.fields
            .iter()
            .map(|w| {
                [
                    w.rho as f32,
                    w.p as f32,
                    w.speed2().sqrt() as f32,
                    if self.grid.model == Model::Relativistic {
                        w.lorentz() as f32
                    } else {
                        1.0
                    },
                ]
            })
            .collect()
    }
    pub fn velocity_texture(&self) -> Vec<[f32; 4]> {
        self.fields
            .iter()
            .map(|w| [w.v[0] as f32, w.v[1] as f32, w.v[2] as f32, 0.0])
            .collect()
    }
}

pub struct Worker {
    pub tx: mpsc::Sender<Command>,
    pub latest: Arc<Mutex<Option<Snapshot>>>,
    handle: Option<thread::JoinHandle<()>>,
}
impl Worker {
    pub fn new(config: Config, paused: bool) -> Self {
        let (tx, rx) = mpsc::channel();
        let latest = Arc::new(Mutex::new(None));
        let out = latest.clone();
        let handle = thread::spawn(move || {
            let mut grid = cases::build(
                config.scene,
                config.model,
                config.resolution,
                config.jet_speed,
            );
            let mut paused = paused;
            let mut rate = 0.3;
            let mut error = None;
            let mut notice = None;
            let mut ms = 0.0;
            let mut revision = 0;
            let mut publish = true;
            let mut next_step = Instant::now();
            loop {
                let begin = Instant::now();
                let mut single = false;
                for cmd in rx.try_iter() {
                    match cmd {
                        Command::Quit => return,
                        Command::Reset(c) => {
                            grid = cases::build(c.scene, c.model, c.resolution, c.jet_speed);
                            error = None;
                            notice = None;
                            publish = true;
                            next_step = Instant::now();
                        }
                        Command::Pause(p) => {
                            paused = p;
                            next_step = Instant::now();
                        }
                        Command::Step => {
                            single = true;
                            paused = true;
                        }
                        Command::Rate(r) => {
                            rate = r;
                            next_step = Instant::now();
                        }
                        Command::Heat(c) => {
                            grid.deposit_heat(c, 0.8);
                            publish = true;
                            notice = Some(
                                "Energy deposited · source included in conservation budget".into(),
                            );
                        }
                        Command::Export => {
                            notice = Some(match export_vtk(&grid) {
                                Ok(p) => format!("Saved {p}"),
                                Err(e) => format!("Export failed: {e}"),
                            });
                            publish = true;
                        }
                    }
                }
                if (!paused || single) && (single || Instant::now() >= next_step) && error.is_none()
                {
                    let timer = Instant::now();
                    match grid.step(0.1) {
                        Ok(d) => next_step = begin + Duration::from_secs_f64(d / rate),
                        Err(e) => {
                            error = Some(e);
                            paused = true;
                        }
                    }
                    ms = timer.elapsed().as_secs_f64() * 1000.0;
                    publish = true;
                }
                if publish {
                    match grid.primitives() {
                        Ok(fields) => {
                            revision += 1;
                            *out.lock().unwrap() = Some(Snapshot {
                                grid: grid.clone(),
                                fields,
                                ms,
                                error: error.clone(),
                                notice: notice.clone(),
                                revision,
                            });
                        }
                        Err(e) => {
                            eprintln!("Fatal recovery error: {e}");
                            return;
                        }
                    }
                    publish = false;
                }
                let wait = if paused {
                    Duration::from_millis(8)
                } else {
                    next_step
                        .saturating_duration_since(Instant::now())
                        .min(Duration::from_millis(8))
                };
                thread::sleep(wait);
            }
        });
        Self {
            tx,
            latest,
            handle: Some(handle),
        }
    }
    pub fn send(&self, c: Command) {
        let _ = self.tx.send(c);
    }
    pub fn take(&self) -> Option<Snapshot> {
        self.latest.lock().unwrap().take()
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.tx.send(Command::Quit);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

pub fn export_vtk(g: &Grid) -> Result<String, Box<dyn std::error::Error>> {
    use std::io::Write;
    std::fs::create_dir_all("exports")?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis();
    let path = format!("exports/fluid-{stamp}-{}.vtk", g.steps);
    let mut f = std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?,
    );
    writeln!(
        f,
        "# vtk DataFile Version 3.0\nFluid Observatory model={:?} gamma={} t={}\nASCII\nDATASET STRUCTURED_POINTS",
        g.model, g.gamma, g.time
    )?;
    writeln!(f, "DIMENSIONS {} {} {}", g.n[0], g.n[1], g.n[2])?;
    let first = g.position([0; 3]);
    writeln!(f, "ORIGIN {} {} {}", first[0], first[1], first[2])?;
    writeln!(
        f,
        "SPACING {} {} {}",
        g.length[0] / g.n[0] as f64,
        g.length[1] / g.n[1] as f64,
        g.length[2] / g.n[2] as f64
    )?;
    writeln!(f, "POINT_DATA {}", g.u.len())?;
    let w = g.primitives()?;
    for (name, k) in [("proper_density", 0), ("pressure", 4)] {
        writeln!(f, "SCALARS {name} double 1\nLOOKUP_TABLE default")?;
        for wi in &w {
            writeln!(f, "{}", wi.fields()[k])?;
        }
    }
    writeln!(f, "VECTORS velocity double")?;
    for wi in &w {
        writeln!(f, "{} {} {}", wi.v[0], wi.v[1], wi.v[2])?;
    }
    if g.model == Model::Relativistic {
        writeln!(f, "SCALARS lorentz_factor double 1\nLOOKUP_TABLE default")?;
        for wi in &w {
            writeln!(f, "{}", wi.lorentz())?;
        }
    }
    for (name, k) in [("lab_mass_density", 0), ("total_energy_density", 4)] {
        writeln!(f, "SCALARS {name} double 1\nLOOKUP_TABLE default")?;
        for u in &g.u {
            writeln!(f, "{}", u[k])?;
        }
    }
    writeln!(f, "FIELD FieldData 4")?;
    for (name, a) in [
        ("initial_integrals", g.initial),
        ("exchanged_integrals", g.exchanged),
        ("current_integrals", g.totals()),
        ("relative_residuals", g.residual()),
    ] {
        write_state(&mut f, name, a)?;
    }
    f.flush()?;
    Ok(path)
}
fn write_state(f: &mut impl std::io::Write, name: &str, a: State) -> std::io::Result<()> {
    writeln!(
        f,
        "{name} 5 1 double\n{} {} {} {} {}",
        a[0], a[1], a[2], a[3], a[4]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wait_snapshot(worker: &Worker, predicate: impl Fn(&Snapshot) -> bool) -> Snapshot {
        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline {
            if let Some(snapshot) = worker.take()
                && predicate(&snapshot)
            {
                return snapshot;
            }
            thread::sleep(Duration::from_millis(2));
        }
        panic!("worker did not acknowledge command before deadline");
    }

    #[test]
    fn pause_step_heat_and_reset_are_consistent() {
        let config = Config {
            resolution: 8,
            ..Config::default()
        };
        let worker = Worker::new(config, true);
        let first = wait_snapshot(&worker, |_| true);
        assert_eq!(first.grid.steps, 0);
        thread::sleep(Duration::from_millis(35));
        assert!(worker.take().is_none(), "paused worker must not evolve");
        worker.send(Command::Step);
        let stepped = wait_snapshot(&worker, |s| s.grid.steps == 1);
        assert!(stepped.grid.time > 0.0);
        thread::sleep(Duration::from_millis(35));
        assert!(worker.take().is_none(), "single step must stay paused");
        worker.send(Command::Heat([0.0; 3]));
        let heated = wait_snapshot(&worker, |s| s.grid.exchanged[4] > stepped.grid.exchanged[4]);
        assert_eq!(heated.grid.steps, 1);
        assert!(heated.grid.residual()[4].abs() < 1e-12);
        worker.send(Command::Reset(Config {
            scene: Scene::Shear,
            model: Model::Newtonian,
            ..config
        }));
        let reset = wait_snapshot(&worker, |s| s.grid.model == Model::Newtonian);
        assert_eq!(reset.grid.steps, 0);
        assert_eq!(reset.grid.time, 0.0);
        assert_eq!(reset.grid.exchanged, [0.0; 5]);
        worker.send(Command::Pause(false));
        let resumed = wait_snapshot(&worker, |s| s.grid.steps >= 2);
        assert!(resumed.error.is_none());
    }

    #[test]
    fn slow_motion_waits_for_the_requested_simulation_clock() {
        let worker = Worker::new(
            Config {
                resolution: 8,
                ..Config::default()
            },
            true,
        );
        wait_snapshot(&worker, |_| true);
        worker.send(Command::Rate(0.05));
        worker.send(Command::Pause(false));
        let first = wait_snapshot(&worker, |s| s.grid.steps == 1);
        assert!(first.grid.last_dt / 0.05 > 0.25);
        thread::sleep(Duration::from_millis(160));
        assert!(
            worker.take().is_none(),
            "a short command-poll interval must not accelerate slow motion"
        );
    }
}
