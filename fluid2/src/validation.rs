//! Independent analytical checks and full three-dimensional scenario checks.
use crate::{
    cases::{self, Scene},
    physics::*,
};
use std::f64::consts::TAU;
use std::time::Instant;

pub struct Check {
    pub name: String,
    pub value: f64,
    pub limit: f64,
}
impl Check {
    pub fn passed(&self) -> bool {
        self.value.is_finite() && self.value <= self.limit
    }
}

pub fn advection_error(model: Model, n: usize) -> Result<f64, String> {
    let v = 0.3;
    let mut g = Grid::new(
        [n, 1, 1],
        [1.0; 3],
        model,
        4.0 / 3.0,
        Boundary::Periodic,
        |[x, _, _]| Primitive::new(1.0 + 0.2 * (TAU * x).sin(), [v, 0.0, 0.0], 0.2),
    );
    g.advance_to(0.4)?;
    let w = g.primitives()?;
    Ok(w.iter()
        .enumerate()
        .map(|(i, w)| {
            let x = g.position(g.coords(i))[0];
            (w.rho - (1.0 + 0.2 * (TAU * (x - v * g.time)).sin())).abs()
        })
        .sum::<f64>()
        / n as f64)
}

fn pressure_function(p: f64, rho: f64, pk: f64, gamma: f64) -> f64 {
    let a = (gamma * pk / rho).sqrt();
    if p > pk {
        (p - pk) * (2.0 / ((gamma + 1.0) * rho) / (p + (gamma - 1.0) / (gamma + 1.0) * pk)).sqrt()
    } else {
        2.0 * a / (gamma - 1.0) * ((p / pk).powf((gamma - 1.0) / (2.0 * gamma)) - 1.0)
    }
}

/// Exact Sod solution (gamma 1.4, diaphragm x=0), independently sampled.
pub fn sod_density(x: f64, t: f64) -> f64 {
    let gamma = 1.4;
    let mut lo = 0.1;
    let mut hi = 1.0;
    for _ in 0..80 {
        let p = 0.5 * (lo + hi);
        if pressure_function(p, 1.0, 1.0, gamma) + pressure_function(p, 0.125, 0.1, gamma) > 0.0 {
            hi = p
        } else {
            lo = p
        }
    }
    let ps = 0.5 * (lo + hi);
    let vs =
        0.5 * (pressure_function(ps, 0.125, 0.1, gamma) - pressure_function(ps, 1.0, 1.0, gamma));
    let xi = x / t;
    let al = gamma.sqrt();
    let ar = (gamma * 0.1 / 0.125).sqrt();
    let astar = al * ps.powf((gamma - 1.0) / (2.0 * gamma));
    if xi < -al {
        1.0
    } else if xi < vs - astar {
        (2.0 / (gamma + 1.0) - (gamma - 1.0) * xi / ((gamma + 1.0) * al)).powf(2.0 / (gamma - 1.0))
    } else if xi < vs {
        ps.powf(1.0 / gamma)
    } else {
        let shock =
            ar * ((gamma + 1.0) / (2.0 * gamma) * ps / 0.1 + (gamma - 1.0) / (2.0 * gamma)).sqrt();
        if xi < shock {
            let r = ps / 0.1;
            0.125 * (r + (gamma - 1.0) / (gamma + 1.0)) / ((gamma - 1.0) / (gamma + 1.0) * r + 1.0)
        } else {
            0.125
        }
    }
}

pub fn sod_error(n: usize) -> Result<f64, String> {
    let mut g = Grid::new(
        [n, 1, 1],
        [1.0; 3],
        Model::Newtonian,
        1.4,
        Boundary::Outflow,
        |[x, _, _]| {
            if x < 0.0 {
                Primitive::new(1.0, [0.0; 3], 1.0)
            } else {
                Primitive::new(0.125, [0.0; 3], 0.1)
            }
        },
    );
    g.advance_to(0.2)?;
    Ok(g.primitives()?
        .iter()
        .enumerate()
        .map(|(i, w)| (w.rho - sod_density(g.position(g.coords(i))[0], 0.2)).abs())
        .sum::<f64>()
        / n as f64)
}

pub fn run() -> Result<Vec<Check>, String> {
    let mut checks = Vec::new();
    let mut add = |name: String, value: f64, limit: f64| checks.push(Check { name, value, limit });
    let mut inversion = 0.0_f64;
    for rho in [0.01, 1.0, 10.0] {
        for p in [0.0001, 0.1, 10.0, 100.0] {
            for speed in [0.0, 0.1, 0.9, 0.99, 0.999] {
                let w = Primitive::new(rho, [speed * 0.8, speed * 0.6, 0.0], p);
                let back = primitive(
                    conserved(w, Model::Relativistic, 4.0 / 3.0),
                    Model::Relativistic,
                    4.0 / 3.0,
                )?;
                for (a, b) in w.fields().iter().zip(back.fields()) {
                    inversion = inversion.max((a - b).abs() / a.abs().max(1e-4));
                }
            }
        }
    }
    add(
        "SR primitive recovery, Lorentz factors up to 22.4".into(),
        inversion,
        2e-6,
    );
    for model in [Model::Newtonian, Model::Relativistic] {
        let coarse = advection_error(model, 48)?;
        let fine = advection_error(model, 96)?;
        add(
            format!("{} smooth advection L1 / amplitude", model.name()),
            fine / 0.2,
            0.006,
        );
        add(
            format!("{} convergence error ratio (96 / 48)", model.name()),
            fine / coarse,
            0.48,
        );
        let w = Primitive::new(1.2, [0.2, -0.1, 0.05], 0.4);
        let mut g = Grid::new(
            [8, 7, 6],
            [1.0; 3],
            model,
            4.0 / 3.0,
            Boundary::Periodic,
            |_| w,
        );
        let before = g.u.clone();
        for _ in 0..4 {
            g.step(1.0)?;
        }
        let err =
            g.u.iter()
                .zip(before)
                .flat_map(|(a, b)| {
                    a.iter()
                        .zip(b)
                        .map(|(x, y)| (x - y).abs())
                        .collect::<Vec<_>>()
                })
                .fold(0.0, f64::max);
        add(
            format!("{} 3D uniform-state preservation", model.name()),
            err,
            1e-12,
        );
    }
    add(
        "Sod shock tube: density L1 at t=0.2, 240 cells".into(),
        sod_error(240)?,
        0.012,
    );
    let slow = Primitive::new(1.0, [0.001, -0.0004, 0.0002], 1e-6);
    let classical = conserved(slow, Model::Newtonian, 5.0 / 3.0);
    let rel = conserved(slow, Model::Relativistic, 5.0 / 3.0);
    add(
        "Newtonian limit of SR momentum (relative)".into(),
        (rel[1] / classical[1] - 1.0).abs(),
        5e-6,
    );
    add(
        "Newtonian limit of SR energy excluding rest mass".into(),
        ((rel[4] - rel[0]) / classical[4] - 1.0).abs(),
        5e-6,
    );
    // In an isentropic rest-state acoustic wave delta rho / rho = delta p / (gamma p).
    let amp = 1e-5;
    let gamma: f64 = 4.0 / 3.0;
    let rho = 1.0;
    let p = 0.2;
    let h = rho + gamma / (gamma - 1.0) * p;
    let cs = (gamma * p / h).sqrt();
    let n = 128;
    let mut sound = Grid::new(
        [n, 1, 1],
        [1.0; 3],
        Model::Relativistic,
        gamma,
        Boundary::Periodic,
        |[x, _, _]| {
            let dp = amp * (TAU * x).sin();
            Primitive::new(
                rho + rho * dp / (gamma * p),
                [dp / (h * cs), 0.0, 0.0],
                p + dp,
            )
        },
    );
    sound.advance_to(0.3)?;
    let err = sound
        .primitives()?
        .iter()
        .enumerate()
        .map(|(i, w)| {
            let x = sound.position(sound.coords(i))[0];
            (w.p - p - amp * (TAU * (x - cs * sound.time)).sin()).abs()
        })
        .sum::<f64>()
        / (n as f64 * amp);
    add(
        "SR acoustic wave: analytical sound speed, normalized L1".into(),
        err,
        0.008,
    );
    // Independent stationary relativistic normal shock, determined by the
    // Rankine–Hugoniot invariants j=rho W v, Q=rho h W² v, P=Q v+p.
    let upstream = Primitive::new(1.0, [0.9, 0.0, 0.0], 0.01);
    let gamma = 4.0 / 3.0;
    let a = gamma / (gamma - 1.0);
    let w = upstream.lorentz();
    let j = upstream.rho * w * upstream.v[0];
    let q = (upstream.rho + a * upstream.p) * w * w * upstream.v[0];
    let momentum = q * upstream.v[0] + upstream.p;
    let mut lo = 0.01;
    let mut hi = 0.5;
    let downstream = |v: f64| {
        Primitive::new(
            j * (1.0 - v * v).sqrt() / v,
            [v, 0.0, 0.0],
            momentum - q * v,
        )
    };
    for _ in 0..80 {
        let v = 0.5 * (lo + hi);
        let d = downstream(v);
        let f = (d.rho + a * d.p) * v / (1.0 - v * v) - q;
        if f > 0.0 { hi = v } else { lo = v }
    }
    let down = downstream(0.5 * (lo + hi));
    let mut shock = Grid::new(
        [240, 1, 1],
        [1.0; 3],
        Model::Relativistic,
        gamma,
        Boundary::Outflow,
        |[x, _, _]| if x < 0.0 { upstream } else { down },
    );
    shock.advance_to(0.25)?;
    let l1 = shock
        .primitives()?
        .iter()
        .enumerate()
        .map(|(i, w)| (w.rho - if i < 120 { upstream.rho } else { down.rho }).abs())
        .sum::<f64>()
        / (240.0 * (down.rho - upstream.rho));
    add(
        "SR stationary strong shock: normalized density L1".into(),
        l1,
        0.02,
    );
    add(
        "SR stationary strong shock: total mass change".into(),
        ((shock.totals()[0] - shock.initial[0]) / shock.initial[0]).abs(),
        1e-10,
    );
    for scene in Scene::ALL {
        for model in [Model::Newtonian, Model::Relativistic] {
            let mut g = cases::build(scene, model, 12, 0.92);
            for _ in 0..24 {
                g.step(1.0)?;
            }
            if scene == Scene::Blast {
                g.deposit_heat([0.0; 3], 0.7);
                g.step(1.0)?;
            }
            let residual = g.residual().iter().map(|x| x.abs()).fold(0.0, f64::max);
            add(
                format!(
                    "{} / {} 3D conservation incl. boundary & sources",
                    scene.id(),
                    model.name()
                ),
                residual,
                2e-11,
            );
            let w = g.primitives()?;
            if model == Model::Relativistic {
                add(
                    format!("{} maximum speed / c", scene.id()),
                    w.iter().map(|w| w.speed2().sqrt()).fold(0.0, f64::max),
                    1.0 - 1e-12,
                );
            }
            add(
                format!("{} / {} rejected steps", scene.id(), model.name()),
                g.rejected as f64,
                0.0,
            );
        }
    }
    Ok(checks)
}

pub fn report() -> Result<bool, String> {
    let start = Instant::now();
    let checks = run()?;
    for check in &checks {
        println!(
            "{}  {:<68} {:>11.4e}  <= {:.4e}",
            if check.passed() { "PASS" } else { "FAIL" },
            check.name,
            check.value,
            check.limit
        );
    }
    let passed = checks.iter().filter(|c| c.passed()).count();
    println!(
        "\n{passed}/{} passed in {:.2}s. Finite resolution; ideal gas; no physical viscosity or gravity.",
        checks.len(),
        start.elapsed().as_secs_f64()
    );
    Ok(passed == checks.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn analytical_and_conservation_suite() {
        for c in run().unwrap() {
            assert!(c.passed(), "{}: {} > {}", c.name, c.value, c.limit);
        }
    }
    #[test]
    fn inadmissible_states_are_rejected() {
        assert!(primitive([1.0, 2.0, 0.0, 0.0, 1.0], Model::Relativistic, 4.0 / 3.0).is_err());
        assert!(primitive([1.0, 0.0, 0.0, 0.0, -1.0], Model::Newtonian, 1.4).is_err());
    }
    #[test]
    fn newtonian_stationary_contact_and_shear_have_no_numerical_mass_flux() {
        let l = Primitive::new(1.0, [0.3, 0.0, -0.1], 0.3);
        let r = Primitive::new(2.0, [-0.3, 0.0, 0.2], 0.3);
        let f = hllc(l, r, 4.0 / 3.0, 1);
        for (actual, exact) in f.iter().zip([0.0, 0.0, 0.3, 0.0, 0.0]) {
            assert!((actual - exact).abs() < 1e-14);
        }
    }
    #[test]
    fn relativistic_characteristics_include_tangential_velocity() {
        for speed in [0.0, 0.5, 0.95, 0.999] {
            let w = Primitive::new(1.0, [speed * 0.6, speed * 0.8, 0.0], 1.0);
            for axis in 0..3 {
                let (l, r) = waves(w, Model::Relativistic, 4.0 / 3.0, axis);
                assert!(l >= -1.0 && r <= 1.0 && l <= r);
            }
        }
        let w = Primitive::new(1.0, [0.7, 0.0, 0.0], 0.2);
        let cs = (4.0 / 3.0 * 0.2 / 1.8_f64).sqrt();
        let (l, r) = waves(w, Model::Relativistic, 4.0 / 3.0, 0);
        assert!((l - (0.7 - cs) / (1.0 - 0.7 * cs)).abs() < 1e-12);
        assert!((r - (0.7 + cs) / (1.0 + 0.7 * cs)).abs() < 1e-12);
    }
}
