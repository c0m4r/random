//! Conservative, unsplit 3D Euler / special-relativistic Euler hydrodynamics.
//! Units: c = 1 in relativity; dimensionless reference units in Newtonian mode.
//! Five conserved fields, HLLC (Newtonian) / HLLE (SR) fluxes, minmod MUSCL, SSP-RK2.

pub type State = [f64; 5];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Model {
    Newtonian,
    Relativistic,
}

impl Model {
    pub fn name(self) -> &'static str {
        match self {
            Self::Newtonian => "Newtonian",
            Self::Relativistic => "Special relativity",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Primitive {
    pub rho: f64,
    pub v: [f64; 3],
    pub p: f64,
}

impl Primitive {
    pub fn new(rho: f64, v: [f64; 3], p: f64) -> Self {
        Self { rho, v, p }
    }
    pub fn speed2(self) -> f64 {
        self.v.iter().map(|x| x * x).sum()
    }
    pub fn lorentz(self) -> f64 {
        1.0 / (1.0 - self.speed2()).sqrt()
    }
    pub fn fields(self) -> State {
        [self.rho, self.v[0], self.v[1], self.v[2], self.p]
    }
    pub fn from_fields(a: State) -> Self {
        Self::new(a[0], [a[1], a[2], a[3]], a[4])
    }
    pub fn valid(self, model: Model) -> bool {
        self.fields().iter().all(|x| x.is_finite())
            && self.rho > 0.0
            && self.p > 0.0
            && (model == Model::Newtonian || self.speed2() < 1.0)
    }
}

pub fn conserved(w: Primitive, model: Model, gamma: f64) -> State {
    match model {
        Model::Newtonian => [
            w.rho,
            w.rho * w.v[0],
            w.rho * w.v[1],
            w.rho * w.v[2],
            w.p / (gamma - 1.0) + 0.5 * w.rho * w.speed2(),
        ],
        Model::Relativistic => {
            let lor = w.lorentz();
            let q = (w.rho + gamma / (gamma - 1.0) * w.p) * lor * lor;
            [w.rho * lor, q * w.v[0], q * w.v[1], q * w.v[2], q - w.p]
        }
    }
}

pub fn primitive(u: State, model: Model, gamma: f64) -> Result<Primitive, &'static str> {
    if !u.iter().all(|v| v.is_finite()) || u[0] <= 0.0 {
        return Err("nonpositive or nonfinite conserved density");
    }
    let s2 = u[1] * u[1] + u[2] * u[2] + u[3] * u[3];
    let w = match model {
        Model::Newtonian => Primitive::new(
            u[0],
            [u[1] / u[0], u[2] / u[0], u[3] / u[0]],
            (gamma - 1.0) * (u[4] - 0.5 * s2 / u[0]),
        ),
        Model::Relativistic => {
            // Admissibility for a gamma-law gas with 1 < gamma <= 2.
            let rest = (u[0] * u[0] + s2).sqrt();
            if u[4] <= rest {
                return Err("inadmissible relativistic energy");
            }
            let a = gamma / (gamma - 1.0);
            let mut lo = 0.0;
            let mut hi = (gamma - 1.0) * u[4];
            let mut p = (gamma - 1.0) * (u[4] - rest);
            for _ in 0..80 {
                let q = u[4] + p;
                let lor = 1.0 / (1.0 - s2 / (q * q)).sqrt();
                let f = u[0] * lor + a * p * lor * lor - p - u[4];
                if f.abs() < 2.0e-13 * u[4] {
                    break;
                }
                if f > 0.0 {
                    hi = p;
                } else {
                    lo = p;
                }
                let dl = -s2 * lor.powi(3) / q.powi(3);
                let df = u[0] * dl + a * lor * lor + 2.0 * a * p * lor * dl - 1.0;
                let next = p - f / df;
                p = if next.is_finite() && next > lo && next < hi {
                    next
                } else {
                    0.5 * (lo + hi)
                };
            }
            let q = u[4] + p;
            let v = [u[1] / q, u[2] / q, u[3] / q];
            Primitive::new(
                u[0] * (1.0 - v.iter().map(|x| x * x).sum::<f64>()).sqrt(),
                v,
                p,
            )
        }
    };
    if !w.valid(model) {
        return Err("nonpositive pressure or acausal velocity");
    }
    // Check inversion rather than silently accepting a failed root solve.
    if model == Model::Relativistic {
        let back = conserved(w, model, gamma);
        if (back[4] - u[4]).abs() > 1.0e-10 * u[4] {
            return Err("primitive recovery did not converge");
        }
    }
    Ok(w)
}

pub fn waves(w: Primitive, model: Model, gamma: f64, axis: usize) -> (f64, f64) {
    let vn = w.v[axis];
    match model {
        Model::Newtonian => {
            let c = (gamma * w.p / w.rho).sqrt();
            (vn - c, vn + c)
        }
        Model::Relativistic => {
            let cs2 = gamma * w.p / (w.rho + gamma / (gamma - 1.0) * w.p);
            let v2 = w.speed2();
            let den = 1.0 - v2 * cs2;
            let rad = (cs2 * (1.0 - v2) * (1.0 - v2 * cs2 - vn * vn * (1.0 - cs2)))
                .max(0.0)
                .sqrt();
            (
                (vn * (1.0 - cs2) - rad) / den,
                (vn * (1.0 - cs2) + rad) / den,
            )
        }
    }
}

pub fn flux(w: Primitive, model: Model, gamma: f64, axis: usize) -> State {
    let u = conserved(w, model, gamma);
    let mut f = u.map(|x| x * w.v[axis]);
    f[axis + 1] += w.p;
    f[4] = match model {
        Model::Newtonian => (u[4] + w.p) * w.v[axis],
        Model::Relativistic => u[axis + 1],
    };
    f
}

pub fn hlle(l: Primitive, r: Primitive, model: Model, gamma: f64, axis: usize) -> State {
    let (lm, lp) = waves(l, model, gamma, axis);
    let (rm, rp) = waves(r, model, gamma, axis);
    let minus = lm.min(rm).min(0.0);
    let plus = lp.max(rp).max(0.0);
    let fl = flux(l, model, gamma, axis);
    let fr = flux(r, model, gamma, axis);
    let ul = conserved(l, model, gamma);
    let ur = conserved(r, model, gamma);
    std::array::from_fn(|k| {
        (plus * fl[k] - minus * fr[k] + minus * plus * (ur[k] - ul[k])) / (plus - minus)
    })
}

/// The Newtonian contact-resolving HLLC flux. Resolves a stationary shear
/// interface exactly; HLLE otherwise diffuses tangential momentum across it.
/// Nonphysical star states use the more dissipative two-wave flux locally.
pub fn hllc(l: Primitive, r: Primitive, gamma: f64, axis: usize) -> State {
    let model = Model::Newtonian;
    let (lm, lp) = waves(l, model, gamma, axis);
    let (rm, rp) = waves(r, model, gamma, axis);
    let sl = lm.min(rm);
    let sr = lp.max(rp);
    let fl = flux(l, model, gamma, axis);
    let fr = flux(r, model, gamma, axis);
    if sl >= 0.0 {
        return fl;
    }
    if sr <= 0.0 {
        return fr;
    }
    let dl = l.rho * (sl - l.v[axis]);
    let dr = r.rho * (sr - r.v[axis]);
    let sm = (r.p - l.p + dl * l.v[axis] - dr * r.v[axis]) / (dl - dr);
    let ps = l.p + dl * (sm - l.v[axis]);
    if !sm.is_finite() || ps <= 0.0 || sm <= sl || sm >= sr {
        return hlle(l, r, model, gamma, axis);
    }
    let star = |w: Primitive, s: f64| {
        let u = conserved(w, model, gamma);
        let rho = w.rho * (s - w.v[axis]) / (s - sm);
        let mut us = [
            rho,
            rho * w.v[0],
            rho * w.v[1],
            rho * w.v[2],
            ((s - w.v[axis]) * u[4] - w.p * w.v[axis] + ps * sm) / (s - sm),
        ];
        us[axis + 1] = rho * sm;
        (u, us)
    };
    let (ul, usl) = star(l, sl);
    let (ur, usr) = star(r, sr);
    if primitive(usl, model, gamma).is_err() || primitive(usr, model, gamma).is_err() {
        return hlle(l, r, model, gamma, axis);
    }
    if sm >= 0.0 {
        std::array::from_fn(|k| fl[k] + sl * (usl[k] - ul[k]))
    } else {
        std::array::from_fn(|k| fr[k] + sr * (usr[k] - ur[k]))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Boundary {
    Periodic,
    Outflow,
    Jet,
}

#[derive(Clone)]
pub struct Grid {
    pub n: [usize; 3],
    pub length: [f64; 3],
    pub model: Model,
    pub gamma: f64,
    pub boundary: Boundary,
    pub jet_speed: f64,
    pub u: Vec<State>,
    pub time: f64,
    pub steps: u64,
    pub rejected: u64,
    pub first_order_steps: u64,
    pub initial: State,
    pub exchanged: State,
    pub last_dt: f64,
}

fn minmod(a: f64, b: f64) -> f64 {
    if a * b <= 0.0 {
        0.0
    } else {
        a.signum() * a.abs().min(b.abs())
    }
}

impl Grid {
    pub fn new(
        n: [usize; 3],
        length: [f64; 3],
        model: Model,
        gamma: f64,
        boundary: Boundary,
        init: impl Fn([f64; 3]) -> Primitive,
    ) -> Self {
        assert!(n.iter().all(|x| *x > 0) && length.iter().all(|x| x.is_finite() && *x > 0.0));
        assert!(gamma > 1.0 && gamma <= 2.0);
        let mut grid = Self {
            n,
            length,
            model,
            gamma,
            boundary,
            jet_speed: 0.92,
            u: Vec::new(),
            time: 0.0,
            steps: 0,
            rejected: 0,
            first_order_steps: 0,
            initial: [0.0; 5],
            exchanged: [0.0; 5],
            last_dt: 0.0,
        };
        for z in 0..n[2] {
            for y in 0..n[1] {
                for x in 0..n[0] {
                    let w = init(grid.position([x, y, z]));
                    assert!(w.valid(model));
                    grid.u.push(conserved(w, model, gamma));
                }
            }
        }
        grid.initial = grid.totals();
        grid
    }
    pub fn index(&self, c: [usize; 3]) -> usize {
        (c[2] * self.n[1] + c[1]) * self.n[0] + c[0]
    }
    pub fn coords(&self, i: usize) -> [usize; 3] {
        [
            i % self.n[0],
            (i / self.n[0]) % self.n[1],
            i / (self.n[0] * self.n[1]),
        ]
    }
    pub fn position(&self, c: [usize; 3]) -> [f64; 3] {
        std::array::from_fn(|a| ((c[a] as f64 + 0.5) / self.n[a] as f64 - 0.5) * self.length[a])
    }
    pub fn cell_volume(&self) -> f64 {
        (0..3).map(|a| self.length[a] / self.n[a] as f64).product()
    }
    pub fn totals(&self) -> State {
        let mut total = [0.0; 5];
        let dv = self.cell_volume();
        for u in &self.u {
            for k in 0..5 {
                total[k] += u[k] * dv;
            }
        }
        total
    }
    pub fn residual(&self) -> State {
        let total = self.totals();
        std::array::from_fn(|k| {
            (total[k] - self.initial[k] - self.exchanged[k]) / self.initial[k].abs().max(1.0)
        })
    }
    pub fn primitives(&self) -> Result<Vec<Primitive>, &'static str> {
        self.recover(&self.u)
    }
    fn recover(&self, u: &[State]) -> Result<Vec<Primitive>, &'static str> {
        u.iter()
            .map(|u| primitive(*u, self.model, self.gamma))
            .collect()
    }
    fn sample(&self, w: &[Primitive], mut c: [isize; 3]) -> Primitive {
        if self.boundary == Boundary::Jet && c[0] < 0 {
            let y = ((c[1] as f64 + 0.5) / self.n[1] as f64 - 0.5) * self.length[1];
            let z = ((c[2] as f64 + 0.5) / self.n[2] as f64 - 0.5) * self.length[2];
            if y * y + z * z < 0.22_f64.powi(2) {
                return Primitive::new(0.15, [self.jet_speed, 0.0, 0.0], 0.02);
            }
        }
        for (a, q) in c.iter_mut().enumerate() {
            *q = if self.boundary == Boundary::Periodic {
                q.rem_euclid(self.n[a] as isize)
            } else {
                (*q).clamp(0, self.n[a] as isize - 1)
            };
        }
        w[self.index(c.map(|x| x as usize))]
    }
    fn rhs(&self, w: &[Primitive], order2: bool) -> (Vec<State>, State) {
        let mut slopes = vec![[[0.0; 5]; 3]; w.len()];
        if order2 {
            for (i, wi) in w.iter().enumerate() {
                let c = self.coords(i).map(|x| x as isize);
                let mid = wi.fields();
                for a in 0..3 {
                    if self.n[a] == 1 {
                        continue;
                    }
                    let mut l = c;
                    l[a] -= 1;
                    let mut r = c;
                    r[a] += 1;
                    let wl = self.sample(w, l).fields();
                    let wr = self.sample(w, r).fields();
                    slopes[i][a] = std::array::from_fn(|k| minmod(mid[k] - wl[k], wr[k] - mid[k]));
                    for sign in [-0.5, 0.5] {
                        if !Primitive::from_fields(std::array::from_fn(|k| {
                            mid[k] + sign * slopes[i][a][k]
                        }))
                        .valid(self.model)
                        {
                            slopes[i][a] = [0.0; 5];
                        }
                    }
                }
            }
        }
        let mut rhs = vec![[0.0; 5]; w.len()];
        let mut boundary = [0.0; 5];
        let dv = self.cell_volume();
        for a in 0..3 {
            if self.n[a] == 1 {
                continue;
            }
            let inv_dx = self.n[a] as f64 / self.length[a];
            let mut face_n = self.n;
            face_n[a] += 1;
            for z in 0..face_n[2] {
                for y in 0..face_n[1] {
                    for x in 0..face_n[0] {
                        let r = [x as isize, y as isize, z as isize];
                        let mut l = r;
                        l[a] -= 1;
                        let inside_l = l[a] >= 0;
                        let inside_r = r[a] < self.n[a] as isize;
                        let mut wl = self.sample(w, l);
                        let mut wr = self.sample(w, r);
                        // Periodic ghost reconstruction matches the opposite boundary face exactly.
                        let il = if inside_l {
                            Some(self.index(l.map(|x| x as usize)))
                        } else if self.boundary == Boundary::Periodic {
                            let mut c = l;
                            c[a] = self.n[a] as isize - 1;
                            Some(self.index(c.map(|x| x as usize)))
                        } else {
                            None
                        };
                        let ir = if inside_r {
                            Some(self.index(r.map(|x| x as usize)))
                        } else if self.boundary == Boundary::Periodic {
                            let mut c = r;
                            c[a] = 0;
                            Some(self.index(c.map(|x| x as usize)))
                        } else {
                            None
                        };
                        if let Some(i) = il {
                            let f = wl.fields();
                            wl = Primitive::from_fields(std::array::from_fn(|k| {
                                f[k] + 0.5 * slopes[i][a][k]
                            }));
                        }
                        if let Some(i) = ir {
                            let f = wr.fields();
                            wr = Primitive::from_fields(std::array::from_fn(|k| {
                                f[k] - 0.5 * slopes[i][a][k]
                            }));
                        }
                        let f = match self.model {
                            Model::Newtonian => hllc(wl, wr, self.gamma, a),
                            Model::Relativistic => hlle(wl, wr, self.model, self.gamma, a),
                        };
                        for k in 0..5 {
                            if inside_l {
                                rhs[il.unwrap()][k] -= f[k] * inv_dx;
                            } else {
                                boundary[k] += f[k] * inv_dx * dv;
                            }
                            if inside_r {
                                rhs[ir.unwrap()][k] += f[k] * inv_dx;
                            } else {
                                boundary[k] -= f[k] * inv_dx * dv;
                            }
                        }
                    }
                }
            }
        }
        (rhs, boundary)
    }
    pub fn stable_dt(&self, w: &[Primitive]) -> f64 {
        let mut max_speed = [0.0_f64; 3];
        for wi in w {
            for (a, s) in max_speed.iter_mut().enumerate() {
                if self.n[a] > 1 {
                    let (l, r) = waves(*wi, self.model, self.gamma, a);
                    *s = s.max(l.abs().max(r.abs()));
                }
            }
        }
        if self.boundary == Boundary::Jet {
            let inlet = Primitive::new(0.15, [self.jet_speed, 0.0, 0.0], 0.02);
            for (a, s) in max_speed.iter_mut().enumerate() {
                if self.n[a] > 1 {
                    let (l, r) = waves(inlet, self.model, self.gamma, a);
                    *s = s.max(l.abs().max(r.abs()));
                }
            }
        }
        let rate: f64 = (0..3)
            .map(|a| max_speed[a] * self.n[a] as f64 / self.length[a])
            .sum();
        if rate == 0.0 { 0.01 } else { 0.35 / rate }
    }
    pub fn step(&mut self, max_dt: f64) -> Result<f64, String> {
        if !max_dt.is_finite() || max_dt <= 0.0 {
            return Err("time step must be finite and positive".into());
        }
        let w = self.primitives()?;
        let mut dt = self.stable_dt(&w).min(max_dt);
        for attempt in 0..12 {
            let order2 = attempt < 4;
            let (r0, b0) = self.rhs(&w, order2);
            let stage: Vec<State> = self
                .u
                .iter()
                .zip(&r0)
                .map(|(u, r)| std::array::from_fn(|k| u[k] + dt * r[k]))
                .collect();
            if let Ok(w1) = self.recover(&stage) {
                let (r1, b1) = self.rhs(&w1, order2);
                let next: Vec<State> = self
                    .u
                    .iter()
                    .zip(&stage)
                    .zip(&r1)
                    .map(|((u, s), r)| std::array::from_fn(|k| 0.5 * (u[k] + s[k] + dt * r[k])))
                    .collect();
                if self.recover(&next).is_ok() {
                    self.u = next;
                    self.time += dt;
                    self.steps += 1;
                    self.last_dt = dt;
                    for k in 0..5 {
                        self.exchanged[k] += 0.5 * dt * (b0[k] + b1[k]);
                    }
                    if !order2 {
                        self.first_order_steps += 1;
                    }
                    return Ok(dt);
                }
            }
            self.rejected += 1;
            dt *= 0.5;
        }
        Err("No admissible update after 12 reduced steps; simulation paused without altering its last valid state".into())
    }
    pub fn advance_to(&mut self, end: f64) -> Result<(), String> {
        while self.time < end - 1e-12 {
            self.step(end - self.time)?;
        }
        Ok(())
    }
    pub fn deposit_heat(&mut self, center: [f64; 3], amount: f64) {
        // A real, explicitly budgeted external source, not a velocity/pressure clamp.
        let dv = self.cell_volume();
        for i in 0..self.u.len() {
            let pos = self.position(self.coords(i));
            let r2: f64 = (0..3).map(|a| (pos[a] - center[a]).powi(2)).sum();
            let de = amount * (-r2 / 0.035).exp();
            self.u[i][4] += de;
            self.exchanged[4] += de * dv;
        }
    }
}
