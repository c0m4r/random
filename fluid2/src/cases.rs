use crate::physics::{Boundary, Grid, Model, Primitive};
use std::f64::consts::TAU;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scene {
    Jet,
    Blast,
    Shear,
    Shock,
}

impl Scene {
    pub const ALL: [Self; 4] = [Self::Jet, Self::Blast, Self::Shear, Self::Shock];
    pub fn id(self) -> &'static str {
        match self {
            Self::Jet => "jet",
            Self::Blast => "blast",
            Self::Shear => "shear",
            Self::Shock => "shock",
        }
    }
    pub fn title(self) -> &'static str {
        match self {
            Self::Jet => "Beyond the sound barrier",
            Self::Blast => "Anatomy of an explosion",
            Self::Shear => "The birth of a vortex",
            Self::Shock => "When pressure breaks free",
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::Jet => "01   Relativistic jet",
            Self::Blast => "02   Spherical blast",
            Self::Shear => "03   Shear instability",
            Self::Shock => "04   Shock tube",
        }
    }
    pub fn description(self) -> &'static str {
        match self {
            Self::Jet => {
                "A narrow beam meets a quiet gas. A bow shock compresses the ambient medium while a hot cocoon expands around the beam."
            }
            Self::Blast => {
                "A small pocket of high pressure launches a spherical shock. Energy flows outward; the center cools as it expands."
            }
            Self::Shear => {
                "Two layers slide past each other. A small transverse perturbation grows into Kelvin–Helmholtz rolls and mixes their boundary."
            }
            Self::Shock => {
                "Remove a diaphragm between two gases. A shock, a contact discontinuity and a rarefaction fan emerge from one pressure jump."
            }
        }
    }
    pub fn default_model(self) -> Model {
        match self {
            Self::Jet | Self::Blast => Model::Relativistic,
            _ => Model::Newtonian,
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|x| x.id() == s)
    }
}

pub fn build(scene: Scene, model: Model, res: usize, jet_speed: f64) -> Grid {
    let n = [res * 3 / 2, res, res];
    let length = [3.0, 2.0, 2.0];
    let gamma = if scene == Scene::Shock {
        1.4
    } else {
        4.0 / 3.0
    };
    let boundary = match scene {
        Scene::Shear => Boundary::Periodic,
        Scene::Jet => Boundary::Jet,
        _ => Boundary::Outflow,
    };
    let mut g = Grid::new(n, length, model, gamma, boundary, |[x, y, z]| {
        match scene {
            Scene::Jet => {
                // A short initial beam makes the nozzle visible before evolution.
                if x < -1.1 && y * y + z * z < 0.22_f64.powi(2) {
                    Primitive::new(0.15, [jet_speed, 0.0, 0.0], 0.02)
                } else {
                    Primitive::new(1.0, [0.0; 3], 0.02)
                }
            }
            Scene::Blast => {
                let r = (x * x + y * y + z * z).sqrt();
                let p = 0.015 + 2.0 * 0.5 * (1.0 - ((r - 0.26) / 0.045).tanh());
                Primitive::new(1.0, [0.0; 3], p)
            }
            Scene::Shear => {
                let layer = 0.5 * (1.0 + ((0.45 - y.abs()) / 0.09).tanh());
                let vy = 0.035
                    * (TAU * x / 1.5).sin()
                    * (1.0 + 0.15 * (TAU * z / 2.0).cos())
                    * (-((y.abs() - 0.45) / 0.18).powi(2)).exp();
                Primitive::new(1.0 + layer, [0.7 * (layer - 0.5), vy, 0.0], 0.3)
            }
            Scene::Shock => {
                if x < 0.0 {
                    Primitive::new(1.0, [0.0; 3], 1.0)
                } else {
                    Primitive::new(0.125, [0.0; 3], 0.1)
                }
            }
        }
    });
    g.jet_speed = jet_speed;
    g
}
