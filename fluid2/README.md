# Fluid Observatory

A native Rust laboratory for classical and special-relativistic fluid motion. Explore a relativistic jet, a spherical explosion, Kelvin–Helmholtz shear, and a shock tube in a live, orbitable 3D volume.

The simulation evolves **three-dimensional compressible ideal-gas hydrodynamics**. The relativistic mode conserves rest mass and energy–momentum in flat spacetime. It is an educational numerical simulation with explicit assumptions and validation, not a general-relativistic or liquid-water solver.

## Run

Requires Rust 1.95 or newer and an OpenGL 3.3 desktop driver. Linux uses Wayland or X11. The first build compiles the GUI dependencies; subsequent launches are fast.

```sh
cargo run --release --locked
```

The app opens maximized. Restore or resize the window normally. The control panel scrolls on smaller displays. Physics runs on a separate worker thread; rendering remains interactive while the solver advances. Balanced mode uses a 42 × 28 × 28 grid and a 144-sample HDR volume renderer with gradient lighting, bloom, and a metric reference grid. Cinematic rendering increases ray samples and pixel resolution; it does not increase physics resolution.

```sh
cargo run --release --locked -- --scene shear
cargo run --release --locked -- --scene jet --model newtonian
cargo run --release --locked -- --scene blast --resolution 40
```

| Control | Action |
| --- | --- |
| Drag / scroll | Orbit / zoom |
| Space / N | Pause or resume / advance one numerical step |
| 1–4 / R | Choose an experiment / reset |
| F / Esc | Immersive view / return to controls |
| H | Scientific field guide |
| P | Save a PNG of the current app window |
| Double-click the volume | Deposit heat on the z = 0 plane |
| Click the volume | Probe the nearest cell on z = 0, including its local clock rate |
| Slice z | Cut into the computed volume |
| Guided tour | Cycle through experiments every 35 wall seconds |
| Save PNG / Export VTK | Capture the app / export physical fields |

Model, grid, and jet-speed changes restart the experiment. Time / s specifies requested simulated units per wall second; actual progress is limited by the CPU. In relativity, time is measured in L₀/c. Speed and Lorentz factor are calculated from the fluid state. The optional Doppler brightness effect is explicitly illustrative; it does not change the dynamics.

## Verify and inspect

```sh
cargo test --locked
cargo run --release --locked -- --validate
cargo clippy --locked --all-targets -- -D warnings
cargo run --release --locked -- --headless --scene jet --steps 600
cargo run --release --locked -- --headless --scene blast --steps 100 --export
```

The validation command reports measured errors against analytical advection, sound waves, the exact Sod solution, a stationary relativistic shock, the Newtonian limit, and 3D conservation. It exits nonzero if a check fails. See [the scientific model](docs/SCIENCE.md) for equations, initial conditions, boundary conditions, and limitations.

```sh
# These two commands require a graphics display.
cargo run --release --locked -- --smoke
cargo run --release --locked -- --scene jet --capture captures/jet.png --capture-after 2.2
```

PNG and VTK files use exclusive creation so existing outputs are not overwritten. Choose a new path when repeating a capture. Interactive saves get unique timestamps. `--paused --capture` requires `--capture-after 0`. VTK files can be opened in ParaView; they contain cell-center samples of proper density, pressure, velocity, lab-frame mass and total energy, plus integral budgets.

`cargo run --release -- --help` lists all options. `--headless` and `--validate` need no display. If a Linux compositor cannot create the graphics context, try an X11 session or `env -u WAYLAND_DISPLAY cargo run --release --locked` on a system with Xwayland available.

## Project map

- `src/physics.rs`: conservative fluid equations, recovery, Riemann flux, 3D stepping.
- `src/cases.rs`: reproducible initial and boundary conditions.
- `src/validation.rs`: analytical checks and conservation regressions.
- `src/worker.rs`: simulation thread, snapshots and VTK export.
- `src/renderer.rs`, `src/shaders/`: GPU volume integration and HDR compositing.
- `src/app.rs`: experiments, measurements, controls and field guide.

No network connection, external image assets, or account is needed at runtime.
