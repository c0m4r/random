# Scientific model and numerical contract

## Scope and units

The material is a compressible, inviscid, gamma-law perfect fluid. Classical and relativistic modes evolve the same five physical conservation laws in their respective regimes. They share primitive initial data when the model selector changes. The box is 3 × 2 × 2 reference lengths, sampled at cell centers. Grid refinement changes the discretization, not the box size.

Newtonian quantities use reference density ρ₀, length L₀, and speed v₀; pressure is measured in ρ₀v₀² and time in L₀/v₀. Relativistic quantities use c = 1, pressure in ρ₀c² and time in L₀/c. A relativistic code-time interval of 1 represents a light crossing of one reference length. No particular astronomical size, gas composition or laboratory SI scale is assumed.

## Continuum equations

In the Newtonian model the conserved vector is U = (ρ, ρvₓ, ρvᵧ, ρv_z, E), where E = p/(Γ−1) + ρ|v|²/2. Along axis i its flux is (ρvᵢ, ρv vᵢ + p eᵢ, (E+p)vᵢ).

In the relativistic model, W = (1−|v|²)^(−1/2), h = 1 + Γp/[(Γ−1)ρ], and U = (D, Sₓ, Sᵧ, S_z, E), with D = ρW, S = ρhW²v, and E = ρhW²−p. The axis-i flux is (Dvᵢ, S vᵢ + p eᵢ, Sᵢ). This implementation evolves total E, **including rest mass**, rather than τ = E−D. In both modes ∂tU + Σᵢ ∂ᵢFᵢ = 0 except for explicitly requested heat deposition.

These are the flat-spacetime perfect-fluid equations described by [Martí & Müller, Numerical Hydrodynamics in Special Relativity (2003)](https://doi.org/10.12942/lrr-2003-7). Their conserved energy variable τ differs from this implementation's E by D. There is no curved metric or gravitational source.

Newtonian acoustic characteristics are vᵢ ± √(Γp/ρ). Relativistic sound speed squared is cₛ² = Γp/(ρh). The multidimensional relativistic characteristics are

```text
λ± = [vᵢ(1−cₛ²) ± cₛ√((1−v²)(1−v²cₛ²−vᵢ²(1−cₛ²)))] / (1−v²cₛ²).
```

The transverse components of velocity therefore affect wave propagation even across a face normal to x. Γ is between 1 and 2 so the gamma-law EOS is causal. In the cold, slow limit, D → ρ, S → ρv, and E−D → Newtonian E.

## Discretization and positivity

Cell averages evolve with conservative finite volumes, minmod-limited MUSCL reconstruction of primitive variables, and unsplit two-stage SSP-RK2 time integration. The SR solver uses two-wave HLLE fluxes. The Newtonian solver uses contact-resolving HLLC fluxes with HLLE fallback if either star state is inadmissible, following the contact-restoration approach of [Toro, Spruce & Speares (1994)](https://doi.org/10.1007/BF01414629). This prevents unnecessary diffusion of stationary classical shear interfaces. One shared flux contributes equal and opposite changes to adjacent cells. Reconstruction falls back to zero local slope if an extrapolated primitive state is inadmissible.

The time step is 0.35 / Σᵢ(max|λᵢ|/Δxᵢ), including the inlet state. This is a multidimensional CFL bound, not three independent one-dimensional limits. The stage and final states must recover finite positive density and pressure and, in SR, |v| < 1. Rejected attempts halve the time step; after four failed attempts first-order spatial fluxes are tried. After twelve failures the worker pauses on the last valid state with a visible error. No accepted cell is repaired by clamping its conserved fields.

SR recovery solves the scalar pressure equation with a bracketed Newton iteration, after checking E > √(D²+|S|²). An independent reconstruction of E checks root convergence. The solver uses f64 throughout; the graphics texture uses f32 and never feeds back into the solver. Total-energy recovery can lose precision for extremely cold, high-Mach states; the provided scenarios and tested recovery envelope avoid that regime. This is not a universal arbitrary-Lorentz-factor solver.

## Initial and boundary conditions

| Experiment | Initial state | Boundary / Γ |
| --- | --- | --- |
| Jet | Ambient ρ=1, p=0.02, v=0. A radius-0.22 circular inlet has ρ=0.15, p=0.02, vₓ=0.92 by default. A short initial beam occupies x < −1.1. | Prescribed inlet at x=−1.5 inside nozzle; zero-gradient elsewhere. Γ=4/3. |
| Spherical blast | ρ=1, v=0. Smooth radial pressure: 0.015 + 1−tanh((r−0.26)/0.045). | Zero-gradient outflow. Γ=4/3. |
| Shear | L=½[1+tanh((0.45−abs(y))/0.09)], ρ=1+L, vₓ=0.7(L−0.5), p=0.3. Transverse seed vᵧ=0.035 sin(2πx/1.5) [1+0.15 cos(πz)] exp(−((abs(y)−0.45)/0.18)²). | Periodic on all axes. Γ=4/3. |
| Shock tube | At x<0: ρ=1, p=1. At x≥0: ρ=0.125, p=0.1. Initially at rest. | Zero-gradient outflow. Γ=1.4. |

Zero-gradient boundaries are approximate open boundaries, not exact non-reflecting boundaries. When expanding structures reach them, the domain is no longer an isolated system. The jet injects mass, momentum and energy continuously. Switching from SR to Newtonian dynamics at the same nominal v does not constitute the cold, low-speed limit: the chosen pressure and velocity must both be small relative to rest energy and c for that comparison.

The heat tool adds a Gaussian ΔE = 0.8 exp(−r²/0.035) directly to lab-frame energy density, keeping lab-frame mass and momentum fixed. This represents an external energy source with zero net supplied momentum. It is included in the energy budget. It is not a force or an instantaneous arbitrary velocity change.

## Measurements and rendering

The HUD shows actual simulation time, last solve duration, frame cadence, maximum speed and W, and conservation residuals. The residual is (current integral − initial integral − integrated exchange) / max(|initial integral|, 1). The exchange uses the same RK stage weights and the actual boundary face fluxes, plus heat sources. For initially near-zero momentum this denominator gives an absolute code-unit scale rather than a relative percentage. Conservation says nothing by itself about truncation error; the analytical tests below measure solution error separately.

The displayed 3D color is a fixed transfer function of proper density, pressure, speed, or W. The legend specifies the linear/log range; values outside it are color-clipped, while the simulation remains unchanged. Opacity depends on the chosen scalar and its gradient. Lighting and bloom are artistic aids, not a radiation calculation. The 2D density trace is sampled from the cells nearest y=z=0, with a displayed vertical range. Streamlines integrate the instantaneous trilinearly sampled velocity direction; they are not pathlines or Lagrangian markers and are drawn as an explanatory overlay.

Optional beaming applies δ³ to illustrative emissivity, with δ = 1/[W(1−v·n)] and n toward the camera. This corresponds to the familiar Doppler factor for a frequency-specific intensity transformation when comparing corresponding emitted/observed frequencies; it is not bolometric δ⁴ radiative transfer. The display caps the multiplier at 15 and floors it at 0.025 to preserve readability. This option ignores retarded time, spectral emission, aberrated source geometry and radiation forces. Camera movement does not Lorentz-transform the computational mesh. The local comoving clock relation is dτ/dt = 1/W; a peak-W measurement is not a globally moving observer's clock.

The distinction between specific intensity and emissivity matters: a full transfer solver would use the respective Lorentz invariants and a spectral source function. See [RAPTOR I, radiative transfer equation (30)](https://www.aanda.org/component/article?access=doi&doi=10.1051%2F0004-6361%2F201732149&mb=0). This app deliberately labels its beaming visualization as illustrative.

## Validation

`cargo test` and `--validate` exercise the same public simulation core used by the live app. The checks include:

- Primitive/conservative SR round trips spanning ρ=0.01–10, p=10⁻⁴–100 and v up to 0.999c (W≈22.4).
- Exact smooth periodic contact advection at 48 and 96 cells in both models. Refinement must reduce L1 error to less than 0.48 of its previous value, consistent with second-order convergence on this limited smooth solution.
- Three-dimensional uniform-state preservation with nonzero velocity in all three axes.
- A Newtonian Sod tube at t=0.2 compared pointwise to an independently constructed exact Riemann solution, including its rarefaction fan, contact and shock.
- The cold Newtonian limit of SR momentum and energy excluding rest mass.
- A small-amplitude SR acoustic wave compared with its analytical phase velocity and perturbation eigenvector.
- A stationary strong SR normal shock, whose downstream state is independently obtained from mass, momentum and energy Rankine–Hugoniot invariants, evolved to t=0.25 and compared with the exact stationary discontinuity.
- All four three-dimensional scenarios in both models, with boundary/source budgets, positivity, subluminal speed, and no rejected time steps during the regression interval.
- Rejection of inadmissible conserved states and causal multidimensional characteristic speeds, including the one-dimensional velocity-addition limit.

The validation report prints tolerances and measured errors. A passing tolerance does not imply infinite resolution, exact shock locations at all later times, or convergence of turbulent statistics. Numerical viscosity from the limiter and HLLE flux broadens interfaces and suppresses unresolved instability. The grid is deliberately modest for interactive CPU computation.

There is no liquid free surface, physical viscosity, surface tension, magnetohydrodynamics, self-gravity, general relativity or quantitatively modeled radiation. These omissions are explicit in the app because they materially affect how a viewer should interpret the simulation.
