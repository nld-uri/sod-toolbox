# SOD Toolbox

Smooth orthogonal decomposition (SOD) for modal identification and reduced-order
modelling, in Python and base MATLAB. Nonlinear Dynamics Laboratory, University of
Rhode Island. It is the common code base of a series of SOD papers in preparation.

SOD solves one real symmetric generalized eigenproblem,

    Sigma_v psi = mu Sigma_x psi,     Sigma_x = <x x'>,  Sigma_v = <xdot xdot'>,

with `mu` ascending (smoothest first). Conventions throughout:

| quantity | definition |
|---|---|
| covectors `Psi` | `psi_i' Sigma_x psi_j = delta_ij` |
| smooth coordinates `Q` | `Q = X Psi` (unit variance) |
| smooth modes `Phi` | `Phi = Sigma_x Psi`, so `Psi' Phi = I` |
| SOD values `mu` | `<qdot_i^2>/<q_i^2>`; `sqrt(mu)` is the SOD frequency |
| smooth orthogonal values | `1/mu` |

For an undamped linear system whose modal coordinates are uncorrelated, the SOVs
are `1/omega_i^2` and the smooth modes are the linear normal modes. Every variant
below is this pencil. The options change only how the covariances are formed.

SOD here is the temporal form: it smooths the coordinates in time. Its spatial dual,
smooth mode decomposition (SMD; Li, Wanchoo, Shukla & Chelidze, MSSP 200 (2023) 110541),
smooths the mode shapes in space and suits full-field data that are richer in space than
in time; the `smd` module implements it (section below). A smooth mode of SOD is the shape paired with a
temporally smooth coordinate, not a spatially smoothed shape.

## Rules the toolbox enforces

- Modes and modal planes are recognised, paired and ordered by frequency or by
  circularity. The SOD value (smoothness) is never used for this.
- Model reduction uses model kinematics. Identification uses estimated kinematics
  (for example from the kinematics toolbox). The two selections are reported apart.

## Functions

| task | Python (`sodtoolbox`) | MATLAB |
|---|---|---|
| configuration SOD (lag, centring, scaling, rank restriction) | `sod` | `sod_decomp` |
| derivative of a record (central step h, gradient) | `derivative` | `sod_deriv` |
| data-subspace basis | `data_rank_basis` | `sod_rank_basis` |
| covariances (plain, symmetric lagged) | `covariances` | `sod_cov` |
| pencil from given covariances | `sod_pencil` | `sod_pencil` |
| state SOD on y = (x, xdot) | `state_sod` | `sod_state` |
| boundary term and corrected fit | `boundary_term`, `csod` | `sod_boundary`, `sod_csod` |
| windowed SOD (backbones, trends) | `sod_window`, `sod_track` | `sod_window`, `sod_track` |
| DCT periodogram, peak and half-power width | `dct_spectrum`, `peak_and_width` | `sod_dct_spectrum`, `sod_peak_width` |
| ID selection, configuration SOD | `select_config` | `sod_select_config` |
| plane pairing by circularity, banded rescue | `pair_state` | `sod_pair_state` |
| plane fit, eigenvalue, damping identity | `fit_L`, `plane_eigs`, `plane_identity` | `sod_fit_L`, `sod_plane_eigs`, `sod_plane_identity` |
| all damping estimates of a decay (per-plane fits are approximate; the subspace fit is not) | `decay_damping` | -- |
| LNM, MTA, POD, smooth-mode bases | `lnm`, `mta`, `pod`, `sod_modes` | `sod_lnm`, `sod_mta`, `sod_pod`, `sod_modes` |
| complex-mode baseline | `complex_modes` | -- |
| Galerkin, energy-Galerkin ROMs, exact ZOH simulation | `galerkin`, `rom_config`, `rom_state`, `simulate_state` | `sod_galerkin` |
| linear core plus residual basis | `residual_basis`, `residual_diagnostics` | `sod_residual_basis` (kinds a, k, x) |
| metrics | `merr`, `mac`, `principal_angles`, `mortho` | `sod_merr`, `sod_mac`, `sod_mortho` |
| SMD grid (line or grid, mask, quadrature weights) | `make_grid` | `sod_smd_grid` |
| spatial derivative fields (fd, Savitzky-Golay, Whittaker) | `smd_derivatives` | `sod_smd_derivs` |
| smooth mode decomposition, cutoff scan (density/stiffness weights) | `smd`, `smd_kc_scan` | `sod_smd`, `sod_smd_kc_scan`, `sod_smd_mode` |
| maximum autocorrelation factors (Switzer-Green) | `maf` | `sod_smd_maf` |

## Quick start

```python
import sodtoolbox as st
R = st.sod(X, V)                      # X, V: samples x channels
R.freq, R.Phi, R.Q                    # SOD frequencies, smooth modes, coordinates
Rs = st.state_sod(Y, Yd)              # Y = [X V], Yd = [V A]
planes = st.pair_state(Rs.Q, dt, wmax, Phi=Rs.Phi)   # (a, b, frequency, circularity)
D = st.decay_damping(Rs, Yd, dt, wmax)               # per-plane eigenvalues and zeta
```

```matlab
R  = sod_decomp(X, V);
Rs = sod_state(Y, Yd);
planes = sod_pair_state(Rs.Q, dt, wmax, struct('Phi', Rs.Phi));
```

## Traps

- **The rank cut depends on channel scaling.** With `tol` set, directions whose
  covariance eigenvalue falls below `tol` times the largest are dropped. Scaling
  the channels (`scale=True`) changes which weak directions survive. On a forced
  cantilever record of data rank 9 out of 60, the two choices kept different
  subspaces and agreed only on the two smoothest SOD values.
- **Lagged covariances can be indefinite.** `lag > 0` removes white sensor noise
  from both matrices, but fast directions may get negative values. They sort last.
- **The damping identity holds for the chord derivative.** With exact `qdot` and
  sampled sums, `tr L` and its end-state form differ by a relative
  `O((|s| dt)^2 / zeta)`. Lightly damped fast planes need fine sampling.
- **Petrov-Galerkin state ROMs with the data covectors are unstable in general.**
  Use the energy weight (`energy_weight(M, K)` in `rom_state`).

## Documentation

- `docs/manual/SOD_manual.pdf`: user manual (LaTeX source alongside; compile with XeLaTeX).
- `docs/slides/SOD_toolbox_slides.pdf` and `SOD_toolbox.pptx`: NLD-branded slide deck with four animations.
- `docs/animations/`: the animations (MP4) and their stills; `make_animations.py` rebuilds them and
  `report_numbers.py` prints every number the manual quotes for them.
- `python/examples/`, `matlab/examples/`: four worked examples (an undamped chain, central-difference bias,
  sensor noise and recognition, damping from a decay); each Python example stores its printed output.

## Smooth mode decomposition (SMD)

SMD is the spatial dual of SOD for full-field data (Li, Wanchoo, Shukla & Chelidze, MSSP 200 (2023)
110541). It smooths mode shapes in space instead of coordinates in time, with spatial derivatives of the
measured field in place of velocities. Snapshots need no time order, spacing or Nyquist rate.

    S_G p = mu S_W p,  S_W = W diag(w) W',  S_G = sum_k G_k diag(w) G_k'   (snapshots x snapshots)
    phi = W' p,  mu = ||D phi||^2 / ||phi||^2,  k = mu^(1/(2m))  (SMD wavenumber)

Pick the operator from the physics: `'grad'` (m = 1) for strings, membranes and flows; `'second'` (m = 2)
for beams (on a grid: the Hessian); `'laplacian'` (m = 2); `'plate'` (m = 2, Kirchhoff bending energy with Poisson
ratio `nu`). Give `density` (mass per length or area) and `stiffness` (T, EI or plate rigidity) to weight the pencil
physically: then `mu = omega^2` and `R.omega` estimates the natural frequencies of a self-adjoint structure, and
non-uniform structures are handled exactly (unit weights bias their modes, because they are mass-orthogonal, not
L2-orthogonal). Pick the derivative estimator
from the data: `'fd'` (finite differences; noise-free or coarse data), `'whittaker'` (penalized smoothing;
fills masked points), `'savgol'` (local polynomials; complete uniform grids). For noisy data scan the
cutoff `kc` with `smd_kc_scan` and keep a value where the wavenumbers do not move: Whittaker about 3-6
times, Savitzky-Golay about 8 times the largest wavenumber of interest (beam test, 0.5% noise).

```python
g = st.make_grid(x=xs, y=ys, mask=measured)          # line: make_grid(x=xs)
D = st.smd_derivatives(W, g, 'grad', 'whittaker', kc=60.0)   # derivative fields of every snapshot
R = st.smd(W, g, 'grad', 'whittaker', kc=60.0)       # W: snapshots x ny x nx (NaN = missing)
R.k, R.mode(0), R.C                                  # wavenumbers, first mode on the grid, coordinates
K = st.smd_kc_scan(W, g, 'grad', 'whittaker', [30, 60, 120])
```

```matlab
G = sod_smd_grid(xs, ys, measured);
D = sod_smd_derivs(W, G, 'grad', 'whittaker', struct('kc', 60));
R = sod_smd(W, G, 'grad', 'whittaker', struct('kc', 60));
M1 = sod_smd_mode(R, 1);
```

What the tests show (`python/tests/test_smd.py`): SMD values depend only on the span of the snapshots,
so shuffled or correlated snapshots give the same result. On a complete domain with the operator matched
to the stiffness, the modes and wavenumbers of strings, cantilever beams and membranes are recovered to
discretization error. A hole in the field, a partial field of view, or a mismatched operator biases them;
these are the spatial counterparts of SOD's boundary terms. With density and stiffness weights, a tapered
pinned-pinned beam gives its frequencies to 1.3e-4 from five snapshots, one per mode.

The pencil is not new as such: `maf` is the maximum autocorrelation factor transform of Switzer and Green (1984)
with snapshots as bands, which equals SMD with a forward-difference gradient and unit weights.

## Tests

- `python/tests/test_sod.py`: 15 self-contained checks of known properties.
- `python/tests/test_smd.py`: 30 checks of SMD, six of them failure controls that must fail.
- `python/tests/make_smd_reference.py` writes `matlab/smd_reference.mat`; `matlab/test_smd.m`
  checks the MATLAB port against it (42 checks; MATLAB R2026b, Python 3.11).
- `python/tests/make_reference.py` writes `matlab/sod_reference.mat`;
  `matlab/test_sod.m` checks the MATLAB port against it (32 checks).

## Layout

    python/sodtoolbox/   core, recognize, damping, tracking, rom, residual, smd, systems, plotting
    python/examples/     four worked examples
    docs/                manual, slides, animations, web images
    python/tests/        self-contained tests and the MATLAB reference generator
    matlab/              sod_*.m, sod_smd*.m, test_sod.m, test_smd.m, sod_reference.mat, smd_reference.mat

## License

MIT. See `LICENSE`. Fonts in `docs/manual/fonts` are IBM Plex, SIL Open Font License (`OFL-LICENSE.txt`).
