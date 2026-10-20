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

## Tests

- `python/tests/test_sod.py`: 15 self-contained checks of known properties.
- `python/tests/make_reference.py` writes `matlab/sod_reference.mat`;
  `matlab/test_sod.m` checks the MATLAB port against it (32 checks).

## Layout

    python/sodtoolbox/   core, recognize, damping, tracking, rom, residual, systems, plotting
    python/examples/     four worked examples
    docs/                manual, slides, animations, web images
    python/tests/        self-contained tests and the MATLAB reference generator
    matlab/              sod_*.m, test_sod.m, sod_reference.mat

## License

MIT. See `LICENSE`. Fonts in `docs/manual/fonts` are IBM Plex, SIL Open Font License (`OFL-LICENSE.txt`).
