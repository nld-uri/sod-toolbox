"""Example 9. SMD reads natural frequencies from sub-Nyquist and randomly timed snapshots; SOD cannot.

Simply supported uniform beam (L = 1, EI = rho A = 1), so omega_j = (j pi)^2; the highest of six modes is at
56.5 Hz. Free decay with equal energy in the six modes and 0.1 percent modal damping, recorded for 10 s at a sampling
rate fs from 400 Hz down to 2 Hz, on 401 points. SMD with density and stiffness weights (mu = omega^2) uses only
the shapes in the span of the snapshots, so its frequency error stays at the central-difference floor of the
spatial grid at every rate, and also with 12 randomly timed snapshots. SOD on 13 sensors with central-difference
velocities and the cd_correct inversion is accurate while omega dt is small, then fails once the highest mode is
undersampled. SOD with the exact velocities (not available in practice) is shown for reference. Its floor (about
1 percent here, 0.1 percent with other phases) is the finite-record correlation of a 10 s decay. Below Nyquist its sampled averages often stay usable but
fail erratically where aliased frequencies nearly coincide, so the derivative estimate is SOD's first limit and
aliasing its second.
"""
from _common import setup, fin
import numpy as np
import matplotlib.pyplot as plt
import sodtoolbox as st
from sodtoolbox import plotting as pl

a = setup(__doc__)
J = np.arange(1, 7); w = (J * np.pi) ** 2; zeta = 0.001
wd = w * np.sqrt(1 - zeta ** 2)
x = np.linspace(0, 1, 401); g = st.make_grid(x=x)
Phi = np.sin(np.outer(x, J * np.pi))
xs = np.arange(1, 14) / 14; Phis = np.sin(np.outer(xs, J * np.pi))   # 13 sensors for SOD
rng = np.random.default_rng(9)
ph = rng.uniform(0, 2 * np.pi, 6); amp = 1 / w                        # equal energy: amplitude ~ 1/omega


def q_qd(t):
    e = np.exp(-zeta * np.outer(t, w)); c = np.cos(np.outer(t, wd) + ph); s = np.sin(np.outer(t, wd) + ph)
    return amp * e * c, amp * e * (-zeta * w * c - wd * s)


def smd_err(t):
    q, _ = q_qd(t)
    R = st.smd(q @ Phi.T, g, 'second', 'fd', center=False, density=1.0, stiffness=1.0)
    return np.max(np.abs(R.omega / w - 1))


def sod_err(fs, exact):
    t = np.arange(0, 10, 1 / fs); q, qd = q_qd(t)
    X = q @ Phis.T
    if exact:
        f = st.sod(X, qd @ Phis.T).freq
    else:
        f = st.cd_correct(st.sod(X, dt=1 / fs).freq, 1 / fs)
    f = np.sort(np.asarray(f))[:6]
    return np.max(np.abs(f / w - 1))


fss = np.geomspace(400, 2, 16)
E = np.array([[smd_err(np.arange(0, 10, 1 / fs)), sod_err(fs, False), sod_err(fs, True)] for fs in fss])
for fs, e in zip(fss, E):
    print(f'fs = {fs:7.2f} Hz (fs/f6 = {fs / (w[-1] / 2 / np.pi):6.3f}): max rel. frequency error  SMD {e[0]:.2e}   '
          f'SOD (central diff.) {e[1]:.2e}   SOD (exact velocity) {e[2]:.2e}')
er = [smd_err(np.sort(rng.uniform(0, 10, 12))) for _ in range(20)]
print(f'SMD, 12 randomly timed snapshots, 20 draws: max error {max(er):.2e}, median {np.median(er):.2e}')

fig, ax = plt.subplots(figsize=(4.8, 3.3))
f6 = w[-1] / 2 / np.pi
ax.loglog(fss / f6, E[:, 1], 's-', ms=3.5, mfc='none', color=pl.color(1), label='SOD, central differences + \\texttt{cd\\_correct}')
ax.loglog(fss / f6, E[:, 2], '^-', ms=3.5, mfc='none', color=pl.color(2), label='SOD, exact velocities')
ax.loglog(fss / f6, E[:, 0], 'o-', ms=3.5, mfc='none', color=pl.color(0), label='SMD, density and stiffness weights')
ax.axvline(2, color=pl.ref(), ls='--', lw=0.8); ax.set_ylim(5e-5, 5)
ax.text(2.15, 1e-4, 'Nyquist', fontsize=7, color=pl.ref())
ax.set_xlabel(r'sampling rate / highest natural frequency, $f_s/f_6$')
ax.set_ylabel(r'$\max_i|\hat\omega_i/\omega_i-1|$')
ax.legend(fontsize=6.5, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False)
fig.tight_layout()
fin(fig, a.out, 'ex09_smd_subnyquist.pdf')
