"""Example 6. SMD of a non-uniform beam: density and stiffness weights give the exact modes and mu = omega^2.

Pinned-pinned beam, L = 1, EI = 1 and rho A = 1 + 2x (mass grows threefold along the span). Reference modes and
frequencies come from a Galerkin model with 100 sine functions, independent of the SMD finite differences. Six
snapshots of the first six modes, with random amplitudes, are enough. With unit weights SMD returns biased shapes,
because the true modes are orthogonal in the mass product, not in plain L2. With density in S_W and EI in S_G the
pencil is the beam's Rayleigh quotient: the modes are exact and sqrt(mu) are the natural frequencies, with the
second-order error of the central differences.
"""
from _common import setup, fin
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import sodtoolbox as st
from sodtoolbox import plotting as pl

a = setup(__doc__)
EI = lambda x: np.ones_like(x)
rho = lambda x: 1 + 2 * x
xq, wq = np.polynomial.legendre.leggauss(800); xq = (xq + 1) / 2; wq = wq / 2
j = np.arange(1, 101) * np.pi; Sq = np.sin(np.outer(xq, j))
Mg = (Sq * (wq * rho(xq))[:, None]).T @ Sq
Kg = (Sq * j ** 2 * (wq * EI(xq))[:, None]).T @ (Sq * j ** 2)
w2, Cg = eigh(Kg, Mg); w = np.sqrt(w2[:6])
rng = np.random.default_rng(6)


def mac(p, q):
    return (p @ q) ** 2 / ((p @ p) * (q @ q))


def run(n, weighted):
    x = np.linspace(0, 1, n); g = st.make_grid(x=x)
    Phi = np.sin(np.outer(x, j)) @ Cg[:, :6]
    W = rng.standard_normal((6, 6)) @ Phi.T
    kw = dict(density=rho(x), stiffness=EI(x)) if weighted else {}
    R = st.smd(W, g, 'second', 'fd', center=False, **kw)
    return x, Phi, R


x, Phi, Ru = run(401, False)
_, _, Rp = run(401, True)
mu = [1 - mac(Ru.Phi[:, i], Phi[:, i]) for i in range(6)]
mp = [1 - mac(Rp.Phi[:, i], Phi[:, i]) for i in range(6)]
print('omega (Galerkin)       :', np.round(w, 4))
print('omega (SMD, physical)  :', np.round(Rp.omega, 4))
print('1 - MAC, unit weights  :', ' '.join(f'{v:.1e}' for v in mu))
print('1 - MAC, physical      :', ' '.join(f'{v:.1e}' for v in mp))
ns = [101, 201, 401, 801, 1601]; ferr = []
for n in ns:
    ferr.append(np.max(np.abs(run(n, True)[2].omega / w - 1)))
    print(f'n = {n:5d}: max |omega_SMD/omega - 1| = {ferr[-1]:.2e}')

fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw=dict(width_ratios=[1.4, 1]))
for i in range(3):
    s = np.sign(Phi[:, i] @ Ru.Phi[:, i]); t = Phi[:, i] / np.max(np.abs(Phi[:, i]))
    u = s * Ru.Phi[:, i] / np.max(np.abs(Ru.Phi[:, i]))
    ax[0].plot(x, t, '-', color=pl.color(i), lw=1.2, label=f'mode {i + 1}, true')
    ax[0].plot(x, u, '--', color=pl.color(i), lw=1.0, label='unit weights')
ax[0].set_xlabel(r'$x/L$'); ax[0].set_ylabel(r'$\phi_i$ (max-normalized)')
ax[0].legend(fontsize=6.5, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False)
h = 1 / (np.array(ns) - 1)
ax[1].loglog(h, ferr, 'o-', color=pl.color(0), mfc='none', label='physical weights')
ax[1].loglog(h, ferr[2] * (h / h[2]) ** 2, '--', color=pl.ref(), lw=0.8, label=r'$\propto h^2$')
ax[1].set_xlabel(r'grid spacing $h/L$'); ax[1].set_ylabel(r'$\max_i|\hat\omega_i/\omega_i-1|$')
ax[1].legend(fontsize=7)
fig.tight_layout()
fin(fig, a.out, 'ex06_smd_nonuniform_beam.pdf')
