"""Example 10. SMD of a simply supported Kirchhoff plate: natural frequencies from the bending energy.

Plate a = 1, b = 0.6, flexural rigidity D = 1, mass per area rho h = 1, Poisson ratio 0.3, so that
omega_mn = pi^2 ((m/a)^2 + (n/b)^2). The field is a random combination of the first eight modes on a 101 x 61 grid,
40 unordered snapshots. With density and stiffness weights the 'plate' operator (the bending energy
(1-nu)(w_xx^2 + 2 w_xy^2 + w_yy^2) + nu (w_xx + w_yy)^2) gives mu = omega^2. On simply supported (and clamped)
rectangles the Gaussian-curvature term integrates to zero, so the Hessian ('second') and Laplacian operators agree
with it to discretization error (all within 0.22 percent here); with free edges only the plate energy is matched.
"""
from _common import setup, fin
import numpy as np
import matplotlib.pyplot as plt
import sodtoolbox as st
from sodtoolbox import plotting as pl

a = setup(__doc__)
A, B = 1.0, 0.6
x, y = np.linspace(0, A, 101), np.linspace(0, B, 61)
g = st.make_grid(x=x, y=y)
mn = sorted([(m, n) for m in range(1, 6) for n in range(1, 4)], key=lambda p: (p[0] / A) ** 2 + (p[1] / B) ** 2)[:8]
w = np.array([np.pi ** 2 * ((m / A) ** 2 + (n / B) ** 2) for m, n in mn])
S = np.stack([np.outer(np.sin(n * np.pi * y / B), np.sin(m * np.pi * x / A)).ravel() for m, n in mn], 1)
rng = np.random.default_rng(10)
W = (rng.standard_normal((40, 8)) @ S.T).reshape(40, len(y), len(x))
err = {}
for op in ('plate', 'second', 'laplacian'):
    R = st.smd(W, g, op, 'fd', center=False, density=1.0, stiffness=1.0, nu=0.3)
    err[op] = R.omega / w - 1
    if op == 'plate':
        Rp = R
    print(f'{op:9s}: omega_SMD/omega - 1 =', ' '.join(f'{v:+.1e}' for v in err[op]))
print('modes (m, n):', mn)
print('omega exact :', np.round(w, 3))

fig = plt.figure(figsize=(6.6, 2.7))
gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 0.5, 1.9], wspace=0.15, hspace=0.3,
                      left=0.01, right=0.98, top=0.92, bottom=0.17)
for i in range(8):
    ax = fig.add_subplot(gs[i // 4, i % 4])
    ph = Rp.mode(i); ph = ph / np.max(np.abs(ph))
    ax.contourf(x, y, ph, levels=np.linspace(-1, 1, 11), cmap='RdBu_r')
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(rf'$({mn[i][0]},{mn[i][1]})$', fontsize=8, pad=2)
ax = fig.add_subplot(gs[:, 5])
for c, (op, mk) in enumerate(zip(err, 'o^s')):
    ax.plot(w, 100 * err[op], mk, mfc='none', ms=4, color=pl.color(c), label=rf'\texttt{{{op}}}')
ax.set_xlabel(r'exact $\omega_{mn}$ (rad/s)'); ax.set_ylabel(r'$\hat\omega/\omega-1$ (\%)')
ax.legend(fontsize=7)
fin(fig, a.out, 'ex10_smd_plate.pdf')
