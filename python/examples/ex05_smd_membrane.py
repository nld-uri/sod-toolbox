"""Example 5. SMD of a membrane: the snapshots need no time order, and as many snapshots as modes suffice.

Rectangular membrane (a = 1, b = 0.7, fixed edges) on an 81 x 57 grid. The field is a random combination of its
first eight modes, sampled as unordered snapshots with random amplitudes: no time axis is used. SMD with the
gradient operator returns the modes and the wavenumbers k = pi sqrt((m/a)^2 + (n/b)^2), to discretization error,
from 200 snapshots, from the same snapshots shuffled, and from only eight snapshots. The remaining error is the
central-difference bias, k -> sqrt((sin(k_x h_x)/h_x)^2 + (sin(k_y h_y)/h_y)^2), the spatial twin of Example 2.
"""
from _common import setup, fin
import numpy as np
import matplotlib.pyplot as plt
import sodtoolbox as st
from sodtoolbox import plotting as pl

a = setup(__doc__)
A, B = 1.0, 0.7
x, y = np.linspace(0, A, 81), np.linspace(0, B, 57)
g = st.make_grid(x=x, y=y)
mn = [(m, n) for m in range(1, 5) for n in range(1, 4)]
kk = np.array([np.pi * np.hypot(m / A, n / B) for m, n in mn])
keep = np.argsort(kk)[:8]
mn = [mn[i] for i in keep]; kk = kk[keep]
S = np.stack([np.outer(np.sin(n * np.pi * y / B), np.sin(m * np.pi * x / A)).ravel() for m, n in mn], 1)
rng = np.random.default_rng(5)


def run(nt, shuffle=False):
    C = rng.standard_normal((nt, 8))
    if shuffle:
        C = C[rng.permutation(nt)]
    W = (C @ S.T).reshape(nt, len(y), len(x))
    return st.smd(W, g, 'grad', 'fd', center=False)


def mac(p, q):
    return (p @ q) ** 2 / ((p @ p) * (q @ q))


R = run(200)
for lab, RR in (('200 snapshots', R), ('200, shuffled', run(200, True)), ('8 snapshots', run(8))):
    err = RR.k / kk - 1
    md = max(1 - mac(RR.Phi[:, i], S[:, i]) for i in range(8))
    print(f'{lab:15s}: max |k/k_exact - 1| = {np.max(np.abs(err)):.2e}, max (1 - MAC) = {md:.1e}')
print('modes (m, n):', mn)
print('k exact     :', np.round(kk, 4))
print('k SMD (200) :', np.round(R.k, 4))

hx, hy = x[1] - x[0], y[1] - y[0]
kx = np.array([m * np.pi / A for m, n in mn]); ky = np.array([n * np.pi / B for m, n in mn])
kfd = np.sqrt((np.sin(kx * hx) / hx) ** 2 + (np.sin(ky * hy) / hy) ** 2)   # central-difference wavenumber
print('central-difference prediction k_fd/k - 1:', ' '.join(f'{v:+.1e}' for v in kfd / kk - 1))
fig = plt.figure(figsize=(6.6, 2.7))
gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 0.55, 1.9], wspace=0.15, hspace=0.3,
                      left=0.01, right=0.98, top=0.92, bottom=0.17)
for i in range(8):
    ax = fig.add_subplot(gs[i // 4, i % 4])
    ph = R.mode(i); ph = ph / np.max(np.abs(ph))
    ax.contourf(x, y, ph, levels=np.linspace(-1, 1, 11), cmap='RdBu_r')
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(rf'$({mn[i][0]},{mn[i][1]})$', fontsize=8, pad=2)
ax = fig.add_subplot(gs[:, 5])
ax.plot(kk, R.k / kk - 1, 'o', mfc='none', color=pl.color(0), label='200 snapshots')
ax.plot(kk, run(8).k / kk - 1, 'x', color=pl.color(1), label='8 snapshots')
ax.plot(kk, kfd / kk - 1, '--', color=pl.ref(), lw=0.8, label='central-difference bias')
ax.set_xlabel(r'exact wavenumber $k_{mn}$'); ax.set_ylabel(r'$\hat k/k-1$')
ax.legend(fontsize=7, loc='lower left')
fin(fig, a.out, 'ex05_smd_membrane.pdf')
