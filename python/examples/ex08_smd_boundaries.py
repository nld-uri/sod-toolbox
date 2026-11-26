"""Example 8. Boundaries, partial fields of view and holes bias SMD: the spatial counterpart of SOD's window bias.

Exactness needs the true shapes to be orthogonal in both the L2 and the energy products over the measured domain.
On the whole structure the boundary conditions remove the integration-by-parts terms; on a sub-domain they do not.
Left: cantilever (first three modes), measured only on a centered window covering a fraction of the span. The
wavenumber bias is near zero on the full span and reaches about 10 percent once a fifth of the span is missing.
Right: rectangular membrane (first four modes) with a circular hole of growing radius at (0.6, 0.35); masked points
are filled by Whittaker smoothing for the derivatives and excluded from both integrals. The (1,1) mode, whose
antinode lies near the hole, is biased most (about 11 percent at r = 0.14).
"""
import warnings
from _common import setup, fin
import numpy as np
import matplotlib.pyplot as plt
import sodtoolbox as st
from sodtoolbox import plotting as pl

warnings.filterwarnings('ignore', message='Whittaker system condition')
a = setup(__doc__)
rng = np.random.default_rng(8)
bl = np.array([1.8751040687, 4.6940911330, 7.8547574382])
x = np.linspace(0, 1, 801)


def cant(b, xx):
    s = (np.cosh(b) + np.cos(b)) / (np.sinh(b) + np.sin(b))
    return np.cosh(b * xx) - np.cos(b * xx) - s * (np.sinh(b * xx) - np.sin(b * xx))


Wb = rng.standard_normal((30, 3)) @ np.column_stack([cant(b, x) for b in bl]).T
fr = np.linspace(0.3, 1.0, 15); kb = []
for f in fr:
    sel = (x >= 0.5 - f / 2 - 1e-12) & (x <= 0.5 + f / 2 + 1e-12)
    kb.append(st.smd(Wb[:, sel], st.make_grid(x=x[sel]), 'second', 'fd', center=False).k / bl - 1)
kb = np.array(kb)
for f, e in zip(fr[::2], kb[::2]):
    print(f'field of view {f:.2f} L: k/beta - 1 =', ' '.join(f'{v:+.2e}' for v in e))

A, B = 1.0, 0.7; xm, ym = np.linspace(0, A, 101), np.linspace(0, B, 71)
mn = [(1, 1), (2, 1), (1, 2), (3, 1)]
km = np.array([np.pi * np.hypot(m / A, n / B) for m, n in mn])
M = np.stack([np.outer(np.sin(n * np.pi * ym / B), np.sin(m * np.pi * xm / A)).ravel() for m, n in mn], 1)
Wm = (rng.standard_normal((30, 4)) @ M.T).reshape(30, len(ym), len(xm))
rad = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14]); kh = []
XX, YY = np.meshgrid(xm, ym)
for r in rad:
    hole = np.hypot(XX - 0.6, YY - 0.35) < r
    g = st.make_grid(x=xm, y=ym, mask=~hole)
    kh.append(st.smd(Wm, g, 'grad', 'whittaker', kc=30 * km.max(), center=False).k / km - 1)
kh = np.array(kh)
for r, e in zip(rad, kh):
    print(f'hole radius {r:.2f}: k/k_mn - 1 =', ' '.join(f'{v:+.2e}' for v in e))

fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
for i in range(3):
    ax[0].semilogy(fr, np.abs(kb[:, i]), 'o-', ms=3.5, mfc='none', color=pl.color(i), label=f'mode {i + 1}')
ax[0].set_xlabel('measured fraction of the span'); ax[0].set_ylabel(r'$|\hat k_i/\beta_i-1|$')
ax[0].legend(fontsize=7); ax[0].set_title('cantilever, centered field of view', fontsize=8)
for i in range(4):
    ax[1].plot(rad, 100 * kh[:, i], 's-', ms=3.5, mfc='none', color=pl.color(i), label=rf'$({mn[i][0]},{mn[i][1]})$')
ax[1].set_xlabel(r'hole radius $r/a$'); ax[1].set_ylabel(r'$\hat k/k_{mn}-1$ (\%)')
ax[1].legend(fontsize=7); ax[1].set_title('membrane with a circular hole', fontsize=8)
fig.tight_layout()
fin(fig, a.out, 'ex08_smd_boundaries.pdf')
