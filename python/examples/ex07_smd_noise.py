"""Example 7. SMD of noisy full-field data: regularized spatial derivatives and the choice of cutoff.

Cantilever beam (401 points), first three modes, 40 snapshots, ten realizations of white measurement noise of 0.1, 0.5 and 2 percent
of the field's standard deviation. The second derivative amplifies noise as k^2, so plain finite differences fail
(wavenumber errors of order one). Whittaker smoothing (penalty order 4, gain 1/(1 + (k/kc)^8)) and Savitzky-Golay
local cubics (window from the measured half-gain cutoff) restore the wavenumbers. The scan over the cutoff kc shows
the trade-off: too low a cutoff attenuates the highest mode, too high a cutoff lets noise in. The flat part of the
curve is where to work; smd_kc_scan finds it on real data, where the true wavenumbers are unknown.
The best Whittaker cutoff falls as the noise grows, from about 5 k_max at 0.1% to about 3 k_max at 2% (medians of
ten noise realizations).
"""
import warnings
from _common import setup, fin
import numpy as np
import matplotlib.pyplot as plt
import sodtoolbox as st
from sodtoolbox import plotting as pl

warnings.filterwarnings('ignore', message='Whittaker system condition')
a = setup(__doc__)
bl = np.array([1.8751040687, 4.6940911330, 7.8547574382])
x = np.linspace(0, 1, 401); g = st.make_grid(x=x)


def cant(b):
    s = (np.cosh(b) + np.cos(b)) / (np.sinh(b) + np.sin(b))
    return np.cosh(b * x) - np.cos(b * x) - s * (np.sinh(b * x) - np.sin(b * x))


rng = np.random.default_rng(7)
W0 = (rng.standard_normal((40, 3)) * [1, 0.5, 0.25]) @ np.column_stack([cant(b) for b in bl]).T
ratios = np.geomspace(1.2, 40, 18)
NR = 10                                                         # noise realizations; medians are reported
res = {}
for lev in (0.001, 0.005, 0.02):
    FD, WH, SG = [], [], []
    sgr = ratios[ratios >= 3]
    for r in range(NR):
        W = W0 + lev * np.std(W0) * rng.standard_normal(W0.shape)
        FD.append(np.max(np.abs(st.smd(W, g, 'second', 'fd', rank=3).k / bl - 1)))
        WH.append(np.max(np.abs(st.smd_kc_scan(W, g, 'second', 'whittaker', ratios * bl[-1], rank=3) / bl - 1), axis=1))
        SG.append(np.max(np.abs(st.smd_kc_scan(W, g, 'second', 'savgol', sgr * bl[-1], rank=3) / bl - 1), axis=1))
    fd, wh, sg = np.median(FD), np.median(WH, 0), np.median(SG, 0)
    res[lev] = (fd, wh, sgr, sg)
    i = np.argmin(wh)
    print(f'noise {100 * lev:.1f}% (median of {NR}): plain fd {fd:.2e}; Whittaker best {wh[i]:.2e} at kc = {ratios[i]:.1f} k_max; '
          f'Savitzky-Golay best {np.min(sg):.2e} at kc = {sgr[np.argmin(sg)]:.1f} k_max')
    for r_, e in zip(ratios[::3], wh[::3]):
        print(f'    kc/k_max = {r_:5.1f}: Whittaker max |k/beta - 1| = {e:.2e}')

fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
for c, lev in enumerate(res):
    fd, wh, sgr, sg = res[lev]
    ax[0].loglog(ratios, wh, 'o-', ms=3.5, mfc='none', color=pl.color(c), label=f'{100 * lev:g}\\% noise')
    ax[0].axhline(fd, color=pl.color(c), ls=':', lw=0.9)
    ax[1].loglog(sgr, sg, 's-', ms=3.5, mfc='none', color=pl.color(c), label=f'{100 * lev:g}\\% noise')
ax[0].set_title('Whittaker (dotted: plain finite differences)', fontsize=8)
ax[1].set_title('Savitzky--Golay', fontsize=8)
for A in ax:
    A.set_xlabel(r'cutoff $k_c/k_{\max}$')
ax[0].set_ylabel(r'$\max_i|\hat k_i/\beta_i-1|$'); ax[1].legend(fontsize=7)
fig.tight_layout()
fin(fig, a.out, 'ex07_smd_noise.pdf')
