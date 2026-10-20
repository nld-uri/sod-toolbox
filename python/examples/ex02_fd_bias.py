"""Example 2. Central differences bias SOD frequencies to sin(omega dt)/dt; cd_correct inverts the bias.

Same chain and record as Example 1. Velocities are formed from the sampled displacement by central differences
(derivative(..., 'central')); for a sinusoid the difference quotient multiplies the amplitude by sin(omega dt)/(omega dt),
so the SOD frequency of an exact mode is sin(omega dt)/dt. The relative error is about -(omega dt)^2/6. After the
correction the error falls to the finite-record floor of Example 1 (about 3e-4 in frequency for mode 1).
"""
from _common import setup, fin
import numpy as np
import matplotlib.pyplot as plt
import sodtoolbox as st
from sodtoolbox import systems as sy, plotting as pl

a = setup(__doc__)
M, K, C = sy.chain(4)
w, _ = sy.lnm_modes(M, K)
y0 = sy.equal_energy(M, K, rng=np.random.default_rng(1))
dts = np.geomspace(0.01, 0.4, 7); raw, cor = [], []
for dt in dts:
    t, X, V, _ = sy.free_decay(M, K, C, y0, dt, int(4000 / dt))
    R = st.sod(X, dt=dt, deriv='central')
    raw.append(R.freq / w - 1); cor.append(st.cd_correct(R.freq, dt) / w - 1)
raw, cor = np.array(raw), np.array(cor)
for dt, r, c in zip(dts, raw, cor):
    print(f'dt = {dt:.3f}: raw', ' '.join(f'{x:+.1e}' for x in r), ' | corrected', ' '.join(f'{x:+.1e}' for x in c))

fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
wdt = np.outer(dts, w)
for i in range(4):
    ax[0].loglog(wdt[:, i], -raw[:, i], 'o', mfc='none', ms=5, color=pl.color(i), label=f'mode {i + 1}')
    ax[1].loglog(wdt[:, i], np.abs(cor[:, i]), 's', mfc='none', ms=4, color=pl.color(i))
g = np.geomspace(3e-3, 1.2, 50)
ax[0].loglog(g, 1 - np.sin(g) / g, '--', color=pl.ref(), lw=0.9, label=r'$1-\sin(\omega\Delta t)/\omega\Delta t$')
ax[0].set_xlabel(r'$\omega_i\Delta t$'); ax[0].set_ylabel(r'$1-\hat\omega_i/\omega_i$ (central differences)')
ax[0].legend(fontsize=7.5, loc='upper left')
ax[1].set_xlabel(r'$\omega_i\Delta t$'); ax[1].set_ylabel(r'$|\hat\omega_i/\omega_i-1|$ after \texttt{cd\_correct}')
ax[1].set_ylim(1e-7, 1e-1)
fig.tight_layout()
fin(fig, a.out, 'ex02_fd_bias.pdf')
