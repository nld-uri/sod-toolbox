"""Example 1. SOD of an undamped chain: smooth orthogonal values are 1/omega^2 and smooth modes are the LNMs.

4-DOF chain, free response with equal energy in every mode, exact velocities. The statement holds exactly when the
modal coordinates are uncorrelated over the record; on a finite record their residual correlation leaves an error
of order 1e-3 here, which does not fall monotonically with record length.
"""
from _common import setup, fin
import numpy as np
import matplotlib.pyplot as plt
import sodtoolbox as st
from sodtoolbox import systems as sy, plotting as pl

a = setup(__doc__)
M, K, C = sy.chain(4)
w, P = sy.lnm_modes(M, K)
y0 = sy.equal_energy(M, K, rng=np.random.default_rng(1))
Ts = [250, 1000, 4000]; err = []
for T in Ts:
    t, X, V, _ = sy.free_decay(M, K, C, y0, 0.02, int(T / 0.02))
    R = st.sod(X, V)
    err.append(np.abs(R.sov * w ** 2 - 1))
    print(f'T = {T:5d}: SOV x omega^2 - 1 =', ' '.join(f'{e:+.1e}' for e in R.sov * w ** 2 - 1))
Pn = P / np.linalg.norm(P, axis=0); Phi = R.modes_unit()
macs = [st.mac(Phi[:, i], Pn[:, i]) for i in range(4)]
print('MAC(smooth mode, LNM), T = 4000:', ' '.join(f'{m:.8f}' for m in macs))

fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
ax[0].loglog(1 / w ** 2, R.sov, 'o', mfc='none', ms=7, color=pl.color(0), label='SOD')
g = np.array([0.1, 10]); ax[0].loglog(g, g, '--', color=pl.ref(), lw=0.8, label=r'$\lambda=1/\omega^2$')
ax[0].set_xlabel(r'$1/\omega_i^2$'); ax[0].set_ylabel(r'SOV $\lambda_i$'); ax[0].legend(loc='upper left')
for i in range(4):
    s = np.sign(Phi[:, i] @ Pn[:, i])
    ax[1].plot(np.arange(5), np.r_[0, Pn[:, i]], '-', color=pl.color(i), lw=2.4, alpha=0.35)
    ax[1].plot(np.arange(5), np.r_[0, s * Phi[:, i]], 'o', ms=4, color=pl.color(i), label=f'{i + 1}')
ax[1].set_xlabel('mass'); ax[1].set_ylabel('shape'); ax[1].set_xticks(range(5))
ax[1].legend(title='mode', loc='upper left', bbox_to_anchor=(1.0, 1.02), handletextpad=0.2)
fig.tight_layout()
fin(fig, a.out, 'ex01_linear_chain.pdf')
