"""Example 4. Damped free decay: state SOD planes, recognized by circularity, and damping from each plane.

4-DOF chain with Rayleigh damping (2 % at modes 1 and 4), free decay from equal modal energies, exact kinematics.
Configuration SOD of a damped decay no longer returns 1/omega^2 exactly. State SOD on y = (x, xdot) returns pairs
of coordinates that each span one modal plane; pair_state finds them by the one-sidedness of q_a + i q_b.
The least-squares plane fit L = <qdot q^T><q q^T>^-1 gives s and zeta per plane; the same fit on all coordinates
at once (the subspace fit) returns the exact eigenvalues.
"""
from _common import setup, fin
import numpy as np
import matplotlib.pyplot as plt
import sodtoolbox as st
from sodtoolbox import systems as sy, plotting as pl

a = setup(__doc__)
M, K, C = sy.chain(4, zeta=0.02)
s_true = sy.complex_eigs(M, K, C); z_true = -s_true.real / abs(s_true)
y0 = sy.equal_energy(M, K, rng=np.random.default_rng(1))
dt, N = 0.01, 60000
t, X, V, Acc = sy.free_decay(M, K, C, y0, dt, N)
Rc = st.sod(X, V)
print('configuration SOD freq / |s| - 1:', ' '.join(f'{x:+.1e}' for x in Rc.freq / abs(s_true) - 1))
Y, Yd = np.c_[X, V], np.c_[V, Acc]
Rs = st.state_sod(Y, Yd)
D = st.decay_damping(Rs, Yd, dt, 10.0, ks=(8,))
planes = sorted(D['plane'], key=lambda p: abs(p[0]))
z_pl = np.array([p[1] for p in planes])
sub = np.array(D['sub8']); sub = sub[np.argsort(abs(sub))]; z_sub = -sub.real / abs(sub)
for i in range(4):
    print(f'mode {i + 1}: zeta exact {z_true[i]:.5f}  per-plane {z_pl[i]:.5f} ({z_pl[i] / z_true[i] - 1:+.1e})'
          f'  subspace {z_sub[i]:.5f} ({z_sub[i] / z_true[i] - 1:+.1e})')
print('planes (a, b, circularity):', [(p[2][0], p[2][1]) for p in planes],
      [round(r[3], 3) for r in st.pair_state(Rs.Q, dt, 10.0, Phi=Rs.Phi)])

fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw={'width_ratios': [1, 1.25]})
for j, p in enumerate(planes[:2]):
    a_, b_ = p[2]; q = Rs.Q[:, [a_, b_]]; q = q / np.abs(q).max()
    ax[0].plot(q[:, 0] + 2.3 * j, q[:, 1], lw=0.25, color=pl.color(j))
    ax[0].text(2.3 * j, 1.15, f'mode {j + 1} plane', ha='center', fontsize=8)
ax[0].set_aspect('equal'); ax[0].set_xticks([]); ax[0].set_yticks([]); ax[0].set_xlabel(r'$q_a$'); ax[0].set_ylabel(r'$q_b$')
k = np.arange(1, 5)
ax[1].semilogy(k, np.abs(z_pl / z_true - 1), 'o', color=pl.color(0), ms=6, label='per-plane fit')
ax[1].semilogy(k, np.maximum(np.abs(z_sub / z_true - 1), 1e-16), 's', color=pl.color(1), ms=6, label='subspace fit')
ax[1].set_xticks(k); ax[1].set_xlabel('mode'); ax[1].set_ylabel(r'$|\hat\zeta/\zeta-1|$'); ax[1].set_ylim(1e-16, 1e-1)
ax[1].legend(fontsize=8, loc='center right')
fig.tight_layout()
fin(fig, a.out, 'ex04_damped_decay.pdf')
