"""Example 3. Sensor noise and recognition by frequency, never by smoothness.

6-DOF chain (Rayleigh 2 %) driven at the last mass by band-limited noise (cutoff 2.6 rad/s, between modes 5 and 6),
displacements with 1 % white sensor noise. Central differences turn the noise into roughness: every SOD frequency
is pulled up and the coordinates no longer agree with their own spectral peaks, so select_config keeps almost
nothing. Lagged symmetric covariances (step h = 2, lag 5 > 2h) remove the white noise from both covariances; the
SOD frequencies return to modes 1-4, which are kept, ordered by their spectral peaks. Mode 5, near the forcing
cutoff and weakly excited, mixes with noise and is rejected; so are the noise directions.
"""
from _common import setup, fin
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
import sodtoolbox as st
from sodtoolbox import systems as sy, plotting as pl

a = setup(__doc__)
n = 6; M, K, C = sy.chain(n, zeta=0.02); w, _ = sy.lnm_modes(M, K)
A, Bs = st.state_matrices(M, K, C, np.eye(n)[:, [n - 1]])
dt, N = 0.05, 80000; rng = np.random.default_rng(3)
u = sosfiltfilt(butter(4, 2.6 / (np.pi / dt), output='sos'), rng.standard_normal((N, 1)), axis=0)
X = st.simulate_state(A, Bs, u, dt)[:, :n]
Xn = X + 0.01 * X.std(0) * rng.standard_normal(X.shape)

out = {}
for name, h, lag in (('central', 1, 0), ('lagged', 2, 5)):
    R = st.sod(Xn, dt=dt, h=h, lag=lag, scale=True)
    wc = st.cd_correct(np.nan_to_num(R.freq, nan=0.0), h * dt)          # bias of a step-h difference
    idx, pk, _ = st.select_config(R.Q, wc ** 2, dt, 10.0, Phi=R.Phi)
    out[name] = (R, wc, idx, pk)
    print(f'{name:8s} SOD freq', np.round(wc, 3), ' kept', list(idx), ' peaks', np.round(pk[idx], 3))
print('true omega        ', np.round(w, 3))

fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7))
for j, (name, mk) in enumerate((('central', 'o'), ('lagged', 's'))):
    R, wc, idx, pk = out[name]
    k = np.arange(1, n + 1); kept = np.isin(np.arange(n), idx)
    ax[0].semilogy(k[~kept], wc[~kept], mk, mfc='none', color=pl.color(j), ms=6)
    ax[0].semilogy(k[kept], wc[kept], mk, color=pl.color(j), ms=6, label=f'{name}')
ax[0].semilogy(np.arange(1, n + 1), w, '_', color='k', ms=14, mew=1.5, label=r'$\omega_i$')
ax[0].set_xlabel('SOD coordinate (smoothest first)'); ax[0].set_ylabel(r'SOD frequency (rad/s)')
ax[0].legend(fontsize=8, loc='upper left'); ax[0].set_title('filled: kept by \\texttt{select\\_config}', fontsize=8.5)
R, wc, idx, pk = out['lagged']
om, F = st.dct_spectrum(R.Q, dt)
for i in range(n):
    s = F[:, i] / F[:, i].max()
    ax[1].semilogy(om, s, color=pl.color(i) if i in idx else pl.ref(), lw=0.9 if i in idx else 0.6)
for x in w:
    ax[1].axvline(x, color=pl.ref(), lw=0.5, ls=':')
ax[1].set_xlim(0, 3.2); ax[1].set_ylim(1e-5, 2); ax[1].set_xlabel(r'$\omega$ (rad/s)'); ax[1].set_ylabel('DCT power (normalized)')
ax[1].set_title('lagged SOD coordinates; gray: rejected', fontsize=8.5)
fig.tight_layout()
fin(fig, a.out, 'ex03_noise_recognition.pdf')
