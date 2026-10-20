"""Write matlab/sod_reference.mat: a 10-storey shear frame, one random-forced record and one free
decay, plus the Python toolbox outputs that matlab/test_sod.m reproduces. Run from this folder."""
import os, sys, numpy as np
from scipy.io import savemat
from scipy.signal import butter, sosfiltfilt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..'))
import sodtoolbox as st

n = 10; k = np.linspace(1.0, 0.55, n) * (2 * np.pi) ** 2 * n; M = np.eye(n); K = np.zeros((n, n))
for i in range(n):
    K[i, i] += k[i]
    if i + 1 < n: K[i, i] += k[i + 1]; K[i, i + 1] -= k[i + 1]; K[i + 1, i] -= k[i + 1]
w = np.sqrt(np.linalg.eigvalsh(K)); a, b = np.linalg.solve([[1 / (2 * w[0]), w[0] / 2], [1 / (2 * w[3]), w[3] / 2]], [0.02, 0.02])
C = a * M + b * K; B = -M @ np.ones((n, 1))
A, Bs = st.state_matrices(M, K, C, B)
dt = 0.005; N = 8000; rng = np.random.default_rng(7)
u = sosfiltfilt(butter(6, 60.0 / (np.pi / dt), output='sos'), rng.standard_normal((N, 1)), axis=0)
Y = st.simulate_state(A, Bs, u, dt); Yd = Y @ A.T + u @ Bs.T; X, V, Acc = Y[:, :n], Y[:, n:], Yd[:, n:]
y0 = np.r_[np.linalg.solve(K, B[:, 0]), np.zeros(n)]
Yf = st.simulate_state(A, Bs, np.zeros((4 * N, 1)), dt, y0); Yfd = Yf @ A.T

ref = dict(M=M, K=K, C=C, B=B, A=A, dt=dt, Y=Y, Yd=Yd, Yf=Yf)      # X, V, Acc, Yfd are rebuilt in test_sod.m
R = st.sod(X, V); ref.update(sod_mu=R.mu, sod_Psi=R.Psi, sod_Phi=R.Phi)
R = st.sod(X, V, scale=True); ref.update(sods_mu=R.mu, sods_Psi=R.Psi)
R = st.sod(X, dt=dt, deriv='central', h=2, lag=5, center=True, tol=None); ref.update(sodl_mu=R.mu, sodl_Psi=R.Psi)
R = st.sod(X, dt=dt, deriv='gradient', center=True, tol=None); ref.update(sodg_mu=R.mu, sodg_Psi=R.Psi)
Rs = st.state_sod(Y, Yd); ref.update(ss_mu=Rs.mu, ss_Psi=Rs.Psi, ss_Phi=Rs.Phi)
Sx, Sv = st.covariances(X[:400], V[:400]); Bd = st.boundary_term(X[:400], V[:400], 399 * dt)
cw, cP = st.csod(Sx, Sv, Bd); ref.update(csod_w=cw, csod_P=cP)
ww, wpsi, wm = st.sod_window(X[:802], dt, boundary=True); ref.update(win_w=ww, win_psi=wpsi, win_mode=wm)
tc, om, md = st.sod_track(X, dt, 600, 300, boundary=False); ref.update(trk_t=tc, trk_om=om, trk_modes=md)
om_, F = st.dct_spectrum(Rs.Q[:, :8], dt); pk, rel = st.peak_and_width(om_, F); ref.update(dct_om=om_, dct_F=F, pk=pk, rel=rel)
Rc = st.sod(X, V, scale=True); Qc = Rc.Q
idx, _, _ = st.select_config(Qc, Rc.mu, dt, 150.0, Phi=Rc.Phi); ref.update(sel_idx=idx + 1)
Q = Rs.Q; pl = st.pair_state(Q, dt, 150.0, Phi=Rs.Phi); ref.update(planes=np.array([[p[0] + 1, p[1] + 1, p[2], p[3]] for p in pl]))
Rf = st.sod(Yf, Yfd, scale=True); Qf, Qfd = Rf.Q, Yfd @ Rf.Psi
pf = st.pair_state(Qf, dt, 150.0, Phi=Rf.Phi); a0, b0 = pf[0][:2]
L = st.fit_L(Qf[:, [a0, b0]], Qfd[:, [a0, b0]]); s, z = st.plane_eigs(L)
ref.update(fplanes=np.array([[p[0] + 1, p[1] + 1, p[2], p[3]] for p in pf]), fit_L=L, plane_s=s, plane_zeta=z,
           plane_ident=st.plane_identity(Qf[:, [a0, b0]], dt))
ref.update(lnm=st.lnm(M, K, 4), mta=st.mta(M, K, B, 4), pod=st.pod(X, 4)[0], pov=st.pod(X, 4)[1])
Pm, wm_ = st.sod_modes(X, V, 4); ref.update(sm_Phi=Pm, sm_w=wm_)
Anl = Acc - 30.0 * X ** 3                          # a nonlinear-looking acceleration record (data-only test)
for kind in 'akx': ref[f'res_{kind}'] = st.residual_basis(X, V, Anl, M, K, 2, 4, kind)
Mr, Kr, Cr, Br = st.galerkin(M, K, C, B, Pm); ref.update(gal_K=Kr, mac12=st.mac(Pm[:, 0], st.lnm(M, K, 1)[:, 0]))
out = os.path.join(HERE, '..', '..', 'matlab', 'sod_reference.mat')
savemat(out, {kk: (np.asarray(v) if np.ndim(v) else v) for kk, v in ref.items()}, do_compression=True)
print('wrote', out, os.path.getsize(out) // 1024, 'kB;', len(pl), 'planes;', len(idx), 'config modes;', 'zeta', z)
