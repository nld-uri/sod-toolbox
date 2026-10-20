"""Self-contained tests of the SOD Toolbox (no paper code needed). Run: python3 test_sod.py
Each check states the property it tests and fails if the property is violated."""
import os, sys, numpy as np
from scipy.linalg import eigh
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import sodtoolbox as st

rows = []
def check(name, dev, tol):
    ok = dev <= tol; rows.append(ok); print(f"{'PASS' if ok else 'FAIL'}  {name:58s} {dev:.1e}  (tol {tol:.0e})")

def chain(n=4, zeta=0.0):
    M = np.diag(np.linspace(1.0, 2.0, n)); k = np.linspace(3.0, 1.0, n + 1); K = np.zeros((n, n))
    for i in range(n):
        K[i, i] = k[i] + k[i + 1]
        if i + 1 < n: K[i, i + 1] = K[i + 1, i] = -k[i + 1]
    w2, P = eigh(K, M); w = np.sqrt(w2)
    a, b = np.linalg.solve([[1 / (2 * w[0]), w[0] / 2], [1 / (2 * w[-1]), w[-1] / 2]], [zeta, zeta])
    return M, K, a * M + b * K, w, P

# 1. Exact covariances of uncorrelated modes: SOVs are 1/omega^2 and covectors are M-weighted LNMs.
M, K, C, w, P = chain()
amp = np.array([1.0, 0.7, 0.4, 0.2])
Sx = P @ np.diag(amp ** 2 / 2) @ P.T; Sv = P @ np.diag(amp ** 2 * w ** 2 / 2) @ P.T
mu, Psi = st.sod_pencil(Sx, Sv)
check('exact covariances: mu = omega^2', np.max(np.abs(mu / w ** 2 - 1)), 1e-12)
MP = M @ P
check('exact covariances: psi parallel to M phi', max(1 - abs(a @ b) / np.linalg.norm(a) / np.linalg.norm(b) for a, b in zip(Psi.T, MP.T)), 1e-12)
R = st.SODResult(mu, np.sqrt(mu), Psi, Sx @ Psi, np.zeros((1, 4)))
check('smooth modes Sigma_x psi parallel to LNMs', max(1 - abs(a @ b) / np.linalg.norm(a) / np.linalg.norm(b) for a, b in zip(R.Phi.T, P.T)), 1e-12)

# 2. Sampled undamped free response, long record: SOD frequencies approach the natural frequencies.
dt = 0.01; N = 400000; t = np.arange(N) * dt; rng = np.random.default_rng(0); ph = rng.uniform(0, 2 * np.pi, 4)
Q = amp * np.cos(np.outer(t, w) + ph); Qd = -amp * w * np.sin(np.outer(t, w) + ph)
X, V = Q @ P.T, Qd @ P.T
R = st.sod(X, V)
check('sampled record (exact V): SOD frequencies vs omega', np.max(np.abs(R.freq / w - 1)), 1e-3)
R2 = st.sod(X, dt=dt)
check('central differences: freq = sin(omega dt)/dt bias', np.max(np.abs(R2.freq / (np.sin(w * dt) / dt) - 1)), 1e-3)
check('psi^T Sigma_x psi = I', np.max(np.abs(R.Psi.T @ (X.T @ X / N) @ R.Psi - np.eye(4))), 1e-10)
check('Psi^T Phi = I', np.max(np.abs(R.Psi.T @ R.Phi - np.eye(4))), 1e-10)

# 3. Recognition by frequency, never by smoothness: all four modes kept, ordered by peak frequency.
idx, pk, ws = st.select_config(R.Q, R.mu, dt, 10.0, Phi=R.Phi)
check('select_config keeps 4 modes in frequency order', 0.0 if list(idx) == [0, 1, 2, 3] else 1.0, 0)

# 4. Damped free decay: state SOD planes, plane-wise damping fit, damping identity.
M, K, C, w, P = chain(zeta=0.02)
A, Bs = st.state_matrices(M, K, C, np.zeros((4, 1)))
y0 = np.r_[P @ np.ones(4), np.zeros(4)]; dt = 0.005; N = 120000
Y = st.simulate_state(A, Bs, np.zeros((N, 1)), dt, y0); Yd = Y @ A.T
s_true = np.linalg.eigvals(A); s_true = np.sort_complex(s_true[s_true.imag > 0])
Rs = st.state_sod(Y, Yd); D = st.decay_damping(Rs, Yd, dt, 10.0)
check('decay: four planes recognised by circularity', abs(len(D['plane']) - 4), 0)
zt = -s_true.real / abs(s_true)
ze = np.array(sorted([(abs(p[0]), p[1]) for p in D['plane']]))[:, 1]
# per-plane fits are approximate: a tilted plane leaks its neighbour's dynamics (errors up to 0.17 were seen on cantilever decays)
check('decay: plane-wise zeta vs exact (approximate, relative)', np.max(np.abs(ze / zt - 1)), 0.1)
se = np.sort_complex(np.array(D['sub12']))
check('decay: subspace fit eigenvalues vs exact (relative)', np.max(np.abs(se - s_true) / abs(s_true)), 1e-6)
idn = np.array(D['identity'])
check('decay: tr L = end-state form (well-sampled planes)', np.max(np.abs(idn[:, 1] / idn[:, 0] - 1)), 1e-2)

# 5. Residual basis: the least-squares core leaves an error orthogonal to its regressors.
X, V, Acc = Y[:, :4], Y[:, 4:], Yd[:, 4:] - 5.0 * Y[:, :4] ** 3
Phi1, e, coef, G = st.linear_core(X, V, Acc, 2)
Pp = np.linalg.pinv(Phi1)
check('linear core: normal equations', np.abs(G.T @ (e @ Pp.T)).max() / (np.linalg.norm(G) * np.linalg.norm(e @ Pp.T)), 1e-10)
T = st.residual_basis(X, V, Acc, M, K, 2, 4, 'k')
check('residual basis is M-orthonormal', np.max(np.abs(T.T @ M @ T - np.eye(4))), 1e-10)

# 6. Exact simulation agrees with an explicit ZOH loop.
Ad, Bd = st.zoh(A, Bs, dt); y = y0.copy(); Yl = np.zeros((200, 8))
for k in range(200): Yl[k] = y; y = Ad @ y
check('simulate_state vs explicit loop', np.abs(Y[:200] - Yl).max() / np.abs(Yl).max(), 1e-12)

print(f"\n{sum(rows)}/{len(rows)} checks passed"); sys.exit(0 if all(rows) else 1)
