"""Self-contained tests of the SMD module (smooth mode decomposition, the spatial dual of SOD).
Run: python3 test_smd.py. Each check states the property it tests and fails if the property is violated.
Failure controls (marked FC) must show a large error: they prove that the checks can fail."""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import sodtoolbox as st

rows = []
def check(name, dev, tol, must_fail=False):
    ok = (dev > tol) if must_fail else (dev <= tol)
    rows.append(ok)
    rel = '>' if must_fail else '<='
    print(f"{'PASS' if ok else 'FAIL'}  {name:62s} {dev:.2e}  (need {rel} {tol:.0e})")

def mac(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return (a @ b) ** 2 / ((a @ a) * (b @ b))

rng = np.random.default_rng(1)
def coords(nt, r, corr=False):
    """Snapshot amplitudes (nt x r): zero-mean, orthogonal columns, or deliberately correlated."""
    Q, _ = np.linalg.qr(rng.standard_normal((nt, r)) - 0.0)
    Q -= Q.mean(0)
    A = Q * np.geomspace(1, 0.2, r)
    if corr:
        A = A @ (np.eye(r) + 0.6 * np.triu(np.ones((r, r)), 1))
    return A

# ---------------------------------------------------------------- 1. string, gradient operator
x = np.linspace(0, 1, 201); g = st.make_grid(x=x)
J = np.arange(1, 5); S = np.sin(np.outer(x, J * np.pi))              # (n, r)
A = coords(60, 4); W = A @ S.T
R = st.smd(W, g, 'grad', 'fd')
check('string, grad, fd: k_j = j pi', np.max(np.abs(R.k / (J * np.pi) - 1)), 1e-3)
check('string: modes match sin(j pi x) (1 - MAC)', max(1 - mac(R.Phi[:, i], S[:, i]) for i in range(4)), 1e-6)
d = st.smd_derivatives(W - W.mean(0), g, 'grad', 'fd')
phi = R.Phi[:, 0]; dphi = (d.G[0].T @ R.P[:, 0])
q = np.sum(g.w * dphi ** 2) / np.sum(g.w * phi ** 2)
check('mu equals the Rayleigh quotient ||D phi||^2/||phi||^2', abs(q / R.mu[0] - 1), 1e-10)
check('coordinates reconstruct the field: W = C Phi^T', np.max(np.abs(R.C @ R.Phi.T - (W - W.mean(0)))), 1e-10)

# ---------------------------------------------------------------- 2. sampling independence and correlation independence
perm = rng.permutation(len(W))
R2 = st.smd(W[perm], g, 'grad', 'fd')
check('snapshot order is irrelevant (shuffled)', np.max(np.abs(R2.mu / R.mu - 1)), 1e-10)
Wc = coords(60, 4, corr=True) @ S.T
R3 = st.smd(Wc, g, 'grad', 'fd')
check('correlated amplitudes, same row space: same SMD values', np.max(np.abs(R3.mu / R.mu - 1)), 1e-8)

# ---------------------------------------------------------------- 3. cantilever beam, second-derivative operator
L = 1.0; bl = np.array([1.8751040687, 4.6940911330, 7.8547574382])
xb = np.linspace(0, L, 401); gb = st.make_grid(x=xb)
def cant(b, xx):
    s = (np.cosh(b) + np.cos(b)) / (np.sinh(b) + np.sin(b))
    return np.cosh(b * xx) - np.cos(b * xx) - s * (np.sinh(b * xx) - np.sin(b * xx))
B = np.column_stack([cant(b, xb) for b in bl])
Wb = coords(40, 3) @ B.T
Rb = st.smd(Wb, gb, 'second', 'fd')
check('cantilever, second, fd: k_j = beta_j', np.max(np.abs(Rb.k / bl - 1)), 1e-3)
check('cantilever: modes match (1 - MAC)', max(1 - mac(Rb.Phi[:, i], B[:, i]) for i in range(3)), 1e-6)
Rg = st.smd(Wb, gb, 'grad', 'fd')
check('FC cantilever with mismatched operator (grad): mode 2 error', 1 - mac(Rg.Phi[:, 1], B[:, 1]), 1e-3, must_fail=True)
sub = (xb >= 0.2) & (xb <= 0.8)
Rp = st.smd(Wb[:, sub], st.make_grid(x=xb[sub]), 'second', 'fd')
check('FC partial field of view (0.2L-0.8L): k bias from boundary terms', np.max(np.abs(Rp.k / bl - 1)), 1e-2, must_fail=True)

# ---------------------------------------------------------------- 4. membrane on a grid
a, bm = 1.0, 0.7
xm, ym = np.linspace(0, a, 81), np.linspace(0, bm, 57); gm = st.make_grid(x=xm, y=ym)
mn = [(1, 1), (2, 1), (1, 2), (3, 1)]
M = np.stack([np.outer(np.sin(n * np.pi * ym / bm), np.sin(m * np.pi * xm / a)).ravel() for m, n in mn], 1)
km = np.array([np.pi * np.hypot(m / a, n / bm) for m, n in mn]); o = np.argsort(km)
Wm = (coords(50, 4) @ M.T).reshape(50, len(ym), len(xm))
Rm = st.smd(Wm, gm, 'grad', 'fd')
check('membrane, grad, fd: k = pi sqrt((m/a)^2 + (n/b)^2)', np.max(np.abs(Rm.k / km[o] - 1)), 2e-3)
check('membrane: modes match (1 - MAC)', max(1 - mac(Rm.Phi[:, i], M[:, o[i]]) for i in range(4)), 1e-5)
Rl = st.smd(Wm, gm, 'laplacian', 'fd')
check('membrane, laplacian: k = same wavenumbers', np.max(np.abs(Rl.k / km[o] - 1)), 2e-3)

# ---------------------------------------------------------------- 5. noise: regularized derivatives
Wn = Wb + 0.005 * np.std(Wb) * rng.standard_normal(Wb.shape)
Rf = st.smd(Wn, gb, 'second', 'fd', rank=3)
check('FC beam, 0.5% noise, plain fd second derivative: k error', np.max(np.abs(Rf.k / bl - 1)), 5e-2, must_fail=True)
for meth, (f0, f1) in (('whittaker', (3.0, 6.0)), ('savgol', (8.0, 6.0))):
    K = st.smd_kc_scan(Wn, gb, 'second', meth, [f0 * bl[-1], f1 * bl[-1]], rank=3)
    check(f'beam, 0.5% noise, {meth}, kc = {f0:g} beta_3: k error', np.max(np.abs(K[0] / bl - 1)), 1e-2)
    check(f'beam, 0.5% noise, {meth}: kc {f0:g} -> {f1:g} beta_3 moves k by', np.max(np.abs(K[1] / K[0] - 1)), 1e-2)
K = st.smd_kc_scan(Wn, gb, 'second', 'whittaker', [1.5 * bl[-1]], rank=3)
check('FC whittaker cutoff too low (kc = 1.5 beta_3): attenuation bias', np.max(np.abs(K[0] / bl - 1)), 1e-2, must_fail=True)

# ---------------------------------------------------------------- 6. masked grid (hole), Whittaker fills, fd skips stencils
hole = np.hypot(*np.meshgrid(xm - 0.6, ym - 0.35)) < 0.08
gh = st.make_grid(x=xm, y=ym, mask=~hole)
Rw20 = st.smd(Wm, gm, 'grad', 'whittaker', kc=20 * km.max())
check('complete membrane: whittaker -> fd as kc grows (kc = 20 k_max)', np.max(np.abs(Rw20.k / Rm.k - 1)), 1e-4)
Rhw = st.smd(Wm, gh, 'grad', 'whittaker', kc=6 * km.max())
check('FC hole in the field: sines no longer orthogonal on the domain, k bias', np.max(np.abs(Rhw.k / km[o] - 1)), 1e-2, must_fail=True)
dW = st.smd_derivatives(Wm, gh, 'grad', 'whittaker', kc=6 * km.max())
ex = (np.outer(np.sin(np.pi * ym / bm), np.pi / a * np.cos(np.pi * xm / a))).ravel()
Wone = np.broadcast_to(M[:, 0].reshape(len(ym), len(xm)), (1, len(ym), len(xm)))
d1 = st.smd_derivatives(Wone, gh, 'grad', 'whittaker', kc=6 * km.max())
v = d1.valid
check('masked: whittaker d/dx of mode (1,1) at measured points (rel max err)',
      np.max(np.abs(d1.G[0][0, v] - ex[v])) / np.max(np.abs(ex)), 1e-2)

print(f'\n{sum(rows)}/{len(rows)} checks passed')
sys.exit(0 if all(rows) else 1)
