"""Numbers quoted in the manual for the animation cases (backbone tracking, trend extraction, geometry).
    python report_numbers.py > report_numbers.txt"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import sodtoolbox as st
from sodtoolbox import systems as sy

def ang(a, b):
    return np.degrees(np.arccos(min(1, abs(a @ b) / np.linalg.norm(a) / np.linalg.norm(b))))


# geometry: 2-DOF chain, extremes of the roughness quotient
M, K, Cd = sy.chain(2, m=[1, 3], k=[1, 2]); w, P = sy.lnm_modes(M, K)
y0 = sy.equal_energy(M, K, rng=np.random.default_rng(4))
t, X, V, _ = sy.free_decay(M, K, Cd, y0, 0.05, 40000); R = st.sod(X, V)
print('geometry: omega', np.round(w, 5), ' SOD freq', np.round(R.freq, 5),
      ' angles (deg): LNM1-LNM2 %.1f, POD2-LNM2 %.1f, SOD2-LNM2 %.3f' % (ang(P[:, 0], P[:, 1]), ang(st.pod(X, 2)[0][:, 1], P[:, 1]),
                                                                         ang(R.Phi[:, 1], P[:, 1])))

# backbone: hardening 2-DOF decay, windowed SOD against zero crossings
M, K, Cd = sy.chain(2, m=[1, 1], k=[1, 1], zeta=0.005); w, P = sy.lnm_modes(M, K)
y0 = np.r_[P[:, 0] / np.abs(P[:, 0]).max() * 2.5, 0, 0]; dt = 0.05
t, X, V, A = sy.duffing_decay(M, K, Cd, 0.5, y0, dt, int(1500 / dt))
x = X[:, 0]; zc = np.nonzero((x[:-1] < 0) & (x[1:] >= 0))[0]; tz = t[zc] - x[zc] / (x[zc + 1] - x[zc]) * dt
fz = 2 * np.pi / np.diff(tz); tm = (tz[1:] + tz[:-1]) / 2
print('backbone: zero-crossing frequency first/last %.4f / %.4f, linear %.4f' % (fz[0], fz[-1], w[0]))
for per in (6, 12):
    for b in (False, True):
        nwin = int(per * 2 * np.pi / w[0] / dt); tc, om, _ = st.sod_track(X, dt, nwin, nwin // 8, boundary=b)
        e = om / np.interp(tc, tm, fz) - 1
        print(f'  window {per:2d} periods, boundary={b!s:5s}: median |dev| {np.median(np.abs(e)):.1e}, max {np.abs(e).max():.1e}')

# trend: SOD (step 15, lag 31) against the oracle unmixing
rng = np.random.default_rng(3); N = 4000; dt = 1 / N; t = (np.arange(N) + 0.5) * dt
k = np.array([1, 6, 80, 160, 300, 450]); amp = np.array([1.0, 0.5, 3.0, 2.0, 1.5, 1.0])
S = amp[:, None] * np.cos(np.pi * k[:, None] * t)
Q1, _ = np.linalg.qr(rng.normal(size=(6, 6))); Q2, _ = np.linalg.qr(rng.normal(size=(6, 6)))
Amix = Q1 @ np.diag(np.linspace(1, 3, 6)) @ Q2; X0 = S.T @ Amix.T; E = rng.standard_normal(X0.shape)
drift = S[0] - S[0].mean(); Ai = np.linalg.inv(Amix)
for lv in (0, 0.003, 0.01, 0.03):
    Xn = X0 + lv * X0.std(0) * E
    R = st.sod(Xn, dt=dt, h=15, lag=31, center=True, tol=None)
    q = R.Q[:, 0] - R.Q[:, 0].mean(); d = drift[15:-15]; q = q * (q @ d) / (q @ q)
    o = (Xn - Xn.mean(0)) @ Ai[0]
    Xc = Xn - Xn.mean(0); p = Xc @ st.pod(Xc, 1)[0][:, 0]
    print(f'trend: noise {100 * lv:.1f}%: drift error SOD {np.linalg.norm(q - d) / np.linalg.norm(d):.1e}, '
          f'oracle {np.linalg.norm(o - drift) / np.linalg.norm(drift):.1e}, |corr(POD 1, drift)| {abs(np.corrcoef(p, drift)[0, 1]):.3f}')
