"""Write matlab/smd_reference.mat: SMD test fields and the Python results that matlab/test_smd.m reproduces.
Run from this folder: python3 make_smd_reference.py"""
import os, sys, numpy as np
from scipy.io import savemat
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..'))
import sodtoolbox as st

rng = np.random.default_rng(3)
def coords(nt, r):
    Q, _ = np.linalg.qr(rng.standard_normal((nt, r))); Q -= Q.mean(0); return Q * np.geomspace(1, 0.2, r)

bl = np.array([1.8751040687, 4.6940911330, 7.8547574382]); xb = np.linspace(0, 1, 401)
def cant(b, xx):
    s = (np.cosh(b) + np.cos(b)) / (np.sinh(b) + np.sin(b))
    return np.cosh(b * xx) - np.cos(b * xx) - s * (np.sinh(b * xx) - np.sin(b * xx))
Wb = coords(40, 3) @ np.column_stack([cant(b, xb) for b in bl]).T
Wn = Wb + 0.005 * np.std(Wb) * rng.standard_normal(Wb.shape)
xn = np.sort(np.r_[0, 1, rng.uniform(0, 1, 299)])          # non-uniform line
Wu = coords(30, 3) @ np.column_stack([np.sin(j * np.pi * xn) for j in (1, 2, 3)]).T
a, bm = 1.0, 0.7; xm, ym = np.linspace(0, a, 61), np.linspace(0, bm, 43)
M = np.stack([np.outer(np.sin(n * np.pi * ym / bm), np.sin(m * np.pi * xm / a)).ravel() for m, n in [(1, 1), (2, 1), (1, 2), (3, 1)]], 1)
Wm = (coords(40, 4) @ M.T).reshape(40, len(ym), len(xm))
hole = np.hypot(*np.meshgrid(xm - 0.6, ym - 0.35)) < 0.08

xq, wq = np.polynomial.legendre.leggauss(400); xq = (xq + 1) / 2; wq = wq / 2   # tapered pinned-pinned beam
jj = np.arange(1, 61) * np.pi; Sq = np.sin(np.outer(xq, jj))
EIq, rhq = 1 - 0.8 * xq, 1 - 0.6 * xq
w2, Cg = __import__('scipy.linalg', fromlist=['eigh']).eigh((Sq * jj ** 2 * (wq * EIq)[:, None]).T @ (Sq * jj ** 2),
                                                          (Sq * (wq * rhq)[:, None]).T @ Sq)
xt = np.linspace(0, 1, 201); EIt, rht = 1 - 0.8 * xt, 1 - 0.6 * xt
Wt = rng.standard_normal((6, 4)) @ (np.sin(np.outer(xt, jj)) @ Cg[:, :4]).T
gt = st.make_grid(x=xt)
gb, gu, gm, gh = st.make_grid(x=xb), st.make_grid(x=xn), st.make_grid(x=xm, y=ym), st.make_grid(x=xm, y=ym, mask=~hole)
cases = {
    'b_fd':   (Wb, gb, 'second', 'fd', {}),
    'b_wh':   (Wn, gb, 'second', 'whittaker', {'kc': 3 * bl[-1], 'rank': 3}),
    'b_sg':   (Wn, gb, 'second', 'savgol', {'kc': 8 * bl[-1], 'rank': 3}),
    'u_fd':   (Wu, gu, 'grad', 'fd', {}),
    'u_wh':   (Wu, gu, 'grad', 'whittaker', {'kc': 30.0}),
    'm_fd':   (Wm, gm, 'grad', 'fd', {}),
    'm_lap':  (Wm, gm, 'laplacian', 'fd', {}),
    'm_hes':  (Wm, gm, 'second', 'savgol', {'window': 9}),
    'm_sgg':  (Wm, gm, 'grad', 'savgol', {'kc': 40.0}),
    'h_wh':   (Wm, gh, 'grad', 'whittaker', {'kc': 60.0}),
    'h_fd':   (Wm, gh, 'grad', 'fd', {}),
    'p_fd':   (Wm, gm, 'plate', 'fd', {'nu': 0.3}),
    'p_sg':   (Wm, gm, 'plate', 'savgol', {'window': 9, 'nu': 0.25}),
    'p_wh':   (Wm, gm, 'plate', 'whittaker', {'kc': 60.0, 'nu': 0.3}),
    't_ph':   (Wt, gt, 'second', 'fd', {'density': rht, 'stiffness': EIt, 'center': False}),
}
ref = dict(xb=xb, Wb=Wb, Wn=Wn, xn=xn, Wu=Wu, xm=xm, ym=ym, Wm=Wm, hole=hole, bl=bl, xt=xt, Wt=Wt, EIt=EIt, rht=rht,
           omega_t=np.sqrt(w2[:4]))
for name, (W, g, op, meth, kw) in cases.items():
    rank = kw.pop('rank', None)
    sk = {k: kw.pop(k) for k in ('density', 'stiffness', 'center') if k in kw}
    R = st.smd(W, g, op, meth, rank=rank, **sk, **kw)
    ref[name + '_k'] = R.k; ref[name + '_Phi'] = np.nan_to_num(R.Phi, nan=0.0)
    d = st.smd_derivatives(W - np.nanmean(W, 0) * sk.get('center', True), g, op, meth, **kw)
    ref[name + '_valid'] = d.valid
    if name in ('b_sg', 'u_fd', 'm_sgg', 'h_wh'):           # derivative fields, first five snapshots
        ref[name + '_G1'] = np.nan_to_num(d.G[0][:5], nan=0.0)
    print(f'{name:6s} k = {np.round(R.k, 6)}')
nu_maf, F_maf, _ = st.maf(Wm, gm); ref['maf_nu'] = nu_maf; ref['maf_F'] = F_maf
nu_mh, F_mh, _ = st.maf(Wm, gh); ref['mafh_nu'] = nu_mh; ref['mafh_F'] = np.nan_to_num(F_mh, nan=0.0)
nu_my, F_my, _ = st.maf(Wm, gh, axes='y'); ref['mafy_nu'] = nu_my; ref['mafy_F'] = np.nan_to_num(F_my, nan=0.0)
savemat(os.path.join(HERE, '..', '..', 'matlab', 'smd_reference.mat'), ref, do_compression=True)
print('written matlab/smd_reference.mat')
