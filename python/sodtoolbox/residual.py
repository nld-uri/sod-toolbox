"""Linear-core-plus-residual bases (SOD + R) for nonlinear ROMs.

Given a training record X, V, A (rows = time) and the order r1 of the linear core, fit qdd = q Ka + qd Ca on the r1
smoothest SOD coordinates (least squares), form the equation error e = A - (q Ka + qd Ca) Phi1^T in physical
coordinates, and take its leading directions:
    'a' : POD of e (acceleration-level residual; data only)
    'k' : POD of K^-1 M e (quasi-static displacement driven by the residual force; needs K)
    'f' : POD of e low-passed at fc (data only)
    'x' : POD of the displacement complement X - q Phi1^T (data only)
The basis is [Phi1, R] M-orthonormalized. In our nonlinear-beam tests 'k' was the useful one; 'a' and 'f' were not.
"""
import numpy as np
from .rom import sod_modes, pod, mortho

__all__ = ['linear_core', 'residual_directions', 'residual_basis', 'residual_diagnostics']


def linear_core(X, V, A, r1, tol=1e-10):
    """Phi1 (r1 smooth modes), equation error e, fitted coefficients [Ka; Ca], regressors G = [q qd]"""
    Phi1, _ = sod_modes(X, V, r1, tol)
    Pp = np.linalg.pinv(Phi1); q, qd, qdd = X @ Pp.T, V @ Pp.T, A @ Pp.T
    G = np.c_[q, qd]; coef, *_ = np.linalg.lstsq(G, qdd, rcond=None)
    return Phi1, A - (G @ coef) @ Phi1.T, coef, G


def _lowpass(Z, dt, fc):
    from scipy.signal import butter, sosfiltfilt
    return sosfiltfilt(butter(6, fc, fs=1 / dt, output='sos'), Z, axis=0)


def residual_directions(X, V, A, M, K, r1, kind, dt=None, fc=None):
    Phi1, e, _, _ = linear_core(X, V, A, r1)
    if kind == 'a':
        Z = e
    elif kind == 'k':
        Z = np.linalg.solve(K, M @ e.T).T
    elif kind == 'f':
        Z = _lowpass(e, dt, fc)
    elif kind == 'x':
        Z = X - (X @ np.linalg.pinv(Phi1).T) @ Phi1.T
    else:
        raise ValueError("kind must be 'a', 'k', 'f' or 'x'")
    return Phi1, e, Z


def residual_basis(X, V, A, M, K, r1, r, kind='k', dt=None, fc=None):
    """r-column basis [Phi1, R], M-orthonormal; R from the residual of the given kind"""
    Phi1, e, Z = residual_directions(X, V, A, M, K, r1, kind, dt, fc)
    R, _ = pod(Z, r - r1 + 2)
    return mortho(np.c_[Phi1, R], M)[:, :r]


def residual_diagnostics(X, V, A, M, K, r1, refs, dt=None, fc=None):
    """residual share, orthogonality check of the normal equations, singular-value decay of each residual kind,
    and mass-metric angles (degrees) of the first residual direction to reference vectors (dict name -> vector)"""
    Phi1, e, coef, G = linear_core(X, V, A, r1); out = {}
    out['share'] = np.sum(e ** 2) / np.sum(A ** 2)
    Pp = np.linalg.pinv(Phi1)
    out['orth'] = np.abs(G.T @ e @ Pp.T).max() / (np.linalg.norm(G) * np.linalg.norm(e @ Pp.T) + 1e-300)
    for kind in ('a', 'k', 'f', 'x'):
        _, _, Z = residual_directions(X, V, A, M, K, r1, kind, dt, fc)
        R, w = pod(Z, 6); out[f'sv_{kind}'] = np.sqrt(np.maximum(w[:6], 0) / w[0])
        d = mortho(np.c_[Phi1, R[:, :1]], M)[:, -1]

        def ang(v):
            b = mortho(np.c_[Phi1, v[:, None]], M)[:, -1]
            return np.degrees(np.arccos(min(1, abs(d @ M @ b))))
        out[f'ang_{kind}'] = {k: ang(v) for k, v in refs.items()}
    return out
