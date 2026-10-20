"""Sliding-window SOD: frequency and mode of the smoothest coordinate along a nonstationary record (e.g. the
backbone of a decaying nonlinear normal mode). Window length is in samples; boundary=True applies the symmetric
boundary correction Sigma_v - sym(B), which removes the leading finite-window bias for undamped motion."""
import numpy as np
from scipy.linalg import eigh
from .core import boundary_term

__all__ = ['sod_window', 'sod_track']


def sod_window(Xw, dt, boundary=False):
    """SOD of one window by central differences: returns (omega_1, psi_1, smooth mode 1)."""
    xc = Xw[1:-1]
    vd = (Xw[2:] - Xw[:-2]) / (2 * dt)
    Nw = len(xc)
    Sx, Sv = xc.T @ xc / Nw, vd.T @ vd / Nw
    if boundary:
        Bd = boundary_term(xc, vd, (Nw - 1) * dt)
        Sv = Sv - (Bd + Bd.T) / 2
    mu, P = eigh(Sv, Sx)
    return np.sqrt(max(mu[0], 0.0)), P[:, 0], np.linalg.inv(P).T[:, 0]


def sod_track(X, dt, nwin, step=None, boundary=False):
    """Slide a window of nwin samples by step (default nwin // 2). Returns window centre times (from t = 0),
    frequencies of the smoothest coordinate, and its smooth modes (one row per window)."""
    step = step or max(1, nwin // 2)
    tc, om, modes = [], [], []
    for i0 in range(0, len(X) - nwin - 1, step):
        w, _, m = sod_window(X[i0:i0 + nwin + 2], dt, boundary)
        tc.append((i0 + 1 + nwin / 2) * dt)
        om.append(w)
        modes.append(m)
    return np.array(tc), np.array(om), np.array(modes)
