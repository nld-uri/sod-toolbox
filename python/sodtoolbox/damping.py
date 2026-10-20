"""Damping from a free decay through SOD coordinates.

The symmetric SOD pencil discards the symmetric part of <ydot y^T>, which on a free decay is the boundary term
(y_T y_T^T - y_0 y_0^T)/2T: the energy lost over the record. The least-squares fit
    L = <qdot q^T> <q q^T>^{-1}
on the coordinates q of one recognised plane recovers its dynamics, with zeta = -tr(L)/2 / sqrt(det L).
Its trace follows from the end states alone (plane_identity), so a record that has decayed is needed.
fit_L on the k smoothest state-SOD coordinates at once is a continuous-time least-squares relative of DMD restricted
to the SOD subspace; it needs no plane recognition.
"""
import numpy as np
from .recognize import pair_state

__all__ = ['fit_L', 'plane_eigs', 'plane_identity', 'zeta_product_rule', 'decay_damping']


def fit_L(Q, Qd):
    """L = <qdot q^T><q q^T>^{-1} (samples x k inputs)."""
    return (Qd.T @ Q) @ np.linalg.inv(Q.T @ Q)


def plane_eigs(L):
    """complex eigenvalue s = sigma + i omega_d of a 2x2 plane fit, and zeta = -sigma/|s|"""
    sig = np.trace(L) / 2
    wn = np.sqrt(max(np.linalg.det(L), 0.0))
    s = complex(sig, np.sqrt(max(wn**2 - sig**2, 0.0)))
    return s, -sig / wn if wn > 0 else np.nan


def plane_identity(Q, dt):
    """tr L from the end states, (q_T^T G^-1 q_T - q_0^T G^-1 q_0)/(2T), G = <q q^T> (sample sums, T = dt).
    Exact for the chord derivative (q_{k+1} - q_k)/dt; with exact qdot and sampled sums the two forms differ by a
    relative O((|s| dt)^2 / zeta), since the integrand oscillates with size |s| about a mean zeta |s|."""
    Gi = np.linalg.inv(Q.T @ Q)
    return (Q[-1] @ Gi @ Q[-1] - Q[0] @ Gi @ Q[0]) / 2 / dt


def zeta_product_rule(mu_a, mu_b, omega_d):
    """|s| = (mu_a mu_b)^{1/4} against the plane rotation rate omega_d: zeta = sqrt(1 - (omega_d/|s|)^2).
    Needs uncorrelated planes; fails on single decays, where the planes are correlated."""
    s = (mu_a * mu_b) ** 0.25
    return np.sqrt(max(1 - (omega_d / s) ** 2, 0.0))


def decay_damping(R, Yd, dt, wmax, ks=(12, 20)):
    """Per-plane and subspace damping estimates from a state-SOD result R (of y) and the record ydot.
    Returns dict: 'plane' -> list of (s, zeta, (a, b)), 'identity' -> list of (tr L, end-state form),
    'sub<k>' -> eigenvalues (upper half plane) of the k-coordinate fit."""
    Q, Qd = R.Q, Yd @ R.Psi
    out = {'plane': [], 'identity': []}
    for a, b, f, c in pair_state(Q, dt, wmax, Phi=R.Phi):
        q, qd = Q[:, [a, b]], Qd[:, [a, b]]
        L = fit_L(q, qd)
        s, z = plane_eigs(L)
        out['plane'].append((s, z, (a, b)))
        out['identity'].append((np.trace(L), plane_identity(q, dt)))
    for k0 in ks:
        k = min(k0, Q.shape[1])
        ev = np.linalg.eigvals(fit_L(Q[:, :k], Qd[:, :k]))
        out[f'sub{k0}'] = list(ev[ev.imag > 0])
    return out
