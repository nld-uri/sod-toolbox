"""Reduced-order models on SOD and baseline bases.

Second-order model  M xdd + C xd + K x = B u,  first-order form  yd = A y + Bs u,  y = (x, xd).

Bases (columns span the reduced subspace):
    lnm(M, K, r)              linear normal modes (mass-normalized), lowest r
    mta(M, K, B, r)           r - m LNMs plus the m static load vectors K^-1 B, M-orthonormalized
    pod(X, r)                 proper orthogonal modes of the record
    sod_modes(X, V, r)        smooth modes from core.sod, unit Euclidean norm, with SOD frequencies
    complex_modes(A, k)       k slowest eigen-planes of A in real form, with dual covectors

ROMs:
    galerkin(M, K, C, B, Phi)                 reduced matrices Phi^T (M, K, C, B) Phi
    rom_config(...)                           simulate the Galerkin ROM, return physical displacements
    rom_state(A, Bs, Psi, Phi, ..., W=...)    state ROM: energy-Galerkin (W = blockdiag(K, M)) or Petrov-Galerkin
Simulation is exact for zero-order-hold input (matrix exponential, diagonalized recursion).
"""
import numpy as np
from scipy.linalg import eigh, expm, eig
from scipy.signal import lfilter
from .core import sod

__all__ = ['state_matrices', 'zoh', 'simulate_state', 'simulate', 'lnm', 'mta', 'pod', 'sod_modes', 'complex_modes',
           'galerkin', 'rom_config', 'rom_state', 'energy_weight', 'merr', 'relerr', 'mac', 'mortho',
           'principal_angles']


# ---------------- model and simulation ----------------
def state_matrices(M, K, C, B):
    """A = [[0, I], [-M^-1 K, -M^-1 C]],  Bs = [0; M^-1 B]"""
    n = len(M); Mi = np.linalg.inv(M)
    A = np.block([[np.zeros((n, n)), np.eye(n)], [-Mi @ K, -Mi @ C]])
    return A, np.vstack([np.zeros_like(B), Mi @ B])


def zoh(A, Bs, dt):
    """exact zero-order-hold discretization: (Ad, Bd)"""
    n = len(A); m = Bs.shape[1]
    E = expm(np.block([[A, Bs], [np.zeros((m, n + m))]]) * dt)
    return E[:n, :n], E[:n, n:]


def simulate_state(A, Bs, u, dt, y0=None, block=16):
    """y_{k+1} = Ad y_k + Bd u_k for k = 0..N-1 (N = len(u)); rows of the output are y_0 .. y_{N-1}.
    Diagonalized recursion (C-speed lfilter); exact up to round-off for diagonalizable Ad."""
    u = np.asarray(u, float)
    if u.ndim == 1:
        u = u[:, None]
    Ad, Bd = zoh(A, Bs, dt); d, V = eig(Ad); Vi = np.linalg.inv(V); N = len(u); n = len(A)
    G = Vi @ Bd
    z0 = np.zeros(n, complex) if y0 is None else Vi @ y0
    Y = np.zeros((N, n))
    for j0 in range(0, n, block):
        js = list(range(j0, min(n, j0 + block))); Zb = np.empty((N, len(js)), complex)
        for c, j in enumerate(js):
            Zb[:, c], _ = lfilter([0, 1], [1, -d[j]], u @ G[j], zi=[z0[j]])
        Y += (Zb @ V[:, js].T).real
    return Y


def simulate(M, K, C, B, u, dt, y0=None):
    """second-order model; returns X, V (N x n)"""
    A, Bs = state_matrices(M, K, C, B); Y = simulate_state(A, Bs, u, dt, y0); n = len(M)
    return Y[:, :n], Y[:, n:]


# ---------------- bases ----------------
def lnm(M, K, r):
    return eigh(K, M)[1][:, :r]


def mta(M, K, B, r):
    """modal truncation augmentation; None if r <= number of load vectors"""
    m = B.shape[1]
    if r <= m:
        return None
    Q = np.c_[lnm(M, K, r - m), np.linalg.solve(K, B)]
    L = np.linalg.cholesky(Q.T @ M @ Q)
    return Q @ np.linalg.inv(L).T


def pod(X, r, tol=1e-10):
    """leading r proper orthogonal modes (uncentred), limited to the data rank (tol=None: no limit); also returns the
    POVs (eigenvalues of X^T X, descending)"""
    w, U = eigh(X.T @ X); U = U[:, ::-1]; w = w[::-1]
    k = r if tol is None else min(r, int(np.sum(w > tol * w.max())))
    return U[:, :k], w


def sod_modes(X, V, r, tol=1e-10):
    """smooth modes (unit Euclidean norm, at most r, within the data subspace) and SOD frequencies"""
    R = sod(X, V, tol=tol)
    k = min(r, R.Phi.shape[1])
    return R.modes_unit()[:, :k], np.sqrt(R.mu[:k])


def complex_modes(A, k):
    """model baseline for state ROMs: the k slowest eigen-planes of A in real form (k even for oscillatory modes).
    Returns covectors Psi and vectors Phi with Psi^T Phi = I."""
    s, Vr = eig(A); s_l, W = eig(A.T)
    o = np.argsort(abs(s)); s = s[o]; Vr = Vr[:, o]
    cols, covs, used, i = [], [], 0, 0
    while used < k and i < len(s):
        lam = s[i]; v = Vr[:, i]; w = W[:, np.argmin(abs(s_l - lam))]; w = w / (w @ v)
        if abs(lam.imag) > 1e-12:
            if lam.imag < 0:
                i += 1; continue
            cols += [v.real, v.imag]; covs += [2 * w.real, -2 * w.imag]; used += 2
        else:
            cols += [v.real]; covs += [w.real / (w.real @ v.real)]; used += 1
        i += 1
    return np.array(covs).T, np.array(cols).T


def mortho(Q, M, tol=1e-10):
    """M-orthonormalize the columns of Q (modified Gram-Schmidt); drops dependent columns"""
    out = []
    for q in Q.T:
        for p in out:
            q = q - (p @ M @ q) * p
        nq = np.sqrt(q @ M @ q)
        if nq > tol:
            out.append(q / nq)
    return np.array(out).T


# ---------------- ROMs ----------------
def galerkin(M, K, C, B, Phi):
    T = Phi.T
    return T @ M @ Phi, T @ K @ Phi, T @ C @ Phi, T @ B


def rom_config(M, K, C, B, Phi, u, dt, y0=None, dec=1):
    """Galerkin ROM on span(Phi); initial state projected M-orthogonally. Returns physical displacements."""
    Mr, Kr, Cr, Br = galerkin(M, K, C, B, Phi)
    q0 = None
    if y0 is not None:
        n = len(M); P = np.linalg.solve(Mr, Phi.T @ M)
        q0 = np.r_[P @ y0[:n], P @ y0[n:]]
    X, _ = simulate(Mr, Kr, Cr, Br, u, dt, q0)
    return X[::dec] @ Phi.T


def energy_weight(M, K):
    """W = blockdiag(K, M): the energy inner product on y = (x, xd)"""
    n = len(M)
    return np.block([[K, np.zeros((n, n))], [np.zeros((n, n)), M]])


def rom_state(A, Bs, Psi, Phi, u, dt, y0=None, n=None, W=None, dec=1):
    """State ROM on span(Phi). W given: energy-Galerkin (test basis W Phi). W None: Petrov-Galerkin with the data
    covectors Psi (not recommended: the reduced operator need not be stable). Returns the first n rows of the
    reconstructed state (displacements for n = number of DOFs) and the largest real part of the reduced spectrum."""
    if W is None:
        Ar = Psi.T @ A @ Phi; Br = Psi.T @ Bs; q0 = None if y0 is None else Psi.T @ y0
    else:
        T = Phi.T @ W; G = T @ Phi
        Ar = np.linalg.solve(G, T @ A @ Phi); Br = np.linalg.solve(G, T @ Bs)
        q0 = None if y0 is None else np.linalg.solve(G, T @ y0)
    Q = simulate_state(Ar, Br, u, dt, q0)
    return Q[::dec] @ Phi[:n].T, np.max(np.linalg.eigvals(Ar).real)


# ---------------- metrics ----------------
def merr(Xr, X, M):
    """mass-weighted relative error sqrt(sum D M D / sum X M X)"""
    D = Xr - X
    return np.sqrt(np.sum((D @ M) * D) / np.sum((X @ M) * X))


def relerr(Xr, X):
    return np.linalg.norm(Xr - X) / np.linalg.norm(X)


def mac(a, b, M=None):
    """modal assurance criterion (M-weighted if M given)"""
    if M is None:
        return (a @ b) ** 2 / ((a @ a) * (b @ b))
    return (a @ M @ b) ** 2 / ((a @ M @ a) * (b @ M @ b))


def principal_angles(A, B, M=None):
    """principal angles (degrees) between span(A) and span(B), in the M inner product if given"""
    if M is not None:
        L = np.linalg.cholesky(M); A = L.T @ A; B = L.T @ B
    Qa, _ = np.linalg.qr(A); Qb, _ = np.linalg.qr(B)
    s = np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), -1, 1)
    return np.degrees(np.arccos(s))
