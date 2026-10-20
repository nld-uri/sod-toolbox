"""Small benchmark systems for the examples and tests.

chain        n-DOF spring-mass chain fixed at the left end, optional Rayleigh damping
lnm_modes    natural frequencies and mass-normalized linear normal modes
complex_eigs the eigenvalues s = sigma + i omega_d (upper half plane), slowest first
equal_energy initial state with equal energy in each of the chosen LNMs and random phases
free_decay   exact free response of a linear chain (x, v, a)
duffing_decay free response of the chain with a cubic spring between mass 1 and ground
"""
import numpy as np
from scipy.linalg import eigh
from scipy.integrate import solve_ivp
from .rom import state_matrices, simulate_state

__all__ = ['chain', 'lnm_modes', 'complex_eigs', 'equal_energy', 'free_decay', 'duffing_decay']


def chain(n=4, m=None, k=None, zeta=0.0, modes=(0, None)):
    """M, K, C of a chain fixed at the left end. m, k: masses (n) and springs (n, spring i joins mass i-1 and i).
    zeta: Rayleigh damping ratio at the two modes in `modes` (default first and last)."""
    m = np.linspace(1.0, 2.0, n) if m is None else np.asarray(m, float)
    k = np.linspace(3.0, 1.0, n) if k is None else np.asarray(k, float)
    M = np.diag(m); K = np.zeros((n, n))
    for i in range(n):
        K[i, i] += k[i]
        if i + 1 < n:
            K[i, i] += k[i + 1]; K[i, i + 1] -= k[i + 1]; K[i + 1, i] -= k[i + 1]
    C = np.zeros((n, n))
    if zeta:
        w = np.sqrt(eigh(K, M, eigvals_only=True)); i, j = modes[0], (n - 1 if modes[1] is None else modes[1])
        a, b = np.linalg.solve([[1 / (2 * w[i]), w[i] / 2], [1 / (2 * w[j]), w[j] / 2]], [zeta, zeta])
        C = a * M + b * K
    return M, K, C


def lnm_modes(M, K):
    w2, P = eigh(K, M)
    return np.sqrt(w2), P


def complex_eigs(M, K, C):
    s = np.linalg.eigvals(state_matrices(M, K, C, np.zeros((len(M), 1)))[0])
    s = s[s.imag > 0]
    return s[np.argsort(abs(s))]


def equal_energy(M, K, modes=None, rng=None):
    """y0 = (x0, v0): mode k gets energy 1/2 with a random phase (q_k = cos th / w_k, qdot_k = -sin th)"""
    rng = np.random.default_rng(0) if rng is None else rng
    w, P = lnm_modes(M, K)
    sel = np.arange(len(w)) if modes is None else np.asarray(modes)
    th = rng.uniform(0, 2 * np.pi, len(sel))
    return np.r_[P[:, sel] @ (np.cos(th) / w[sel]), P[:, sel] @ (-np.sin(th))]


def free_decay(M, K, C, y0, dt, N):
    """exact free response sampled at t_k = k dt, k = 0..N-1; returns t, X, V, A"""
    A, Bs = state_matrices(M, K, C, np.zeros((len(M), 1)))
    Y = simulate_state(A, Bs, np.zeros((N, 1)), dt, y0)
    n = len(M); Yd = Y @ A.T
    return np.arange(N) * dt, Y[:, :n], Y[:, n:], Yd[:, n:]


def duffing_decay(M, K, C, k3, y0, dt, N, rtol=1e-10):
    """chain plus a cubic spring k3 x_1^3 from mass 1 to ground (hardening for k3 > 0); returns t, X, V, A"""
    n = len(M); Mi = np.linalg.inv(M)

    def f(t, y):
        x, v = y[:n], y[n:]
        fn = np.zeros(n); fn[0] = k3 * x[0] ** 3
        return np.r_[v, -Mi @ (K @ x + C @ v + fn)]
    t = np.arange(N) * dt
    s = solve_ivp(f, (0, t[-1]), y0, t_eval=t, rtol=rtol, atol=rtol * 1e-2, method='DOP853')
    Y = s.y.T; Yd = np.array([f(0, y) for y in Y])
    return t, Y[:, :n], Y[:, n:], Yd[:, n:]
