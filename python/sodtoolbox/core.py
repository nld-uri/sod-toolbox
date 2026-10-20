"""Core smooth orthogonal decomposition (SOD).

SOD is the real symmetric generalized eigenproblem of the displacement and velocity covariances,

    Sigma_v psi = mu Sigma_x psi,      Sigma_x = <x x^T>,  Sigma_v = <xdot xdot^T>,

with mu ascending (smoothest first). Conventions used throughout the toolbox:

    psi_i^T Sigma_x psi_j = delta_ij           (covectors, "smooth projective vectors")
    q = X Psi                                  (smooth coordinates, unit variance)
    Phi = Sigma_x Psi                          (smooth modes; Psi^T Phi = I, X = Q Phi^T for full-rank data)
    mu_i = <qdot_i^2> / <q_i^2>                (SOD values; omega_i = sqrt(mu_i) is the SOD frequency)
    lambda_i = 1 / mu_i                        (smooth orthogonal values, SOVs)

Data are arrays with rows = time samples and columns = channels. Everything here is the same pencil; the options
change only how the covariances are formed (derivative estimate, lag, centring, channel scaling, rank restriction).
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh, eig

__all__ = ['SODResult', 'derivative', 'data_rank_basis', 'covariances', 'sod_pencil', 'sod', 'state_sod',
           'boundary_term', 'csod']


@dataclass
class SODResult:
    """Output of :func:`sod`.

    mu   : SOD values, ascending (mean-square frequencies of the smooth coordinates)
    freq : SOD frequencies sqrt(mu) (rad per time unit; nan where mu <= 0)
    Psi  : covectors (channels x k), psi^T Sigma_x psi = 1, in the units of the input channels
    Phi  : smooth modes Sigma_x Psi (channels x k)
    Q    : smooth coordinates X Psi (samples x k), for the samples actually used
    """
    mu: np.ndarray
    freq: np.ndarray
    Psi: np.ndarray
    Phi: np.ndarray
    Q: np.ndarray

    @property
    def sov(self):
        """smooth orthogonal values 1/mu"""
        with np.errstate(divide='ignore'):
            return 1.0 / self.mu

    def modes_unit(self):
        """smooth modes scaled to unit Euclidean norm"""
        return self.Phi / np.linalg.norm(self.Phi, axis=0)


def derivative(X, dt, method='central', h=1):
    """Time derivative of the columns of X and the matching samples of X.

    method 'central'  : (x_{j+h} - x_{j-h}) / (2 h dt); returns X[h:-h] and V (drops h samples at each end)
    method 'gradient' : numpy.gradient (second-order interior, first-order ends); keeps every sample
    For noisy measured records use a regularized estimator (e.g. the kinematics toolbox) and pass V to sod().
    """
    X = np.asarray(X, float)
    if method == 'gradient':
        return X, np.gradient(X, dt, axis=0)
    if method == 'central':
        return X[h:-h], (X[2 * h:] - X[:-2 * h]) / (2 * h * dt)
    raise ValueError("method must be 'central' or 'gradient'")


def data_rank_basis(X, tol=1e-10):
    """Orthonormal basis of the data subspace: eigenvectors of X^T X with eigenvalue above tol * largest."""
    w, U = eigh(X.T @ X)
    return U[:, w > tol * w.max()]


def _lagcov(A, B, lag):
    if lag == 0:
        return A.T @ B / len(A)
    C = A[:-lag].T @ B[lag:] / (len(A) - lag)
    return (C + C.T) / 2


def covariances(X, V, lag=0):
    """Sigma_x, Sigma_v as sample means (divided by N); lag > 0 gives the symmetrized lag-`lag` covariances,
    which remove white sensor noise from both matrices but may be indefinite in fast directions."""
    return _lagcov(X, X, lag), _lagcov(V, V, lag)


def sod_pencil(Sx, Sv, indefinite=False):
    """Solve Sigma_v psi = mu Sigma_x psi from given covariances. Returns mu ascending and Psi with
    psi^T Sigma_x psi = 1. indefinite=True uses the general eigensolver (for lagged covariances) and sorts the
    positive values first, ascending; non-positive values follow (they are the roughest directions)."""
    if not indefinite:
        return eigh(Sv, Sx)
    mu, P = eig(Sv, Sx)
    mu = mu.real
    P = P.real
    o = np.argsort(np.where(mu > 0, mu, np.inf))
    mu, P = mu[o], P[:, o]
    nrm = np.sqrt(np.abs(np.einsum('ij,jk,ki->i', P.T, Sx, P)))
    return mu, P / np.where(nrm > 0, nrm, 1.0)


def sod(X, V=None, dt=None, *, deriv='central', h=1, lag=0, center=False, scale=False, tol=1e-10, k=None):
    """Configuration SOD of a record.

    X      : samples x channels (displacements, or any signal)
    V      : its time derivative, same shape (model kinematics or a regularized estimate). If None, V is formed
             from X with `deriv` ('central', step h; or 'gradient') and dt.
    lag    : symmetric lag-`lag` covariances (noise-robust variant; general eigensolver)
    center : remove the mean of X first, before any differencing (trend and stationary data); V is not centred,
             since <xdot> is part of the signal's roughness. Off for decays about an equilibrium of interest
    scale  : divide each channel by its RMS before forming the pencil (scale-invariant rank detection; results are
             returned in the original units). The pencil is invariant to channel scaling for full-rank data.
    tol    : restrict to the data subspace, eigenvalues of X^T X above tol * largest (None: no restriction)
    k      : number of smoothest coordinates returned (default all)
    """
    X = np.asarray(X, float)
    if center:
        X = X - X.mean(0)
    if V is None:
        if dt is None:
            raise ValueError('give V or dt')
        X, V = derivative(X, dt, deriv, h)
    V = np.asarray(V, float)
    s = np.ones(X.shape[1])
    if scale:
        s = X.std(0)
        s[s == 0] = 1.0
    Xs, Vs = X / s, V / s
    U = np.eye(X.shape[1]) if tol is None else data_rank_basis(Xs, tol)
    Sx, Sv = covariances(Xs @ U, Vs @ U, lag)
    mu, P = sod_pencil(Sx, Sv, indefinite=lag > 0)
    if k is not None:
        mu, P = mu[:k], P[:, :k]
    Psi = (U @ P) / s[:, None]
    Phi = s[:, None] * (U @ (Sx @ P))
    with np.errstate(invalid='ignore'):
        freq = np.where(mu > 0, np.sqrt(np.abs(mu)), np.nan)
    return SODResult(mu=mu, freq=freq, Psi=Psi, Phi=Phi, Q=X @ Psi)


def state_sod(Y, Yd, k=None, tol=1e-10, scale=True):
    """State-space SOD: the same symmetric pencil on y = (x, xdot) and ydot = (xdot, xddot).
    Mixed units are harmless (the pencil is invariant under state-coordinate changes); scale=True only makes the
    rank tolerance unit-free. For uncorrelated modal planes each complex mode s_i gives a pair with
    mu_a mu_b = |s_i|^4 whose two smooth modes span the plane (Re v_i, Im v_i)."""
    return sod(Y, Yd, scale=scale, tol=tol, k=k)


def boundary_term(X, V, T):
    """B = (x_T v_T^T - x_0 v_0^T) / T: the integration-by-parts boundary term of a finite window."""
    return (np.outer(X[-1], V[-1]) - np.outer(X[0], V[0])) / T


def csod(Sx, Sv, Bd):
    """Boundary-corrected fit on a finite window: A^T = Sigma_x^{-1} (Sigma_v - B). Exact for undamped linear
    systems on any window. Not SOD (non-symmetric); returns frequencies ascending and the right eigenvectors."""
    mu, P = eig(np.linalg.solve(Sx, Sv - Bd))
    i = np.argsort(mu.real)
    return np.sqrt(mu.real[i]), P[:, i].real
