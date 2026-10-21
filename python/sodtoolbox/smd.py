"""Smooth mode decomposition (SMD): the spatial dual of SOD for full-field data.

SOD smooths coordinates in time. SMD smooths mode shapes in space (Li, Wanchoo, Shukla & Chelidze,
Mech. Syst. Signal Process. 200 (2023) 110541). Spatial derivatives of the measured field take the place of
velocities, and the snapshots need no time order, spacing or Nyquist rate.

Data layout
    W : field snapshots, shape (n_t, n) on a line or (n_t, ny, nx) on a grid. NaN marks missing points.
    Rows (snapshots) may be in any order; SMD never uses time.

Pencil (data level). With w_q the quadrature weights of the field points and w_g those of the derivative points,
    S_W = W diag(w_q) W^T,   S_G = sum_k G_k diag(w_g) G_k^T      (n_t x n_t),
    S_G p = mu S_W p,  mu ascending (smoothest first),  p^T S_W p = 1.
Each covector p defines a mode phi = W^T p, unit norm in the weighted L2 inner product, and
    mu = ||D phi||^2 / ||phi||^2,
the Rayleigh quotient of the chosen spatial operator D. The coordinates are C = S_W P, so W = C Phi^T on the data
subspace. For sinusoidal shapes mu = k^(2m) with m the operator order; k = mu^(1/(2m)) is the SMD wavenumber.

Operators (choose by the stiffness of the structure or the physics of the field)
    'grad'      m = 1: d/dx on a line; (d/dx, d/dy) on a grid. Strings, membranes, flows (Dirichlet energy).
    'second'    m = 2: d2/dx2 on a line (beams); on a grid the Hessian (xx, sqrt2 xy, yy) (plate bending energy).
    'laplacian' m = 2: d2/dx2 on a line; d2/dx2 + d2/dy2 on a grid.
    'plate'     m = 2: Kirchhoff plate bending energy (1-nu)(w_xx^2 + 2 w_xy^2 + w_yy^2) + nu (w_xx + w_yy)^2 with
                Poisson ratio nu (keyword nu, default 0.3); 'second' on a line.

Physical weighting (Rayleigh quotient). With density (mass per unit length or area) in S_W and stiffness (T, EI or the
plate rigidity D) in S_G, mu = strain energy / kinetic-energy norm. For a self-adjoint structure whose stiffness is
matched by the operator, mu_i = omega_i^2 exactly (rad^2/s^2) and the modes are the mass-orthonormal normal modes,
from snapshots in any order (theory: SOD-SMD paper, exactness theorem). Without weights the modes of a non-uniform
structure are biased, because they are mass-orthogonal, not L2-orthogonal.

Maximum autocorrelation factors. maf() is the MAF transform of Switzer and Green (1984) with the snapshots as bands
and the grid points as pixels: the pencil of the covariance of unit-lag differences against the field covariance.
It is SMD with a first-difference gradient and unit weights, so SMD's first-order case is not new as a pencil.

Derivative estimators (all linear in the data, so the pencil is exactly the operator quotient of the modes)
    'fd'         second-order finite differences (non-uniform lines allowed); a derivative is valid only where its
                 whole stencil lies on measured points. Amplifies noise as k^m.
    'savgol'     Savitzky-Golay local polynomial fits (uniform spacing, no missing points); window set directly or
                 from a cutoff wavenumber kc, using the MEASURED gain of the smoothing filter (gain 1/2 at kc).
    'whittaker'  penalized least-squares smoothing of each snapshot, (Wm + lam D_p^T D_p) u = Wm y with
                 lam = kc^(-2p) (gain 1/(1 + (k/kc)^(2p)), 1/2 at kc), then finite differences. Fills masked
                 points from their neighbours; derivatives are reported only on measured points.
Conditioning: the Whittaker system has condition number up to about 1 + (2/(h kc))^(2p) (h the smallest spacing,
Gershgorin bound computed and warned above 1e12). Roundoff then reaches the results: Python (splu) and MATLAB
(backslash) agreed to 2e-5 in the SMD wavenumbers at bounds 1.8e12 and 4.8e14 (tests/make_smd_reference.py).
Cutoff choice (beam test, 0.5% noise, second derivative; tests/test_smd.py): Whittaker needs kc about 3 to 6 times the
largest wavenumber of interest; Savitzky-Golay, whose passband droops, needs about 8 times or more and loses accuracy
near the edges of the field. A cutoff near the band attenuates the highest mode. Always scan kc with smd_kc_scan and
keep a value where the SMD wavenumbers do not move. smooth_field=True puts the smoothed field in S_W as well; in the
beam test it changed the wavenumbers by less than 0.4%, so it is not a cure for a cutoff set too low.
"""
from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.linalg import eigh
from scipy.signal import savgol_filter, savgol_coeffs

__all__ = ['Grid', 'make_grid', 'DerivFields', 'smd_derivatives', 'SMDResult', 'smd_pencil', 'smd', 'smd_kc_scan',
           'savgol_window_for_cutoff', 'savgol_cutoff', 'maf']

_ORDER = {'grad': 1, 'second': 2, 'laplacian': 2, 'plate': 2}


# ----------------------------------------------------------------------------------------------- grid
@dataclass
class Grid:
    """Sampling geometry. shape: (n,) or (ny, nx). x: point coordinates along the line (1D) or the x axis (2D);
    y: y axis coordinates (2D). mask: bool array of `shape`, True where the field is measured.
    w: trapezoidal quadrature weights on the full grid (zero outside the mask)."""
    shape: tuple
    x: np.ndarray
    y: np.ndarray = None
    mask: np.ndarray = None
    w: np.ndarray = None

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def npts(self):
        return int(np.prod(self.shape))

    def uniform(self, tol=1e-9):
        ok = np.ptp(np.diff(self.x)) <= tol * abs(np.mean(np.diff(self.x)))
        if self.ndim == 2:
            ok = ok and np.ptp(np.diff(self.y)) <= tol * abs(np.mean(np.diff(self.y)))
        return bool(ok)


def _trap(x):
    w = np.zeros(len(x)); d = np.diff(x)
    w[:-1] += d / 2; w[1:] += d / 2
    return w


def make_grid(shape=None, x=None, y=None, h=1.0, mask=None):
    """Grid from coordinates or from a shape and spacing h (scalar, or (hy, hx) on a grid).
    1D: make_grid(x=xs) or make_grid(shape=(n,), h=dx).  2D: make_grid(x=xs, y=ys) or make_grid(shape=(ny, nx), h=...)."""
    if x is not None and y is None and shape is None:
        x = np.asarray(x, float); shape = (len(x),)
    elif x is not None and y is not None:
        x = np.asarray(x, float); y = np.asarray(y, float); shape = (len(y), len(x))
    else:
        shape = tuple(shape)
        hh = np.broadcast_to(np.atleast_1d(np.asarray(h, float)), (len(shape),))
        if len(shape) == 1:
            x = np.arange(shape[0]) * hh[0]
        else:
            y = np.arange(shape[0]) * hh[0]; x = np.arange(shape[1]) * hh[1]
    if np.any(np.diff(x) <= 0) or (y is not None and np.any(np.diff(y) <= 0)):
        raise ValueError('grid coordinates must increase strictly')
    mask = np.ones(shape, bool) if mask is None else np.asarray(mask, bool).reshape(shape)
    w = _trap(x) if len(shape) == 1 else np.outer(_trap(y), _trap(x))
    return Grid(shape, x, y, mask, np.where(mask, w, 0.0))


# ----------------------------------------------------------------------------------------------- 1D difference matrices
def _d1(x):
    """First derivative, second order on non-uniform points (one-sided three-point formulas at the ends)."""
    n = len(x); D = sp.lil_matrix((n, n))
    for i in range(1, n - 1):
        h1, h2 = x[i] - x[i - 1], x[i + 1] - x[i]
        D[i, i - 1] = -h2 / (h1 * (h1 + h2)); D[i, i] = (h2 - h1) / (h1 * h2); D[i, i + 1] = h1 / (h2 * (h1 + h2))
    for i, (a, b, c) in ((0, (0, 1, 2)), (n - 1, (n - 1, n - 2, n - 3))):
        h1, h2 = x[b] - x[a], x[c] - x[a]          # Lagrange derivative at x[a] through a, b, c
        D[i, a] = -(h1 + h2) / (h1 * h2); D[i, b] = h2 / (h1 * (h2 - h1)); D[i, c] = -h1 / (h2 * (h2 - h1))
    return D.tocsr()


def _d2(x):
    """Second derivative, second order in the interior on non-uniform points; four-point one-sided at the ends."""
    n = len(x); D = sp.lil_matrix((n, n))
    for i in range(1, n - 1):
        h1, h2 = x[i] - x[i - 1], x[i + 1] - x[i]
        D[i, i - 1] = 2 / (h1 * (h1 + h2)); D[i, i] = -2 / (h1 * h2); D[i, i + 1] = 2 / (h2 * (h1 + h2))
    for i, idx in ((0, [0, 1, 2, 3]), (n - 1, [n - 1, n - 2, n - 3, n - 4])):
        xs = x[idx] - x[i]                        # weights of the exact second derivative of the cubic through 4 points
        V = np.vander(xs, 4, increasing=True).T   # rows: moments 0..3
        rhs = np.array([0.0, 0.0, 2.0, 0.0])
        c = np.linalg.solve(V, rhs)
        for j, cj in zip(idx, c):
            D[i, j] = cj
    return D.tocsr()


def _dp(x, p):
    """p-th divided-difference penalty matrix ((n-p) x n), scaled to approximate d^p/dx^p."""
    D = sp.identity(len(x), format='csr'); xx = np.asarray(x, float)
    for q in range(1, p + 1):
        n = D.shape[0]
        S = sp.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n))
        dx = (xx[q:] - xx[:-q]) / q
        D = sp.diags(1.0 / dx) @ S @ D
    return D.tocsr()


def _ops(grid, operator, nu=0.3):
    """Sparse operators on the full flattened grid and their weights in the energy sum."""
    if grid.ndim == 1:
        x = grid.x
        if operator == 'grad':
            return [_d1(x)], ['d/dx'], [1.0]
        return [_d2(x)], ['d2/dx2'], [1.0]
    ny, nx = grid.shape
    Ix, Iy = sp.identity(nx, format='csr'), sp.identity(ny, format='csr')
    D1x, D1y, D2x, D2y = _d1(grid.x), _d1(grid.y), _d2(grid.x), _d2(grid.y)
    Dx, Dy = sp.kron(Iy, D1x, format='csr'), sp.kron(D1y, Ix, format='csr')
    if operator == 'grad':
        return [Dx, Dy], ['d/dx', 'd/dy'], [1.0, 1.0]
    Dxx, Dyy = sp.kron(Iy, D2x, format='csr'), sp.kron(D2y, Ix, format='csr')
    if operator == 'laplacian':
        return [(Dxx + Dyy).tocsr()], ['laplacian'], [1.0]
    Dxy = sp.kron(D1y, D1x, format='csr')
    if operator == 'plate':
        return ([Dxx, Dxy, Dyy, (Dxx + Dyy).tocsr()], ['d2/dx2', 'd2/dxdy', 'd2/dy2', 'laplacian'],
                [1.0 - nu, 2.0 * (1.0 - nu), 1.0 - nu, nu])
    return [Dxx, sp.kron(D1y, D1x, format='csr'), Dyy], ['d2/dx2', 'd2/dxdy', 'd2/dy2'], [1.0, 2.0, 1.0]


# ----------------------------------------------------------------------------------------------- Savitzky-Golay cutoff
def savgol_cutoff(window, polyorder, nfft=1 << 15):
    """Measured cutoff of the SG smoothing filter, in cycles per sample, where its gain falls to 1/2."""
    H = np.abs(np.fft.rfft(savgol_coeffs(window, polyorder), nfft)); f = np.arange(len(H)) / nfft
    return f[np.argmax(H < 0.5)]


def savgol_window_for_cutoff(kc, h, polyorder=3):
    """Smallest odd window whose MEASURED half-gain cutoff lies at or below kc (rad per length unit)."""
    fc = kc * h / (2 * np.pi)                      # cycles per sample
    if fc >= 0.5:
        raise ValueError('kc is above the grid Nyquist wavenumber pi/h')
    win = polyorder + 2 + (polyorder + 1) % 2
    while savgol_cutoff(win, polyorder) > fc:
        win += 2
        if win > 4097:
            raise ValueError('kc too low for any reasonable Savitzky-Golay window')
    return win


# ----------------------------------------------------------------------------------------------- derivatives
@dataclass
class DerivFields:
    """Spatial derivative fields of the snapshots.

    F       : field used in the pencil, (n_t, n_points) flattened, NaN outside the mask
    G       : list of derivative fields, each (n_t, n_points) flattened, NaN where not valid
    names   : component names (e.g. 'd/dx', 'd/dy')
    cw      : component weights in the energy sum (2 for the mixed Hessian term)
    valid   : bool array (n_points,), points where the derivative is valid
    order   : operator order m
    grid, operator, method, info
    """
    F: np.ndarray
    G: list
    names: list
    cw: list
    valid: np.ndarray
    order: int
    grid: Grid
    operator: str
    method: str
    info: dict = field(default_factory=dict)

    def component(self, k, snapshot=None):
        """Derivative component k reshaped to the grid (all snapshots, or one)."""
        A = self.G[k].reshape((-1,) + self.grid.shape)
        return A if snapshot is None else A[snapshot]


def _as_flat(W, grid):
    W = np.asarray(W, float)
    if W.shape[1:] != grid.shape:
        raise ValueError(f'field shape {W.shape[1:]} does not match grid shape {grid.shape}')
    return W.reshape(len(W), -1)


def _whittaker(Y, grid, kc, p):
    """Penalized least squares smoothing of every snapshot (rows of Y), masked points weighted zero."""
    m = grid.mask.ravel().astype(float)
    if grid.ndim == 1:
        P = (lambda D: D.T @ D)(_dp(grid.x, p))
    else:
        ny, nx = grid.shape
        Dx = sp.kron(sp.identity(ny), _dp(grid.x, p)); Dy = sp.kron(_dp(grid.y, p), sp.identity(nx))
        P = Dx.T @ Dx + Dy.T @ Dy
    lam = kc ** (-2.0 * p)
    cond = 1.0 + lam * abs(P).sum(axis=1).max()          # Gershgorin bound on the condition number
    if cond > 1e12:
        import warnings
        warnings.warn(f'Whittaker system condition bound {cond:.1e}: roundoff can reach the SMD results (different '
                      f'solvers agreed to 2e-5 in the toolbox tests at bounds 1.8e12-4.8e14); raise kc, lower the '
                      f'penalty order, or coarsen clustered points if more agreement is needed')
    A = (sp.diags(m) + lam * P).tocsc()
    lu = splu(A)
    Yz = np.where(np.isfinite(Y), Y, 0.0) * m
    return lu.solve(Yz.T).T


def smd_derivatives(W, grid, operator='grad', method='fd', kc=None, window=None, polyorder=3, penalty=None,
                    smooth_field=False, nu=0.3):
    """Spatial derivative fields of full-field snapshots W for SMD (or for any other use).

    W        : (n_t, n) or (n_t, ny, nx); NaN = missing point (in addition to grid.mask)
    operator : 'grad' | 'second' | 'laplacian' | 'plate'   (see module docstring; nu: Poisson ratio for 'plate')
    method   : 'fd' | 'savgol' | 'whittaker'
    kc       : cutoff wavenumber (rad per length unit) for 'savgol' (if window is None) and 'whittaker'
    window   : Savitzky-Golay window (odd); overrides kc
    penalty  : Whittaker penalty order p (default m + 2)
    smooth_field : use the smoothed field (not the raw one) in the pencil, for 'savgol' and 'whittaker'
    """
    if operator not in _ORDER:
        raise ValueError("operator must be 'grad', 'second', 'laplacian' or 'plate'")
    Y = _as_flat(W, grid)
    m = _ORDER[operator]
    mask = grid.mask.ravel() & np.all(np.isfinite(Y), axis=0)
    g2 = Grid(grid.shape, grid.x, grid.y, mask.reshape(grid.shape), np.where(mask.reshape(grid.shape), grid.w, 0.0))
    ops, names, cw = _ops(grid, operator, nu)
    info = {}
    F = np.where(mask, Y, np.nan)
    if method == 'fd':
        Yz = np.where(mask, Y, 0.0)
        G, valid = [], mask.copy()
        bad = (~mask).astype(float)
        for D in ops:
            valid &= (abs(D) @ bad) == 0
        for D in ops:
            G.append(np.where(valid, (D @ Yz.T).T, np.nan))
    elif method == 'whittaker':
        if kc is None:
            raise ValueError("method 'whittaker' needs a cutoff wavenumber kc")
        p = penalty or m + 2
        U = _whittaker(Y, g2, kc, p)
        valid = mask.copy()
        G = [np.where(valid, (D @ U.T).T, np.nan) for D in ops]
        if smooth_field:
            F = np.where(mask, U, np.nan)
        info.update(kc=kc, penalty=p, lam=kc ** (-2.0 * p))
    elif method == 'savgol':
        if not mask.all():
            raise ValueError("method 'savgol' needs a complete grid; use 'whittaker' for masked or gappy data")
        if not grid.uniform():
            raise ValueError("method 'savgol' needs uniform spacing; use 'fd' or 'whittaker'")
        hx = grid.x[1] - grid.x[0]
        hy = grid.y[1] - grid.y[0] if grid.ndim == 2 else None
        if window is None:
            if kc is None:
                raise ValueError("method 'savgol' needs window or kc")
            window = savgol_window_for_cutoff(kc, min(hx, hy) if hy else hx, polyorder)
        info.update(window=window, polyorder=polyorder,
                    kc_measured=2 * np.pi * savgol_cutoff(window, polyorder) / (min(hx, hy) if hy else hx))
        A = Y.reshape((len(Y),) + grid.shape)

        def sg(B, axis, d, hh):
            return savgol_filter(B, window, polyorder, deriv=d, delta=hh, axis=axis, mode='interp')
        if grid.ndim == 1:
            parts = {'grad': [(1,)], 'second': [(2,)], 'laplacian': [(2,)], 'plate': [(2,)]}[operator]
            Gs = [sg(A, 1, d[0], hx) for d in parts]
            S = sg(A, 1, 0, hx)
        else:                                       # separable: derivative along one axis, smoothing along the other
            def d2(a, b):
                return sg(sg(A, 2, a, hx), 1, b, hy)
            if operator == 'grad':
                Gs = [d2(1, 0), d2(0, 1)]
            elif operator == 'laplacian':
                Gs = [d2(2, 0) + d2(0, 2)]
            elif operator == 'plate':
                Gs = [d2(2, 0), d2(1, 1), d2(0, 2), d2(2, 0) + d2(0, 2)]
            else:
                Gs = [d2(2, 0), d2(1, 1), d2(0, 2)]
            S = d2(0, 0)
        G = [g.reshape(len(Y), -1) for g in Gs]
        valid = mask.copy()
        if smooth_field:
            F = S.reshape(len(Y), -1)
    else:
        raise ValueError("method must be 'fd', 'savgol' or 'whittaker'")
    return DerivFields(F, G, names, cw, valid, m, g2, operator, method, info)


# ----------------------------------------------------------------------------------------------- SMD
@dataclass
class SMDResult:
    """Output of :func:`smd`.

    mu    : SMD values, ascending: ||D phi||^2 / ||phi||^2 (smoothest first)
    k     : SMD wavenumbers mu^(1/(2m)) (rad per length unit)
    Phi   : modes, (n_points, r) on the full flattened grid, NaN outside the mask; unit weighted L2 norm
    C     : coordinates (n_t, r), W = C Phi^T on the data subspace (mean-removed if centred)
    P     : covectors (n_t, r), P^T S_W P = I
    sw    : eigenvalues of S_W, descending (the field's weighted energy spectrum)
    rank  : number of directions kept
    physical : True when density and stiffness were both given; then sqrt(mu) estimates natural frequencies (rad/s)
    """
    mu: np.ndarray
    k: np.ndarray
    Phi: np.ndarray
    C: np.ndarray
    P: np.ndarray
    sw: np.ndarray
    rank: int
    order: int
    grid: Grid
    operator: str
    method: str
    physical: bool = False

    @property
    def omega(self):
        """sqrt(mu): natural frequencies (rad/s) when the pencil is physically weighted, else a scale."""
        return np.sqrt(np.where(self.mu > 0, self.mu, np.nan))

    def mode(self, i):
        """Mode i reshaped to the grid."""
        return self.Phi[:, i].reshape(self.grid.shape)


def smd_pencil(SW, SG, rank=None, rtol=1e-10):
    """Solve S_G p = mu S_W p on the leading eigenspace of S_W (truncated SMD). Returns mu, P, eigenvalues of S_W."""
    s, U = eigh(SW)
    s, U = s[::-1], U[:, ::-1]
    r = int(np.sum(s > rtol * s[0])) if rank is None else int(rank)
    T = U[:, :r] / np.sqrt(s[:r])
    mu, A = eigh(T.T @ SG @ T)
    return mu, T @ A, s


def _field(a, grid, name):
    if a is None:
        return np.ones(grid.npts)
    a = np.broadcast_to(np.asarray(a, float), grid.shape).ravel()
    if np.any(a[grid.mask.ravel()] <= 0):
        raise ValueError(f'{name} must be positive on the measured points')
    return a


def smd(W, grid=None, operator='grad', method='fd', center=True, rank=None, rtol=1e-10, derivs=None,
        density=None, stiffness=None, **kw):
    """Smooth mode decomposition of full-field snapshots.

    W        : (n_t, n) on a line or (n_t, ny, nx) on a grid; NaN = missing point. Snapshot order is irrelevant.
    grid     : Grid from make_grid (default: unit spacing, complete)
    operator : 'grad' | 'second' | 'laplacian'  (physics: string/membrane/flow -> 'grad', beam -> 'second',
               plate -> 'second' (Hessian) or 'laplacian')
    method   : derivative estimator 'fd' | 'savgol' | 'whittaker' (kw: kc, window, polyorder, penalty, smooth_field)
    center   : remove the mean snapshot first (static or mean-flow part)
    rank     : number of directions kept (default: numerical rank of S_W at rtol)
    derivs   : precomputed DerivFields (then operator/method/kw are taken from it)
    density  : mass per unit length/area (scalar or grid-shaped), weights S_W (mass-orthonormal modes)
    stiffness: T, EI or plate rigidity D (scalar or grid-shaped), weights S_G; with both, mu = omega^2
    """
    W = np.asarray(W, float)
    if grid is None:
        grid = make_grid(shape=W.shape[1:])
    if center:
        W = W - np.nanmean(W, axis=0, keepdims=True)
    d = derivs if derivs is not None else smd_derivatives(W, grid, operator, method, **kw)
    g = d.grid
    # Both integrals run over the same domain: the points where the derivative is valid (finite differences lose a
    # rim next to the edges of a mask; smoothing methods keep every measured point).
    fm, gv = g.mask.ravel(), d.valid
    wq = g.w.ravel()[gv]
    rho, kap = _field(density, g, 'density')[gv], _field(stiffness, g, 'stiffness')[gv]
    Wq = d.F[:, gv] * np.sqrt(wq * rho)               # S_W = Wq Wq^T, never formed (n_t may be large)
    # Truncated SMD through the thin SVD of Wq: the leading left singular vectors span the data, then the small
    # r x r symmetric eigenproblem of S_G on that span. Same result as smd_pencil(S_W, S_G), O(n_t n r) work.
    U, sv, _ = np.linalg.svd(Wq, full_matrices=False)
    s = sv ** 2
    r = int(np.sum(s > rtol * s[0])) if rank is None else int(rank)
    T = U[:, :r] / sv[:r]
    Hm = sum(c * ((T.T @ Gk[:, gv]) * (wq * kap)) @ (Gk[:, gv].T @ T) for c, Gk in zip(d.cw, d.G))
    mu, A = eigh((Hm + Hm.T) / 2)
    P = T @ A
    Phi = np.full((g.npts, P.shape[1]), np.nan)
    Phi[fm] = d.F[:, fm].T @ P                      # modes on every measured point, normalized on the valid domain
    with np.errstate(invalid='ignore'):
        k = np.where(mu > 0, mu, np.nan) ** (1.0 / (2 * d.order))
    C = Wq @ (Wq.T @ P)                             # = S_W P
    return SMDResult(mu, k, Phi, C, P, s, P.shape[1], d.order, g, d.operator, d.method,
                     density is not None and stiffness is not None)


def smd_kc_scan(W, grid, operator, method, kcs, rank=None, **kw):
    """SMD wavenumbers for a list of cutoffs kcs (rows) -- use it to pick a cutoff where the answer is stable."""
    out = []
    for kc in kcs:
        R = smd(W, grid, operator, method, rank=rank, kc=kc, **kw)
        out.append(R.k)
    n = min(len(r) for r in out)
    return np.array([r[:n] for r in out])


def maf(W, grid=None, center=True, rank=None, rtol=1e-10, axes=None):
    """Maximum autocorrelation factors (Switzer and Green 1984) of full-field snapshots, snapshots as bands.

    Pencil: Sigma_D p = nu Sigma p, with Sigma = W W^T over the measured points and Sigma_D the sum over grid axes of
    the covariance of unit-lag differences W(x + e) - W(x), over pairs of measured neighbours. Factors f = W^T p,
    nu ascending (most autocorrelated first; autocorrelation 1 - nu/2 per axis for a stationary field).
    axes: lags summed, 'xy' (default; SMD with a forward-difference gradient, h = 1, unit weights), 'x' or 'y'
    (a single lag, as in Green et al. 1988), or a tuple of array axes of the grid shape (ny, nx).
    Returns nu, factors (n_points, r), P."""
    W = np.asarray(W, float)
    if grid is None:
        grid = make_grid(shape=W.shape[1:])
    Y = _as_flat(W, grid)
    mask = grid.mask.ravel() & np.all(np.isfinite(Y), axis=0)
    if center:
        Y = Y - np.where(mask, Y, 0.0).mean(axis=0, keepdims=True)   # remove the mean snapshot (masked columns: 0)
    Y = np.where(mask, Y, 0.0)
    idx = np.arange(grid.npts).reshape(grid.shape)
    SD = 0.0
    if axes is None or axes == 'xy':
        axes = tuple(range(grid.ndim))
    elif isinstance(axes, str):
        if grid.ndim == 1 and axes == 'x':
            axes = (0,)
        elif grid.ndim == 2 and axes in ('x', 'y'):
            axes = (1,) if axes == 'x' else (0,)
        else:
            raise ValueError("axes must be 'xy', 'x', 'y' (grid) or a tuple of array axes")
    for ax in axes:
        a = np.take(idx, np.arange(grid.shape[ax] - 1), axis=ax).ravel()
        b = np.take(idx, np.arange(1, grid.shape[ax]), axis=ax).ravel()
        ok = mask[a] & mask[b]
        Dd = Y[:, b[ok]] - Y[:, a[ok]]
        SD = SD + Dd @ Dd.T
    Ym = Y[:, mask]
    nu, P, _ = smd_pencil(Ym @ Ym.T, SD, rank, rtol)
    F = np.full((grid.npts, P.shape[1]), np.nan)
    F[mask] = Ym.T @ P
    return nu, F, P
