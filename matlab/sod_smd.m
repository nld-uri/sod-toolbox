function R = sod_smd(W, G, operator, method, opts)
%SOD_SMD  Smooth mode decomposition (SMD): the spatial dual of SOD for full-field snapshots.
%   R = SOD_SMD(W, G, operator, method, opts)
%   SOD smooths coordinates in time; SMD smooths mode shapes in space (Li, Wanchoo, Shukla and Chelidze,
%   Mech. Syst. Signal Process. 200 (2023) 110541). Snapshot order, spacing and rate are irrelevant.
%   W        : nt x n (line) or nt x ny x nx (grid); NaN = missing point
%   G        : grid from SOD_SMD_GRID (default: unit spacing, complete)
%   operator : 'grad' (strings, membranes, flows), 'second' (beams; Hessian on a grid), 'laplacian',
%              'plate' (Kirchhoff bending energy, Poisson ratio opts.nu)
%   method   : 'fd' | 'savgol' | 'whittaker'   (see SOD_SMD_DERIVS; opts.kc, opts.window, ...)
%   opts     : center (true: remove the mean snapshot), rank ([] = numerical rank at rtol), rtol (1e-10),
%              derivs (precomputed output of SOD_SMD_DERIVS), plus the SOD_SMD_DERIVS options;
%              density (mass per length/area, scalar or field-shaped) weights S_W; stiffness (T, EI or D)
%              weights S_G; with both, mu = omega^2 and R.omega = sqrt(mu) (rad/s)
%   Pencil   : S_G p = mu S_W p, S_W = W diag(w) W', S_G = sum_k cw_k G_k diag(w) G_k' over the points where
%              the derivative is valid; p' S_W p = 1; mu ascending (smoothest first).
%   R fields : mu (||D phi||^2/||phi||^2), k (wavenumbers mu^(1/(2m))), Phi (npts x r modes, unit weighted norm,
%              NaN outside the mask; reshape with SOD_SMD_MODE), C (nt x r coordinates, W = C Phi'), P (covectors),
%              sw (eigenvalues of S_W, descending), rank, order, grid.
if nargin < 2 || isempty(G)
    if ndims(W) == 2, G = sod_smd_grid(0:size(W, 2)-1); else, G = sod_smd_grid(0:size(W, 3)-1, 0:size(W, 2)-1); end
end
if nargin < 3 || isempty(operator), operator = 'grad'; end
if nargin < 4 || isempty(method), method = 'fd'; end
if nargin < 5, opts = struct(); end
if ~isfield(opts, 'center'), opts.center = true; end
if ~isfield(opts, 'rank'), opts.rank = []; end
if ~isfield(opts, 'rtol'), opts.rtol = 1e-10; end
if opts.center, W = W - mean(W, 1, 'omitnan'); end
if isfield(opts, 'derivs') && ~isempty(opts.derivs), D = opts.derivs; else, D = sod_smd_derivs(W, G, operator, method, opts); end
w = G.w; if numel(G.shape) == 2, w = w.'; end
w = w(:); gv = D.valid;
rho = pfield(opts, 'density', G); kap = pfield(opts, 'stiffness', G);
wq = (w(gv) .* rho(gv)).'; wk = (w(gv) .* kap(gv)).';
Wq = D.F(:, gv) .* sqrt(wq);            % S_W = Wq Wq', never formed (nt may be large)
[U, sv] = svd(Wq, 'econ'); sv = diag(sv); s = sv .^ 2;   % thin SVD: leading left vectors span the data
if isempty(opts.rank), r = sum(s > opts.rtol * s(1)); else, r = opts.rank; end
T = U(:, 1:r) ./ sv(1:r).';
Hm = zeros(r);
for k = 1:numel(D.Gk), B = T.' * D.Gk{k}(:, gv); Hm = Hm + D.cw(k) * (B .* wk) * B.'; end
Hm = (Hm + Hm.') / 2;
[A, mu] = eig(Hm, 'vector'); [mu, o] = sort(mu); A = A(:, o);
P = T * A;
Phi = NaN(numel(w), r);
Phi(D.mask, :) = D.F(:, D.mask).' * P;
R.mu = mu; R.k = mu .^ (1 / (2 * D.order)); R.k(mu <= 0) = NaN;
R.Phi = Phi; R.C = Wq * (Wq.' * P); R.P = P; R.sw = s; R.rank = r; R.order = D.order; R.grid = G;
R.operator = D.operator; R.method = D.method;
R.physical = isfield(opts, 'density') && isfield(opts, 'stiffness');
R.omega = sqrt(mu); R.omega(mu <= 0) = NaN;
end

function a = pfield(opts, name, G)
%PFIELD  Pointwise weight (density or stiffness) flattened like the field (x fastest); ones if absent.
n = prod(G.shape);
if ~isfield(opts, name) || isempty(opts.(name)), a = ones(n, 1); return, end
a = opts.(name);
if isscalar(a), a = a * ones(n, 1); return, end
if numel(G.shape) == 2, a = reshape(a, G.shape).'; end
a = a(:);
m = G.mask; if numel(G.shape) == 2, m = m.'; end
if any(a(m(:)) <= 0), error('%s must be positive on the measured points', name); end
end
