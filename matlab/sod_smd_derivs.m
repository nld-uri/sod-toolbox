function D = sod_smd_derivs(W, G, operator, method, opts)
%SOD_SMD_DERIVS  Spatial derivative fields of full-field snapshots, for SMD or any other use.
%   D = SOD_SMD_DERIVS(W, G, operator, method, opts)
%   W        : nt x n (line) or nt x ny x nx (grid) snapshots; NaN = missing point (in addition to G.mask)
%   G        : grid from SOD_SMD_GRID
%   operator : 'grad'      (m = 1) d/dx on a line, [d/dx, d/dy] on a grid     strings, membranes, flows
%              'second'    (m = 2) d2/dx2 on a line, Hessian [xx, xy, yy] on a grid (xy weighted 2)  beams, plates
%              'laplacian' (m = 2) d2/dx2 on a line, d2/dx2 + d2/dy2 on a grid
%   method   : 'fd'        second-order finite differences; valid only where the whole stencil is measured
%              'savgol'    Savitzky-Golay local polynomial fits (uniform, complete grids); opts.window, or opts.kc
%                          with the window chosen from the MEASURED half-gain cutoff of the smoothing filter
%              'whittaker' penalized least squares (Wm + lam Dp'Dp) u = Wm y, lam = kc^(-2p), then finite
%                          differences; fills masked points; derivatives reported on measured points only
%   opts     : kc (rad per length unit), window, polyorder (3), penalty (m + 2), smooth_field (false)
%   D fields : F (field used in the pencil, nt x npts, NaN outside), Gk (cell of derivative fields nt x npts),
%              names, cw (component weights), valid (npts x 1), order, mask (npts x 1), info.
%   Flattening: grid points are numbered row by row (x fastest), as in the Python toolbox.
%   Cutoffs: Whittaker kc about 3-6 times the largest wavenumber of interest, Savitzky-Golay about 8 times
%   (beam test, 0.5% noise); scan kc with SOD_SMD_KC_SCAN and keep a value where the answer does not move.
if nargin < 3 || isempty(operator), operator = 'grad'; end
if nargin < 4 || isempty(method), method = 'fd'; end
if nargin < 5, opts = struct(); end
dflt = struct('kc', [], 'window', [], 'polyorder', 3, 'penalty', [], 'smooth_field', false);
f = fieldnames(dflt);
for i = 1:numel(f), if ~isfield(opts, f{i}), opts.(f{i}) = dflt.(f{i}); end, end
m = struct('grad', 1, 'second', 2, 'laplacian', 2); 
if ~isfield(m, operator), error('operator must be grad, second or laplacian'); end
m = m.(operator);
Y = flat(W, G);
nt = size(Y, 1); npts = size(Y, 2);
mask = G.mask; if numel(G.shape) == 2, mask = mask.'; end
mask = mask(:) & all(isfinite(Y), 1).';
[ops, names, cw] = buildops(G, operator);
D.info = struct();
F = Y; F(:, ~mask) = NaN;
switch method
    case 'fd'
        Yz = Y; Yz(:, ~mask) = 0;
        valid = mask; bad = double(~mask);
        for k = 1:numel(ops), valid = valid & (abs(ops{k}) * bad == 0); end
        Gk = cell(1, numel(ops));
        for k = 1:numel(ops), g = (ops{k} * Yz.').'; g(:, ~valid) = NaN; Gk{k} = g; end
    case 'whittaker'
        if isempty(opts.kc), error('method whittaker needs opts.kc'); end
        p = opts.penalty; if isempty(p), p = m + 2; end
        if numel(G.shape) == 1
            Dp = dpen(G.x, p); P = Dp.' * Dp;
        else
            ny = G.shape(1); nx = G.shape(2);
            Dx = kron(speye(ny), dpen(G.x, p)); Dy = kron(dpen(G.y, p), speye(nx));
            P = Dx.' * Dx + Dy.' * Dy;
        end
        lam = opts.kc ^ (-2 * p);
        cnd = 1 + lam * full(max(sum(abs(P), 2)));        % Gershgorin bound on the condition number
        if cnd > 1e12
            warning('sod_smd_derivs:cond', ['Whittaker system condition bound %.1e: roundoff can reach the SMD ' ...
                'results (Python and MATLAB agreed to 2e-5 at bounds 1.8e12-4.8e14); raise kc, lower the penalty ' ...
                'order, or coarsen clustered points if more agreement is needed'], cnd);
        end
        A = spdiags(double(mask), 0, npts, npts) + lam * P;
        Yz = Y; Yz(~isfinite(Yz)) = 0; Yz(:, ~mask) = 0;
        U = (A \ Yz.').';
        valid = mask;
        Gk = cell(1, numel(ops));
        for k = 1:numel(ops), g = (ops{k} * U.').'; g(:, ~valid) = NaN; Gk{k} = g; end
        if opts.smooth_field, F = U; F(:, ~mask) = NaN; end
        D.info.kc = opts.kc; D.info.penalty = p; D.info.lam = lam;
    case 'savgol'
        if ~all(mask), error('method savgol needs a complete grid; use whittaker for masked data'); end
        if ~isuniform(G.x) || (numel(G.shape) == 2 && ~isuniform(G.y)), error('method savgol needs uniform spacing'); end
        hx = G.x(2) - G.x(1); h = hx;
        if numel(G.shape) == 2, hy = G.y(2) - G.y(1); h = min(hx, hy); end
        win = opts.window;
        if isempty(win)
            if isempty(opts.kc), error('method savgol needs opts.window or opts.kc'); end
            win = sod_smd_sgwindow(opts.kc, h, opts.polyorder);
        end
        po = opts.polyorder;
        D.info.window = win; D.info.polyorder = po; D.info.kc_measured = 2 * pi * sgcut(win, po) / h;
        if numel(G.shape) == 1
            n = G.shape;
            Gk = {(sgmat(n, win, po, m, hx) * Y.').'};
            S = (sgmat(n, win, po, 0, hx) * Y.').';
        else
            ny = G.shape(1); nx = G.shape(2);
            Sx = @(d) sgmat(nx, win, po, d, hx); Sy = @(d) sgmat(ny, win, po, d, hy);
            op2 = @(a, b) (kron(Sy(b), Sx(a)) * Y.').';
            switch operator
                case 'grad', Gk = {op2(1, 0), op2(0, 1)};
                case 'laplacian', Gk = {op2(2, 0) + op2(0, 2)};
                otherwise, Gk = {op2(2, 0), op2(1, 1), op2(0, 2)};
            end
            S = op2(0, 0);
        end
        valid = mask;
        if opts.smooth_field, F = S; end
    otherwise
        error('method must be fd, savgol or whittaker');
end
D.F = F; D.Gk = Gk; D.names = names; D.cw = cw; D.valid = valid; D.mask = mask; D.order = m;
D.grid = G; D.operator = operator; D.method = method;
end

% ------------------------------------------------------------------------------------------------ helpers
function Y = flat(W, G)
if numel(G.shape) == 1
    if size(W, 2) ~= G.shape, error('field size does not match the grid'); end
    Y = W;
else
    if size(W, 2) ~= G.shape(1) || size(W, 3) ~= G.shape(2), error('field size does not match the grid'); end
    Y = reshape(permute(W, [1 3 2]), size(W, 1), []);   % x fastest
end
end

function ok = isuniform(x)
d = diff(x); ok = (max(d) - min(d)) <= 1e-9 * abs(mean(d));
end

function [ops, names, cw] = buildops(G, operator)
if numel(G.shape) == 1
    if strcmp(operator, 'grad'), ops = {d1(G.x)}; names = {'d/dx'}; else, ops = {d2(G.x)}; names = {'d2/dx2'}; end
    cw = 1; return
end
ny = G.shape(1); nx = G.shape(2); Ix = speye(nx); Iy = speye(ny);
switch operator
    case 'grad'
        ops = {kron(Iy, d1(G.x)), kron(d1(G.y), Ix)}; names = {'d/dx', 'd/dy'}; cw = [1 1];
    case 'laplacian'
        ops = {kron(Iy, d2(G.x)) + kron(d2(G.y), Ix)}; names = {'laplacian'}; cw = 1;
    otherwise
        ops = {kron(Iy, d2(G.x)), kron(d1(G.y), d1(G.x)), kron(d2(G.y), Ix)};
        names = {'d2/dx2', 'd2/dxdy', 'd2/dy2'}; cw = [1 2 1];
end
end

function D = d1(x)
n = numel(x); I = []; J = []; V = [];
for i = 2:n-1
    h1 = x(i) - x(i-1); h2 = x(i+1) - x(i);
    I = [I, i, i, i]; J = [J, i-1, i, i+1];
    V = [V, -h2 / (h1 * (h1 + h2)), (h2 - h1) / (h1 * h2), h1 / (h2 * (h1 + h2))];
end
E = {[1 2 3], [n n-1 n-2]}; rows = [1 n];
for e = 1:2
    a = E{e}(1); b = E{e}(2); c = E{e}(3); h1 = x(b) - x(a); h2 = x(c) - x(a);
    I = [I, rows(e) * [1 1 1]]; J = [J, a, b, c];
    V = [V, -(h1 + h2) / (h1 * h2), h2 / (h1 * (h2 - h1)), -h1 / (h2 * (h2 - h1))];
end
D = sparse(I, J, V, n, n);
end

function D = d2(x)
n = numel(x); I = []; J = []; V = [];
for i = 2:n-1
    h1 = x(i) - x(i-1); h2 = x(i+1) - x(i);
    I = [I, i, i, i]; J = [J, i-1, i, i+1];
    V = [V, 2 / (h1 * (h1 + h2)), -2 / (h1 * h2), 2 / (h2 * (h1 + h2))];
end
E = {[1 2 3 4], [n n-1 n-2 n-3]}; rows = [1 n];
for e = 1:2
    idx = E{e}; xs = x(idx) - x(rows(e));
    Vm = [ones(1, 4); xs; xs.^2; xs.^3];
    c = Vm \ [0; 0; 2; 0];
    I = [I, rows(e) * ones(1, 4)]; J = [J, idx]; V = [V, c.'];
end
D = sparse(I, J, V, n, n);
end

function D = dpen(x, p)
n = numel(x); D = speye(n);
for q = 1:p
    r = size(D, 1);
    S = spdiags([-ones(r-1, 1), ones(r-1, 1)], [0 1], r-1, r);
    dx = (x(q+1:end) - x(1:end-q)) / q;
    D = spdiags(1 ./ dx(:), 0, r-1, r-1) * S * D;
end
end

function S = sgmat(n, win, po, d, h)
% Savitzky-Golay derivative matrix (n x n), polynomial fits on windows of `win` points; the first and last
% half-windows are evaluated from the fits on the first and last windows (as scipy mode='interp').
if win > n, error('Savitzky-Golay window longer than the data'); end
mh = (win - 1) / 2; z = (-mh:mh).';
A = z .^ (0:po); Pi = pinv(A);                          % (po+1) x win: polynomial coefficients
rowat = @(z0) sum(Pi(d+1:end, :) .* (factorial(d:po) ./ factorial(0:po-d)).' .* (z0 .^ (0:po-d)).', 1) / h^d;
c0 = rowat(0);
I = []; J = []; V = [];
for i = mh+1:n-mh
    I = [I, i * ones(1, win)]; J = [J, i-mh:i+mh]; V = [V, c0];
end
for i = 1:mh
    I = [I, i * ones(1, win)]; J = [J, 1:win]; V = [V, rowat(i - mh - 1)];
    k = n - mh + i;
    I = [I, k * ones(1, win)]; J = [J, n-win+1:n]; V = [V, rowat(i)];
end
S = sparse(I, J, V, n, n);
end

function fc = sgcut(win, po)
mh = (win - 1) / 2; z = (-mh:mh).'; Pi = pinv(z .^ (0:po)); c = Pi(1, :);
nfft = 2^15; H = abs(fft(c, nfft)); H = H(1:nfft/2+1); f = (0:nfft/2) / nfft;
fc = f(find(H < 0.5, 1));
end
