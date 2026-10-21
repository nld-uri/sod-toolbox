function [nu, F, P] = sod_smd_maf(W, G, opts)
%SOD_SMD_MAF  Maximum autocorrelation factors (Switzer and Green 1984), snapshots as bands, grid points as pixels.
%   [nu, F, P] = SOD_SMD_MAF(W, G, opts)
%   Pencil Sigma_D p = nu Sigma p: Sigma = W W' over the measured points; Sigma_D = sum over grid axes of the
%   covariance of unit-lag differences between measured neighbours. Factors F = W' P (npts x r), nu ascending.
%   It is SMD with a forward-difference gradient, unit spacing and unit weights (see SOD_SMD).
%   opts: center (true), rank ([]), rtol (1e-10), axes ('xy' sums both unit lags; 'x' or 'y' is the single lag
%         of Green et al. 1988; on a line only 'x').
if nargin < 2 || isempty(G)
    if ndims(W) == 2, G = sod_smd_grid(0:size(W, 2)-1); else, G = sod_smd_grid(0:size(W, 3)-1, 0:size(W, 2)-1); end
end
if nargin < 3, opts = struct(); end
if ~isfield(opts, 'center'), opts.center = true; end
if ~isfield(opts, 'rank'), opts.rank = []; end
if ~isfield(opts, 'rtol'), opts.rtol = 1e-10; end
if ~isfield(opts, 'axes'), opts.axes = 'xy'; end
nt = size(W, 1); npts = prod(G.shape);
if numel(G.shape) == 1, Y = reshape(W, nt, npts); mask = G.mask(:);
else, Y = reshape(permute(W, [1 3 2]), nt, npts); mask = reshape(G.mask.', [], 1); end
mask = mask & all(isfinite(Y), 1).';
if opts.center, Yn = Y; Yn(:, ~mask) = NaN; Y = Y - mean(Yn, 1, 'omitnan'); end
Y(:, ~mask) = 0;
if numel(G.shape) == 1, idx = (1:npts).'; else, idx = reshape(1:npts, G.shape(2), G.shape(1)); end  % x fastest
SD = zeros(nt);
switch opts.axes      % idx: dimension 1 is x, dimension 2 is y
    case 'xy', dims = 1:numel(G.shape);
    case 'x', dims = 1;
    case 'y', if numel(G.shape) == 1, error('a line has no y lag'); end, dims = 2;
    otherwise, error('axes must be xy, x or y');
end
for ax = dims
    if ax == 1, a = idx(1:end-1, :); b = idx(2:end, :); else, a = idx(:, 1:end-1); b = idx(:, 2:end); end
    a = a(:); b = b(:); ok = mask(a) & mask(b);
    Dd = Y(:, b(ok)) - Y(:, a(ok)); SD = SD + Dd * Dd.';
end
Ym = Y(:, mask); SW = Ym * Ym.'; SW = (SW + SW.') / 2; SD = (SD + SD.') / 2;
[U, s] = eig(SW, 'vector'); [s, o] = sort(s, 'descend'); U = U(:, o);
if isempty(opts.rank), r = sum(s > opts.rtol * s(1)); else, r = opts.rank; end
T = U(:, 1:r) ./ sqrt(s(1:r)).';
Hm = T.' * SD * T; Hm = (Hm + Hm.') / 2;
[A, nu] = eig(Hm, 'vector'); [nu, o] = sort(nu); A = A(:, o);
P = T * A;
F = NaN(npts, r); F(mask, :) = Ym.' * P;
end
