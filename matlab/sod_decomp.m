function R = sod_decomp(X, V, opts)
%SOD_DECOMP  Configuration smooth orthogonal decomposition of a record.
%   R = SOD_DECOMP(X, V)        X, V: samples x channels (displacement and its time derivative)
%   R = SOD_DECOMP(X, [], opts) forms V from X with opts.dt and opts.deriv ('central' with step opts.h, or 'gradient')
%   Pencil:  Sigma_v psi = mu Sigma_x psi,  psi' Sigma_x psi = 1,  mu ascending (smoothest first).
%   opts fields (defaults): dt, deriv ('central'), h (1), lag (0: plain covariances; >0: symmetrized lag-lag
%   covariances, general eigensolver), center (false: remove mean of X before differencing), scale (false: unit-RMS
%   channels for the rank test), tol (1e-10; [] = no data-rank restriction), k ([] = all).
%   R fields: mu, freq (sqrt(mu)), sov (1/mu), Psi (covectors), Phi (smooth modes Sigma_x Psi), Q (= X Psi).
if nargin < 3, opts = struct(); end
d = struct('dt', [], 'deriv', 'central', 'h', 1, 'lag', 0, 'center', false, 'scale', false, 'tol', 1e-10, 'k', []);
f = fieldnames(d);
for i = 1:numel(f), if ~isfield(opts, f{i}), opts.(f{i}) = d.(f{i}); end, end
if opts.center, X = X - mean(X, 1); end
if isempty(V)
    [X, V] = sod_deriv(X, opts.dt, opts.deriv, opts.h);
end
s = ones(1, size(X, 2));
if opts.scale, s = std(X, 1, 1); s(s == 0) = 1; end
Xs = X ./ s; Vs = V ./ s;
if isempty(opts.tol), U = eye(size(X, 2)); else, U = sod_rank_basis(Xs, opts.tol); end
[Sx, Sv] = sod_cov(Xs * U, Vs * U, opts.lag);
[mu, P] = sod_pencil(Sx, Sv, opts.lag > 0);
if ~isempty(opts.k), k = min(opts.k, numel(mu)); mu = mu(1:k); P = P(:, 1:k); end
R.mu = mu;
R.freq = sqrt(abs(mu)); R.freq(mu <= 0) = NaN;
R.sov = 1 ./ mu;
R.Psi = (U * P) ./ s.';
R.Phi = s.' .* (U * (Sx * P));
R.Q = X * R.Psi;
end
