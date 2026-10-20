function [Phi, w] = sod_modes(X, V, r, tol)
%SOD_MODES  Smooth modes (unit Euclidean norm, data subspace) and SOD frequencies, smoothest first, at most r.
if nargin < 4, tol = 1e-10; end
R = sod_decomp(X, V, struct('tol', tol)); k = min(r, size(R.Phi, 2));
Phi = R.Phi(:, 1:k) ./ vecnorm(R.Phi(:, 1:k)); w = sqrt(R.mu(1:k));
end
