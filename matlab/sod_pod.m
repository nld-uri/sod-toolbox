function [U, w] = sod_pod(X, r, tol)
%SOD_POD  Leading r proper orthogonal modes (uncentred) and the POVs (descending). tol ([] = no rank limit).
if nargin < 3, tol = 1e-10; end
[U, w] = eig(X.' * X, 'vector'); [w, o] = sort(w, 'descend'); U = U(:, o);
if isempty(tol), k = r; else, k = min(r, sum(w > tol * max(w))); end
U = U(:, 1:k);
end
