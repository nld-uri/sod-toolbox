function U = sod_rank_basis(X, tol)
%SOD_RANK_BASIS  Orthonormal basis of the data subspace: eigenvectors of X'X with eigenvalue above tol*largest.
if nargin < 2, tol = 1e-10; end
[U, W] = eig(X.' * X, 'vector');
U = U(:, W > tol * max(W));
end
