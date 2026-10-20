function P = sod_mortho(Q, M, tol)
%SOD_MORTHO  M-orthonormalize the columns of Q (modified Gram-Schmidt); drops dependent columns.
if nargin < 3, tol = 1e-10; end
P = zeros(size(Q, 1), 0);
for j = 1:size(Q, 2)
    q = Q(:, j);
    for i = 1:size(P, 2), q = q - (P(:, i).' * M * q) * P(:, i); end
    nq = sqrt(q.' * M * q);
    if nq > tol, P(:, end+1) = q / nq; end %#ok<AGROW>
end
end
