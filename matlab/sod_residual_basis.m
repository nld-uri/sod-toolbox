function [T, Phi1, e] = sod_residual_basis(X, V, A, M, K, r1, r, kind)
%SOD_RESIDUAL_BASIS  Linear core plus residual basis for nonlinear ROMs.
%   Fit qdd = [q qd] coef on the r1 smoothest SOD coordinates; e = A - ([q qd] coef) Phi1' (equation error).
%   kind 'k' (default): POD of K\(M e'); 'a': POD of e; 'x': POD of the displacement complement X - q Phi1'.
%   (The low-passed variant 'f' of the Python toolbox is not ported.) T = M-orthonormal [Phi1, R], r columns.
if nargin < 8, kind = 'k'; end
Phi1 = sod_modes(X, V, r1); Pp = pinv(Phi1);
q = X * Pp.'; qd = V * Pp.'; qdd = A * Pp.'; G = [q qd]; coef = G \ qdd;
e = A - (G * coef) * Phi1.';
switch kind
    case 'a', Z = e;
    case 'k', Z = (K \ (M * e.')).';
    case 'x', Z = X - q * Phi1.';
    otherwise, error('kind must be ''a'', ''k'' or ''x''');
end
R = sod_pod(Z, r - r1 + 2, []); T = sod_mortho([Phi1, R], M); T = T(:, 1:r);
end
