function [mu, P] = sod_pencil(Sx, Sv, indefinite)
%SOD_PENCIL  Solve Sigma_v psi = mu Sigma_x psi; mu ascending, psi' Sigma_x psi = 1.
%   indefinite = true (lagged covariances): general eigensolver; positive values first, ascending.
if nargin < 3, indefinite = false; end
if ~indefinite
    Sx = (Sx + Sx.') / 2; Sv = (Sv + Sv.') / 2;
    [P, mu] = eig(Sv, Sx, 'chol', 'vector');
    [mu, o] = sort(real(mu)); P = real(P(:, o));
else
    [P, mu] = eig(Sv, Sx, 'vector');
    mu = real(mu); P = real(P);
    key = mu; key(mu <= 0) = Inf;
    [~, o] = sort(key); mu = mu(o); P = P(:, o);
end
nrm = sqrt(abs(sum(P .* (Sx * P), 1))); nrm(nrm == 0) = 1;
P = P ./ nrm;
end
