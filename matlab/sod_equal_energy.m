function y0 = sod_equal_energy(M, K, modes, seed)
%SOD_EQUAL_ENERGY  Initial state with energy 1/2 in each chosen LNM and random phases.
[P, w2] = eig(K, M, 'chol', 'vector'); [w2, o] = sort(w2); P = P(:, o); w = sqrt(w2);
P = P ./ sqrt(sum(P .* (M * P), 1));
if nargin < 3 || isempty(modes), modes = 1:numel(w); end
if nargin >= 4, rng(seed); end
th = 2*pi*rand(numel(modes), 1);
y0 = [P(:, modes) * (cos(th) ./ w(modes)); P(:, modes) * (-sin(th))];
end
