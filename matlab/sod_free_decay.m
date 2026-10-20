function [t, X, V, A] = sod_free_decay(M, K, C, y0, dt, N)
%SOD_FREE_DECAY  Exact free response of a linear chain at t_k = k dt, k = 0..N-1 (matrix exponential).
%   State y = [x; v],  ydot = As y,  As = [0 I; -M\K -M\C].
n = size(M, 1); As = [zeros(n) eye(n); -M\K -M\C];
[Vs, d] = eig(As, 'vector'); c = Vs \ y0; t = (0:N-1).' * dt;
Y = real((exp(t * d.') .* c.') * Vs.');          % rows: y(t_k)'
X = Y(:, 1:n); V = Y(:, n+1:end); Yd = Y * As.'; A = Yd(:, n+1:end);
end
