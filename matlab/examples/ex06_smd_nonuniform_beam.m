%EX06_SMD_NONUNIFORM_BEAM  Density and stiffness weights give exact modes and mu = omega^2 for a non-uniform beam.
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));
C0 = sod_set_style("matlab");
EI = @(x) ones(size(x)); rho = @(x) 1 + 2 * x;                  % pinned-pinned, L = 1
[xq, wq] = gauss_legendre(800); j = (1:100) * pi; Sq = sin(xq * j);
Mg = Sq.' * (Sq .* (wq .* rho(xq))); Kg = (Sq .* j.^2).' * ((Sq .* j.^2) .* (wq .* EI(xq)));
[Cg, w2] = eig((Kg + Kg.') / 2, (Mg + Mg.') / 2, 'vector'); [w2, o] = sort(w2); Cg = Cg(:, o); w = sqrt(w2(1:6));
rng(6); ns = [101 201 401 801 1601]; ferr = zeros(size(ns));
for q = 1:numel(ns)
    x = linspace(0, 1, ns(q)); G = sod_smd_grid(x); Phi = sin(x.' * j) * Cg(:, 1:6);
    W = randn(6) * Phi.';
    Rp = sod_smd(W, G, 'second', 'fd', struct('center', false, 'density', rho(x), 'stiffness', EI(x)));
    ferr(q) = max(abs(Rp.omega ./ w - 1));
    if ns(q) == 401
        Ru = sod_smd(W, G, 'second', 'fd', struct('center', false));
        fprintf('1 - MAC, unit weights: %s\n', sprintf('%.1e ', 1 - arrayfun(@(i) sod_mac(Ru.Phi(:,i), Phi(:,i)), 1:6)));
        fprintf('1 - MAC, physical    : %s\n', sprintf('%.1e ', 1 - arrayfun(@(i) sod_mac(Rp.Phi(:,i), Phi(:,i)), 1:6)));
    end
    fprintf('n = %5d: max |omega_SMD/omega - 1| = %.2e\n', ns(q), ferr(q));
end
h = 1 ./ (ns - 1);
figure('Position', [100 100 420 300]); loglog(h, ferr, 'o-', 'Color', C0(1,:)); hold on
loglog(h, ferr(3) * (h / h(3)).^2, '--', 'Color', [0.5 0.5 0.5]); xlabel('$h/L$'); ylabel('$\max_i|\hat\omega_i/\omega_i-1|$');
legend('physical weights', '$\propto h^2$', 'Location', 'northwest');

function [x, w] = gauss_legendre(n)
% Gauss-Legendre nodes and weights on [0, 1] (Golub-Welsch)
b = (1:n-1) ./ sqrt(4 * (1:n-1).^2 - 1); [V, D] = eig(diag(b, 1) + diag(b, -1));
[x, o] = sort(diag(D)); w = 2 * V(1, o).'.^2; x = (x + 1) / 2; w = w / 2;
end
