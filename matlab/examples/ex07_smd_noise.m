%EX07_SMD_NOISE  Noisy full-field data: regularized spatial derivatives and the choice of cutoff (one realization).
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));
C0 = sod_set_style("matlab"); warning('off', 'all');
bl = [1.8751040687; 4.6940911330; 7.8547574382]; x = linspace(0, 1, 401); G = sod_smd_grid(x);
cant = @(b) cosh(b * x) - cos(b * x) - (cosh(b) + cos(b)) / (sinh(b) + sin(b)) * (sinh(b * x) - sin(b * x));
rng(7); W0 = (randn(40, 3) .* [1 0.5 0.25]) * [cant(bl(1)); cant(bl(2)); cant(bl(3))];
r = logspace(log10(1.2), log10(40), 18); lev = [0.001 0.005 0.02];
figure('Position', [100 100 460 300]);
for c = 1:3
    W = W0 + lev(c) * std(W0(:)) * randn(size(W0));
    fd = max(abs(sod_smd(W, G, 'second', 'fd', struct('rank', 3)).k ./ bl - 1));
    K = sod_smd_kc_scan(W, G, 'second', 'whittaker', r * bl(end), struct('rank', 3));
    e = max(abs(K ./ bl.' - 1), [], 2); [em, i] = min(e);
    fprintf('noise %.1f%%: plain fd %.2e; Whittaker best %.2e at kc = %.1f k_max\n', 100 * lev(c), fd, em, r(i));
    loglog(r, e, 'o-', 'Color', C0(c,:), 'DisplayName', sprintf('%g\\%% noise', 100 * lev(c))); hold on
    yline(fd, ':', 'Color', C0(c,:), 'HandleVisibility', 'off');
end
xlabel('$k_c/k_{\max}$'); ylabel('$\max_i|\hat k_i/\beta_i-1|$'); legend('Location', 'southeast');
title('Whittaker (dotted: plain finite differences)');
