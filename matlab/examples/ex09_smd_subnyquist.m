%EX09_SMD_SUBNYQUIST  SMD reads natural frequencies from sub-Nyquist and randomly timed snapshots; SOD cannot.
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));
C0 = sod_set_style("matlab"); rng(9);
J = 1:6; w = (J * pi).^2; zeta = 0.001; wd = w * sqrt(1 - zeta^2);
x = linspace(0, 1, 401); G = sod_smd_grid(x); Phi = sin(x.' * (J * pi));
xs = (1:13) / 14; Phis = sin(xs.' * (J * pi)); ph = 2 * pi * rand(1, 6); amp = 1 ./ w;
qf = @(t) amp .* exp(-zeta * t(:) * w) .* cos(t(:) * wd + ph);
qdf = @(t) amp .* exp(-zeta * t(:) * w) .* (-zeta * w .* cos(t(:) * wd + ph) - wd .* sin(t(:) * wd + ph));
smd_err = @(t) max(abs(sod_smd(qf(t) * Phi.', G, 'second', 'fd', ...
    struct('center', false, 'density', 1, 'stiffness', 1)).omega(:).' ./ w - 1));
fss = logspace(log10(400), log10(2), 16); E = zeros(16, 3);
for q = 1:16
    t = 0:1 / fss(q):10 - 1e-12; X = qf(t) * Phis.';
    Rc = sod_decomp(X, [], struct('dt', 1 / fss(q))); fc = sort(sod_cd_correct(Rc.freq, 1 / fss(q)));
    Re = sod_decomp(X, qdf(t) * Phis.'); fe = sort(Re.freq);
    E(q, :) = [smd_err(t), max(abs(fc(1:6).' ./ w - 1)), max(abs(fe(1:6).' ./ w - 1))];
    fprintf('fs/f6 = %6.3f: SMD %.2e  SOD (central diff.) %.2e  SOD (exact velocity) %.2e\n', fss(q) / (w(6) / 2 / pi), E(q, :));
end
er = arrayfun(@(k) smd_err(sort(10 * rand(12, 1))), 1:20); fprintf('SMD, 12 random snapshots: max error %.2e\n', max(er));
f6 = w(6) / 2 / pi; figure('Position', [100 100 520 320]);
loglog(fss / f6, E(:, 2), 's-', 'Color', C0(2,:)); hold on; loglog(fss / f6, E(:, 3), '^-', 'Color', C0(3,:));
loglog(fss / f6, E(:, 1), 'o-', 'Color', C0(1,:)); xline(2, '--', 'Nyquist');
xlabel('$f_s/f_6$'); ylabel('$\max_i|\hat\omega_i/\omega_i-1|$');
legend('SOD, central differences', 'SOD, exact velocities', 'SMD, physical weights', 'Location', 'southwest');
