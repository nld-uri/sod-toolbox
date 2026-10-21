%EX08_SMD_BOUNDARIES  Partial fields of view and holes bias SMD (boundary terms).
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));
C0 = sod_set_style("matlab"); warning('off', 'all'); rng(8);
bl = [1.8751040687; 4.6940911330; 7.8547574382]; x = linspace(0, 1, 801);
cant = @(b) cosh(b * x) - cos(b * x) - (cosh(b) + cos(b)) / (sinh(b) + sin(b)) * (sinh(b * x) - sin(b * x));
Wb = randn(30, 3) * [cant(bl(1)); cant(bl(2)); cant(bl(3))];
fr = linspace(0.3, 1, 15); kb = zeros(15, 3);
for q = 1:15
    sel = x >= 0.5 - fr(q) / 2 - 1e-12 & x <= 0.5 + fr(q) / 2 + 1e-12;
    R = sod_smd(Wb(:, sel), sod_smd_grid(x(sel)), 'second', 'fd', struct('center', false)); kb(q, :) = R.k(:).' ./ bl.' - 1;
end
fprintf('field of view 0.80 L: %s\n', sprintf('%+.2e ', kb(fr > 0.799 & fr < 0.801, :)));
A = 1; B = 0.7; xm = linspace(0, A, 101); ym = linspace(0, B, 71); mn = [1 1; 2 1; 1 2; 3 1];
km = pi * hypot(mn(:,1) / A, mn(:,2) / B); S = zeros(numel(xm) * numel(ym), 4);
for i = 1:4, S(:, i) = reshape((sin(mn(i,2) * pi * ym.' / B) * sin(mn(i,1) * pi * xm / A)).', [], 1); end
Wm = permute(reshape((randn(30, 4) * S.').', numel(xm), numel(ym), 30), [3 2 1]);
[XX, YY] = meshgrid(xm, ym); rad = 0:0.02:0.14; kh = zeros(numel(rad), 4);
for q = 1:numel(rad)
    G = sod_smd_grid(xm, ym, ~(hypot(XX - 0.6, YY - 0.35) < rad(q)));
    R = sod_smd(Wm, G, 'grad', 'whittaker', struct('kc', 30 * max(km), 'center', false)); kh(q, :) = R.k(:).' ./ km.' - 1;
end
fprintf('hole radius 0.14: %s\n', sprintf('%+.2e ', kh(end, :)));
figure('Position', [100 100 800 300]);
subplot(1,2,1); semilogy(fr, abs(kb), 'o-'); xlabel('measured fraction of the span'); ylabel('$|\hat k_i/\beta_i-1|$');
legend('mode 1', 'mode 2', 'mode 3', 'Location', 'southwest');
subplot(1,2,2); plot(rad, 100 * kh, 's-'); xlabel('hole radius $r/a$'); ylabel('$\hat k/k_{mn}-1$ (\%)');
legend('$(1,1)$', '$(2,1)$', '$(1,2)$', '$(3,1)$', 'Location', 'northwest');
