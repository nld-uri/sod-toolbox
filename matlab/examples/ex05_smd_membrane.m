%EX05_SMD_MEMBRANE  SMD of a membrane from unordered snapshots; eight snapshots suffice for eight modes.
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));
C0 = sod_set_style("matlab");
A = 1; B = 0.7; x = linspace(0, A, 81); y = linspace(0, B, 57); G = sod_smd_grid(x, y);
[mm, nn] = meshgrid(1:4, 1:3); mm = mm(:); nn = nn(:);
k = pi * hypot(mm / A, nn / B); [k, o] = sort(k); k = k(1:8); mm = mm(o(1:8)); nn = nn(o(1:8));
S = zeros(numel(x) * numel(y), 8);
for i = 1:8, S(:, i) = reshape((sin(nn(i) * pi * y.' / B) * sin(mm(i) * pi * x / A)).', [], 1); end
rng(5);
for nt = [200 8]
    W = permute(reshape((randn(nt, 8) * S.').', numel(x), numel(y), nt), [3 2 1]);   % nt x ny x nx
    R = sod_smd(W, G, 'grad', 'fd', struct('center', false));
    mac = arrayfun(@(i) sod_mac(R.Phi(:, i), S(:, i)), 1:8);
    fprintf('%3d snapshots: max |k/k_exact - 1| = %.2e, max (1 - MAC) = %.1e\n', nt, max(abs(R.k ./ k - 1)), max(1 - mac));
end
figure('Position', [100 100 900 320]);
for i = 1:8
    subplot(2, 6, i + 2 * (i > 4)); M1 = sod_smd_mode(R, i);
    contourf(x, y, M1 / max(abs(M1(:))), linspace(-1, 1, 11), 'LineStyle', 'none'); axis equal off
    title(sprintf('$(%d,%d)$', mm(i), nn(i)));
end
colormap(sod_rdbu());
subplot(2, 6, [5 6 11 12]); plot(k, R.k ./ k - 1, 'o', 'Color', C0(1,:)); xlabel('exact $k_{mn}$'); ylabel('$\hat k/k-1$');
title('eight snapshots');

function c = sod_rdbu()
% diverging blue-white-red map for mode shapes
t = linspace(0, 1, 128).'; c = [interp1([0 .5 1], [0.13 1 0.70], t), interp1([0 .5 1], [0.40 1 0.09], t), ...
    interp1([0 .5 1], [0.67 1 0.17], t)];
end
