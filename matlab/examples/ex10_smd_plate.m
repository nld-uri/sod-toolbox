%EX10_SMD_PLATE  SMD of a simply supported Kirchhoff plate: omega^2 from the bending energy.
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));
C0 = sod_set_style("matlab"); rng(10);
A = 1; B = 0.6; x = linspace(0, A, 101); y = linspace(0, B, 61); G = sod_smd_grid(x, y);
[mm, nn] = meshgrid(1:5, 1:3); mm = mm(:); nn = nn(:);
w = pi^2 * ((mm / A).^2 + (nn / B).^2); [w, o] = sort(w); w = w(1:8); mm = mm(o(1:8)); nn = nn(o(1:8));
S = zeros(numel(x) * numel(y), 8);
for i = 1:8, S(:, i) = reshape((sin(nn(i) * pi * y.' / B) * sin(mm(i) * pi * x / A)).', [], 1); end
W = permute(reshape((randn(40, 8) * S.').', numel(x), numel(y), 40), [3 2 1]);
ops = {'plate', 'second', 'laplacian'}; figure('Position', [100 100 420 300]); hold on; box on
for c = 1:3
    R = sod_smd(W, G, ops{c}, 'fd', struct('center', false, 'density', 1, 'stiffness', 1, 'nu', 0.3));
    e = R.omega(:) ./ w - 1; fprintf('%-9s: %s\n', ops{c}, sprintf('%+.1e ', e));
    plot(w, 100 * e, 'o', 'Color', C0(c,:), 'DisplayName', ['\texttt{' ops{c} '}']);
end
xlabel('exact $\omega_{mn}$ (rad/s)'); ylabel('$\hat\omega/\omega-1$ (\%)'); legend('Location', 'southwest');
