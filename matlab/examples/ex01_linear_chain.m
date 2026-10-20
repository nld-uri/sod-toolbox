%EX01_LINEAR_CHAIN  SOD of an undamped chain: SOVs are 1/omega^2 and smooth modes are the LNMs.
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));
C0 = sod_set_style("matlab");
[M, K, C] = sod_chain(4);
[P, w2] = eig(K, M, 'chol', 'vector'); [w2, o] = sort(w2); P = P(:, o); w = sqrt(w2);
y0 = sod_equal_energy(M, K, [], 1);
[t, X, V] = sod_free_decay(M, K, C, y0, 0.02, 200000);           % 4000 time units
R = sod_decomp(X, V);
fprintf('SOV x omega^2 - 1: %s\n', sprintf('%+.1e ', R.sov .* w2 - 1));
Pn = P ./ vecnorm(P); Phi = R.Phi ./ vecnorm(R.Phi);
fprintf('MAC(smooth mode, LNM): %s\n', sprintf('%.8f ', arrayfun(@(i) sod_mac(Phi(:,i), Pn(:,i)), 1:4)));
figure('Position', [100 100 760 300]);
subplot(1,2,1); loglog(1 ./ w2, R.sov, 'o', 'Color', C0(1,:), 'MarkerSize', 8); hold on
loglog([0.1 10], [0.1 10], '--', 'Color', [0.5 0.5 0.5]); xlabel('$1/\omega_i^2$'); ylabel('SOV $\lambda_i$');
legend('SOD', '$\lambda = 1/\omega^2$', 'Location', 'northwest');
subplot(1,2,2); hold on; box on
for i = 1:4
    s = sign(Phi(:,i).' * Pn(:,i));
    plot(0:4, [0; Pn(:,i)], '-', 'Color', [C0(i,:) 0.35], 'LineWidth', 2.4, 'HandleVisibility', 'off');
    plot(0:4, [0; s * Phi(:,i)], 'o', 'Color', C0(i,:), 'MarkerFaceColor', C0(i,:), 'DisplayName', sprintf('%d', i));
end
xlabel('mass'); ylabel('shape'); legend('Location', 'eastoutside'); title('dots: smooth modes, lines: LNMs');
