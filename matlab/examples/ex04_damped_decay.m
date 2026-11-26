%EX04_DAMPED_DECAY  State SOD planes of a damped decay, recognized by circularity, and damping per plane.
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));
C0 = sod_set_style("matlab");
[M, K, C] = sod_chain(4, 0.02);
n = 4; s = eig([zeros(n) eye(n); -M\K -M\C]); s = s(imag(s) > 0); [~, o] = sort(abs(s)); s = s(o);
zt = -real(s) ./ abs(s);
y0 = sod_equal_energy(M, K, [], 1); dt = 0.01;
[t, X, V, A] = sod_free_decay(M, K, C, y0, dt, 60000);
Y = [X V]; Yd = [V A]; R = sod_state(Y, Yd); Qd = Yd * R.Psi;
pl = sod_pair_state(R.Q, dt, 10, struct('Phi', R.Phi));
zp = zeros(size(pl, 1), 1);
for p = 1:size(pl, 1)
    ab = pl(p, 1:2); [~, zp(p)] = sod_plane_eigs(sod_fit_L(R.Q(:, ab), Qd(:, ab)));
end
ev = eig(sod_fit_L(R.Q, Qd)); ev = ev(imag(ev) > 0); [~, o] = sort(abs(ev)); ev = ev(o); zs = -real(ev) ./ abs(ev);
for i = 1:4
    fprintf('mode %d: zeta exact %.5f  per-plane %.5f (%+.1e)  subspace %.5f (%+.1e)\n', i, zt(i), ...
        zp(i), zp(i)/zt(i) - 1, zs(i), zs(i)/zt(i) - 1);
end
figure('Position', [100 100 760 300]);
subplot(1,2,1); q = R.Q(:, pl(1,1:2)); plot(q(:,1), q(:,2), 'LineWidth', 0.3); axis equal; xlabel('$q_a$'); ylabel('$q_b$'); title('mode 1 plane');
subplot(1,2,2); semilogy(1:4, abs(zp ./ zt - 1), 'o', 1:4, max(abs(zs ./ zt - 1), 1e-16), 's');
xlabel('mode'); ylabel('$|\hat\zeta/\zeta-1|$'); legend('per-plane fit', 'subspace fit'); ylim([1e-16 1e-1]);
