%EX02_FD_BIAS  Central differences bias SOD frequencies to sin(omega dt)/dt; sod_cd_correct inverts it.
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));
C0 = sod_set_style("matlab");
[M, K, C] = sod_chain(4); w = sqrt(sort(eig(K, M)));
y0 = sod_equal_energy(M, K, [], 1);
dts = logspace(log10(0.01), log10(0.4), 7); raw = zeros(7, 4); cor = raw;
for j = 1:7
    [~, X] = sod_free_decay(M, K, C, y0, dts(j), round(4000 / dts(j)));
    R = sod_decomp(X, [], struct('dt', dts(j)));
    raw(j,:) = R.freq.' ./ w.' - 1; cor(j,:) = sod_cd_correct(R.freq, dts(j)).' ./ w.' - 1;
    fprintf('dt = %.3f: raw %s | corrected %s\n', dts(j), sprintf('%+.1e ', raw(j,:)), sprintf('%+.1e ', cor(j,:)));
end
figure('Position', [100 100 760 300]); wdt = dts.' * w.';
subplot(1,2,1); loglog(wdt, -raw, 'o'); hold on; g = logspace(-2.5, 0.1, 50);
loglog(g, 1 - sin(g)./g, '--', 'Color', [0.5 0.5 0.5]); xlabel('$\omega_i\Delta t$'); ylabel('$1-\hat\omega_i/\omega_i$');
subplot(1,2,2); loglog(wdt, abs(cor), 's'); xlabel('$\omega_i\Delta t$'); ylabel('$|\hat\omega_i/\omega_i-1|$ corrected');
