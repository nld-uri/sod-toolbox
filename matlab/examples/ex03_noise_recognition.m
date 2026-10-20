%EX03_NOISE_RECOGNITION  Sensor noise: central differences versus lagged covariances; recognition by frequency.
addpath(fullfile(fileparts(mfilename('fullpath')), '..'));
C0 = sod_set_style("matlab");
n = 6; [M, K, C] = sod_chain(n, 0.02); w = sqrt(sort(eig(K, M)));
As = [zeros(n) eye(n); -M\K -M\C]; Bs = [zeros(n,1); M \ [zeros(n-1,1); 1]];
dt = 0.05; N = 80000; rng(3);
U = fft(randn(N, 1)); om = 2*pi*[0:N/2, -N/2+1:-1].' / (N*dt); U(abs(om) > 2.6) = 0;
u = real(ifft(U)); u = u / std(u);                                         % band-limited noise, base MATLAB
E = expm([As Bs; zeros(1, 2*n+1)] * dt); Ad = E(1:2*n, 1:2*n); Bd = E(1:2*n, end);
Y = zeros(N, 2*n); y = zeros(2*n, 1);
for k = 1:N, Y(k,:) = y.'; y = Ad * y + Bd * u(k); end                  % exact ZOH recursion
X = Y(:, 1:n); Xn = X + 0.01 * std(X, 1) .* randn(size(X));
cases = {'central', 1, 0; 'lagged', 2, 5};
for c = 1:2
    R = sod_decomp(Xn, [], struct('dt', dt, 'h', cases{c,2}, 'lag', cases{c,3}, 'scale', true));
    f = R.freq; f(isnan(f)) = 0; wc = sod_cd_correct(f, cases{c,2} * dt);
    [idx, pk] = sod_select_config(R.Q, wc.^2, dt, 10, struct('Phi', R.Phi));
    fprintf('%-8s SOD freq %s  kept %s\n', cases{c,1}, sprintf('%.3f ', wc), mat2str(idx));
end
fprintf('true omega        %s\n', sprintf('%.3f ', w));
