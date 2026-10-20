function [om, F] = sod_dct_spectrum(Q, dt, b)
%SOD_DCT_SPECTRUM  DCT-II (orthonormal) periodogram of each column, smoothed over a constant relative bandwidth +-b.
%   Base MATLAB: the DCT-II is computed from an FFT of the even extension. No wrap-around for non-periodic records.
if nargin < 3, b = 0.02; end
[N, m] = size(Q); Q = Q - mean(Q, 1);
Y = fft([Q; flipud(Q)]); k = (0:N-1).';
C = real(exp(-1i * pi * k / (2*N)) .* Y(1:N, :)) / 2 * sqrt(2/N);
C(1, :) = C(1, :) / sqrt(2);
F = C.^2; om = pi * k / (N * dt);
if b
    lo = floor(k * (1 - b)); hi = min(ceil(k * (1 + b)), N - 1);
    cs = [zeros(1, m); cumsum(F, 1)];
    F = (cs(hi + 2, :) - cs(lo + 1, :)) ./ (hi - lo + 1);
end
end
