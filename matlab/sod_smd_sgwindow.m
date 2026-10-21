function win = sod_smd_sgwindow(kc, h, polyorder)
%SOD_SMD_SGWINDOW  Smallest odd Savitzky-Golay window whose MEASURED half-gain cutoff is at or below kc.
%   win = SOD_SMD_SGWINDOW(kc, h, polyorder)   kc in rad per length unit, h the grid spacing.
if nargin < 3, polyorder = 3; end
fc = kc * h / (2 * pi);
if fc >= 0.5, error('kc is above the grid Nyquist wavenumber pi/h'); end
win = polyorder + 2 + mod(polyorder + 1, 2);
while cut(win, polyorder) > fc
    win = win + 2;
    if win > 4097, error('kc too low for any reasonable Savitzky-Golay window'); end
end
end

function fc = cut(win, po)
mh = (win - 1) / 2; z = (-mh:mh).'; Pi = pinv(z .^ (0:po)); c = Pi(1, :);
nfft = 2^15; H = abs(fft(c, nfft)); H = H(1:nfft/2+1); f = (0:nfft/2) / nfft;
fc = f(find(H < 0.5, 1));
end
