function [w, psi, mode1] = sod_window(Xw, dt, boundary)
%SOD_WINDOW  SOD of one window by central differences: frequency, covector and smooth mode of the smoothest coordinate.
%   boundary = true subtracts sym(B) from Sigma_v (leading finite-window correction for undamped motion).
if nargin < 3, boundary = false; end
xc = Xw(2:end-1, :); vd = (Xw(3:end, :) - Xw(1:end-2, :)) / (2*dt); Nw = size(xc, 1);
Sx = xc.' * xc / Nw; Sv = vd.' * vd / Nw;
if boundary
    Bd = sod_boundary(xc, vd, (Nw - 1) * dt); Sv = Sv - (Bd + Bd.') / 2;
end
[mu, P] = sod_pencil(Sx, Sv);
w = sqrt(max(mu(1), 0)); psi = P(:, 1); Mi = inv(P).'; mode1 = Mi(:, 1);
end
