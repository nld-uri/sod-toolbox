function R = sod_state(Y, Yd, k, tol)
%SOD_STATE  State-space SOD on y = (x, xdot), ydot = (xdot, xddot); unit-RMS channels for the rank test.
%   For uncorrelated modal planes each complex mode s gives a pair with mu_a mu_b = |s|^4.
if nargin < 3, k = []; end
if nargin < 4, tol = 1e-10; end
R = sod_decomp(Y, Yd, struct('scale', true, 'tol', tol, 'k', k));
end
