function [w, P] = sod_csod(Sx, Sv, Bd)
%SOD_CSOD  Boundary-corrected fit A' = Sigma_x^{-1}(Sigma_v - B); exact for undamped linear systems on any window.
%   Not SOD (non-symmetric). Frequencies ascending and right eigenvectors.
[P, mu] = eig(Sx \ (Sv - Bd), 'vector');
[~, o] = sort(real(mu)); w = sqrt(real(mu(o))); P = real(P(:, o));
end
