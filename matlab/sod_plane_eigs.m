function [s, zeta] = sod_plane_eigs(L)
%SOD_PLANE_EIGS  Complex eigenvalue s = sigma + i omega_d of a 2x2 plane fit and zeta = -sigma/|s|.
sig = trace(L) / 2; wn = sqrt(max(det(L), 0));
s = complex(sig, sqrt(max(wn^2 - sig^2, 0)));
if wn > 0, zeta = -sig / wn; else, zeta = NaN; end
end
