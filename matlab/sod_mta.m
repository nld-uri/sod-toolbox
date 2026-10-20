function T = sod_mta(M, K, B, r)
%SOD_MTA  Modal truncation augmentation: r - m LNMs plus the m static load vectors K\B, M-orthonormalized.
m = size(B, 2); if r <= m, T = []; return; end
Q = [sod_lnm(M, K, r - m), K \ B]; L = chol(Q.' * M * Q, 'lower'); T = Q / L.';
end
