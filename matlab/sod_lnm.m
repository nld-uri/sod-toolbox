function P = sod_lnm(M, K, r)
%SOD_LNM  Lowest r linear normal modes, mass-normalized.
[P, w2] = eig(K, M, 'chol', 'vector'); [~, o] = sort(w2); P = P(:, o(1:r));
P = P ./ sqrt(sum(P .* (M * P), 1));
end
