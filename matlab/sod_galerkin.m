function [Mr, Kr, Cr, Br] = sod_galerkin(M, K, C, B, Phi)
%SOD_GALERKIN  Reduced matrices Phi'(M, K, C)Phi and Phi'B.
Mr = Phi.' * M * Phi; Kr = Phi.' * K * Phi; Cr = Phi.' * C * Phi; Br = Phi.' * B;
end
