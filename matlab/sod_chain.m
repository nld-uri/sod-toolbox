function [M, K, C] = sod_chain(n, zeta, m, k)
%SOD_CHAIN  n-DOF spring-mass chain fixed at the left end, with optional Rayleigh damping.
%   Defaults: masses linspace(1,2,n), springs linspace(3,1,n); zeta (0) at the first and last modes.
if nargin < 2, zeta = 0; end
if nargin < 3 || isempty(m), m = linspace(1, 2, n); end
if nargin < 4 || isempty(k), k = linspace(3, 1, n); end
M = diag(m); K = zeros(n);
for i = 1:n
    K(i,i) = K(i,i) + k(i);
    if i < n, K(i,i) = K(i,i) + k(i+1); K(i,i+1) = -k(i+1); K(i+1,i) = -k(i+1); end
end
C = zeros(n);
if zeta
    w = sqrt(sort(eig(K, M))); w1 = w(1); w2 = w(end);
    ab = [1/(2*w1) w1/2; 1/(2*w2) w2/2] \ [zeta; zeta];
    C = ab(1) * M + ab(2) * K;
end
end
