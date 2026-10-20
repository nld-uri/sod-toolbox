function [Sx, Sv] = sod_cov(X, V, lag)
%SOD_COV  Sample covariances divided by N; lag > 0 gives symmetrized lag-lag covariances (remove white noise).
if nargin < 3 || lag == 0
    Sx = X.' * X / size(X, 1); Sv = V.' * V / size(V, 1);
else
    n = size(X, 1) - lag;
    Sx = X(1:n, :).' * X(1+lag:end, :) / n; Sx = (Sx + Sx.') / 2;
    Sv = V(1:n, :).' * V(1+lag:end, :) / n; Sv = (Sv + Sv.') / 2;
end
end
