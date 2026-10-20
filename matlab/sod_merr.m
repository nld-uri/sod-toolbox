function e = sod_merr(Xr, X, M)
%SOD_MERR  Mass-weighted relative error sqrt(sum D M D / sum X M X), rows = samples.
D = Xr - X; e = sqrt(sum(sum((D * M) .* D)) / sum(sum((X * M) .* X)));
end
