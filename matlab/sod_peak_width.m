function [pk, rel] = sod_peak_width(om, F)
%SOD_PEAK_WIDTH  Peak frequency and relative half-power width of each column's periodogram.
[~, k] = max(F, [], 1); pk = om(k).'; rel = zeros(1, size(F, 2)); n = size(F, 1);
for j = 1:size(F, 2)
    f = F(:, j); h = f(k(j)) / 2; lo = k(j); hi = k(j);
    while lo > 1 && f(lo) > h, lo = lo - 1; end
    while hi < n && f(hi) > h, hi = hi + 1; end
    rel(j) = (om(hi) - om(lo)) / max(pk(j), 1e-12);
end
end
