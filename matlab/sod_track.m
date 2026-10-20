function [tc, om, modes] = sod_track(X, dt, nwin, step, boundary)
%SOD_TRACK  Sliding-window SOD of the smoothest coordinate (e.g. the backbone of a decaying NNM).
%   Window nwin samples, shifted by step (default floor(nwin/2)). Times measured from t = 0 at sample 1.
if nargin < 4 || isempty(step), step = max(1, floor(nwin / 2)); end
if nargin < 5, boundary = false; end
i0s = 0:step:(size(X, 1) - nwin - 2);
tc = zeros(numel(i0s), 1); om = tc; modes = zeros(numel(i0s), size(X, 2));
for j = 1:numel(i0s)
    i0 = i0s(j);
    [om(j), ~, m] = sod_window(X(i0+1:i0+nwin+2, :), dt, boundary);
    tc(j) = (i0 + 1 + nwin / 2) * dt; modes(j, :) = m.';
end
end
