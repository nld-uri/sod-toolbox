function G = sod_smd_grid(x, y, mask)
%SOD_SMD_GRID  Sampling geometry for smooth mode decomposition (SMD) of full-field data.
%   G = SOD_SMD_GRID(x)            points along a line (strictly increasing, non-uniform allowed)
%   G = SOD_SMD_GRID(x, y)         regular grid with x (columns) and y (rows) axes; fields are ny x nx
%   G = SOD_SMD_GRID(x, y, mask)   mask (ny x nx, or n for a line) true where the field is measured
%   Fields: shape, x, y, mask, w (trapezoidal quadrature weights, zero outside the mask).
%   Snapshots are passed to the SMD functions as nt x n (line) or nt x ny x nx (grid) arrays.
if nargin < 2, y = []; end
x = x(:).';
if any(diff(x) <= 0) || (~isempty(y) && any(diff(y(:)) <= 0)), error('grid coordinates must increase strictly'); end
if isempty(y)
    G.shape = numel(x); G.y = [];
    w = trapw(x);
else
    y = y(:).'; G.shape = [numel(y), numel(x)]; G.y = y;
    w = trapw(y).' * trapw(x);
end
G.x = x;
if nargin < 3 || isempty(mask), mask = true(size(w)); end
G.mask = reshape(logical(mask), size(w));
G.w = w .* G.mask;
end

function w = trapw(x)
d = diff(x); w = zeros(size(x));
w(1:end-1) = w(1:end-1) + d / 2; w(2:end) = w(2:end) + d / 2;
end
