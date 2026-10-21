function M = sod_smd_mode(R, i)
%SOD_SMD_MODE  SMD mode i of R (from SOD_SMD) on the grid: a row vector on a line, ny x nx on a grid.
G = R.grid;
if numel(G.shape) == 1, M = R.Phi(:, i).'; else, M = reshape(R.Phi(:, i), G.shape(2), G.shape(1)).'; end
end
