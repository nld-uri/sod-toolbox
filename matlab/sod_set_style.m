function C = sod_set_style(palette)
%SOD_SET_STYLE  LaTeX interpreters and a categorical palette ('matlab' default ColorOrder or 'nld').
%   MATLAB palette: dark green #055432 and plum #67355C replace green and purple (colour-vision safety).
if nargin < 1, palette = "matlab"; end
if palette == "nld"
    C = hex2rgb(["#1F5A85" "#B4622A" "#588A75" "#CA863E" "#774860"]);
else
    C = hex2rgb(["#0072BD" "#D95319" "#EDB120" "#67355C" "#055432" "#4DBEEE" "#A2142F"]);
end
set(groot, 'defaultAxesColorOrder', C, 'defaultTextInterpreter', 'latex', ...
    'defaultAxesTickLabelInterpreter', 'latex', 'defaultLegendInterpreter', 'latex');
end

function c = hex2rgb(h)
c = zeros(numel(h), 3);
for i = 1:numel(h), s = char(h(i)); c(i,:) = sscanf(s(2:end), '%2x%2x%2x').' / 255; end
end
