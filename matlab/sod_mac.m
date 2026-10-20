function m = sod_mac(a, b, M)
%SOD_MAC  Modal assurance criterion (M-weighted if M is given).
if nargin < 3, m = (a.' * b)^2 / ((a.' * a) * (b.' * b));
else, m = (a.' * M * b)^2 / ((a.' * M * a) * (b.' * M * b)); end
end
