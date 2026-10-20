function [Xo, V] = sod_deriv(X, dt, method, h)
%SOD_DERIV  Time derivative of the columns of X and the matching samples of X.
%   'central'  : (x_{j+h} - x_{j-h})/(2 h dt); drops h samples at each end
%   'gradient' : second-order interior, first-order ends (numpy.gradient convention); keeps every sample
%   For noisy measured records use a regularized estimator (kinematics toolbox) and pass V to sod_decomp.
if nargin < 3, method = 'central'; end
if nargin < 4, h = 1; end
switch method
    case 'central'
        Xo = X(h+1:end-h, :); V = (X(2*h+1:end, :) - X(1:end-2*h, :)) / (2*h*dt);
    case 'gradient'
        Xo = X; V = zeros(size(X));
        V(2:end-1, :) = (X(3:end, :) - X(1:end-2, :)) / (2*dt);
        V(1, :) = (X(2, :) - X(1, :)) / dt; V(end, :) = (X(end, :) - X(end-1, :)) / dt;
    otherwise
        error('method must be ''central'' or ''gradient''');
end
end
