function B = sod_boundary(X, V, T)
%SOD_BOUNDARY  Integration-by-parts boundary term (x_T v_T' - x_0 v_0')/T of a finite window.
B = (X(end, :).' * V(end, :) - X(1, :).' * V(1, :)) / T;
end
