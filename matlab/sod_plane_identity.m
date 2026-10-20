function trL = sod_plane_identity(Q, dt)
%SOD_PLANE_IDENTITY  tr L from the end states: (q_T' G^-1 q_T - q_0' G^-1 q_0)/(2 dt), G = sum q q'.
%   Exact for the chord derivative; with exact qdot the relative difference is O((|s| dt)^2 / zeta).
G = Q.' * Q; trL = (Q(end, :) / G * Q(end, :).' - Q(1, :) / G * Q(1, :).') / 2 / dt;
end
