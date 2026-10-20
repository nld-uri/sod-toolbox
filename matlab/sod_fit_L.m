function L = sod_fit_L(Q, Qd)
%SOD_FIT_L  Least-squares plane (or subspace) dynamics L = <qdot q'><q q'>^{-1}.
L = (Qd.' * Q) / (Q.' * Q);
end
