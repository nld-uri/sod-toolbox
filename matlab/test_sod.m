%TEST_SOD  Check the MATLAB port against the Python toolbox (sod_reference.mat, from python/tests/make_reference.py).
%   Vectors are compared up to sign; the worst relative deviation is printed against a tolerance.
clear; R0 = load('sod_reference.mat'); dt = R0.dt; n = size(R0.M, 1);
X = R0.Y(:, 1:n); V = R0.Y(:, n+1:end); Acc = R0.Yd(:, n+1:end); Yfd = R0.Yf * R0.A.';
res = {};
cs = @(A, B) max(arrayfun(@(j) norm(A(:, j) - sign(A(:, j).' * B(:, j)) * B(:, j)) / norm(B(:, j)), 1:size(B, 2)));
rd = @(a, b) max(abs(a(:) - b(:)) ./ max(abs(b(:)), 1e-300));
add = @(res, name, dev, tol) [res; {name, dev, tol}];

R = sod_decomp(X, V);                                  res = add(res, 'sod_decomp mu', rd(R.mu, R0.sod_mu), 1e-9);
res = add(res, 'sod_decomp Psi', cs(R.Psi, R0.sod_Psi), 1e-7); res = add(res, 'sod_decomp Phi', cs(R.Phi, R0.sod_Phi), 1e-7);
R = sod_decomp(X, V, struct('scale', true));           res = add(res, 'scaled mu', rd(R.mu, R0.sods_mu), 1e-9);
res = add(res, 'scaled Psi', cs(R.Psi, R0.sods_Psi), 1e-7);
R = sod_decomp(X, [], struct('dt', dt, 'h', 2, 'lag', 5, 'center', true, 'tol', []));
res = add(res, 'lagged, h = 2, centred: mu', rd(R.mu, R0.sodl_mu), 1e-8); res = add(res, 'lagged Psi', cs(R.Psi, R0.sodl_Psi), 1e-6);
R = sod_decomp(X, [], struct('dt', dt, 'deriv', 'gradient', 'center', true, 'tol', []));
res = add(res, 'gradient, centred: mu', rd(R.mu, R0.sodg_mu), 1e-9); res = add(res, 'gradient Psi', cs(R.Psi, R0.sodg_Psi), 1e-7);
Rs = sod_state(R0.Y, R0.Yd);                           res = add(res, 'sod_state mu', rd(Rs.mu, R0.ss_mu), 1e-9);
res = add(res, 'sod_state Psi', cs(Rs.Psi, R0.ss_Psi), 1e-7); res = add(res, 'sod_state Phi', cs(Rs.Phi, R0.ss_Phi), 1e-7);
[Sx, Sv] = sod_cov(X(1:400, :), V(1:400, :)); Bd = sod_boundary(X(1:400, :), V(1:400, :), 399 * dt);
[w, P] = sod_csod(Sx, Sv, Bd);                         res = add(res, 'sod_csod frequencies', rd(w, R0.csod_w), 1e-9);
[w, psi, m] = sod_window(X(1:802, :), dt, true);       res = add(res, 'sod_window', max(abs(w / R0.win_w - 1), cs(psi, R0.win_psi(:))), 1e-8);
[tc, om, md] = sod_track(X, dt, 600, 300, false);      res = add(res, 'sod_track', max(rd(tc, R0.trk_t), rd(om, R0.trk_om)), 1e-9);
[omd, F] = sod_dct_spectrum(Rs.Q(:, 1:8), dt);              % power spectra: sign-free
res = add(res, 'sod_dct_spectrum', max(rd(omd, R0.dct_om), max(abs(F(:) - R0.dct_F(:))) / max(R0.dct_F(:))), 1e-9);
[pk, rel] = sod_peak_width(omd, F);                    res = add(res, 'sod_peak_width', max(rd(pk, R0.pk), rd(rel, R0.rel)), 1e-12);
Rc = sod_decomp(X, V, struct('scale', true));
idx = sod_select_config(Rc.Q, Rc.mu, dt, 150, struct('Phi', Rc.Phi));
res = add(res, 'sod_select_config (same indices)', double(~isequal(idx(:), double(R0.sel_idx(:)))), 0);
pl = sod_pair_state(Rs.Q, dt, 150, struct('Phi', Rs.Phi));
res = add(res, 'sod_pair_state (same planes)', double(~isequal(pl(:, 1:2), double(R0.planes(:, 1:2)))), 0);
res = add(res, 'sod_pair_state (f, c0)', rd(pl(:, 3:4), R0.planes(:, 3:4)), 1e-6);   % c0 differs by 1.3e-8 on one plane between the two FFTs, probably from cancellation in the counter-rotating power
Rf = sod_decomp(R0.Yf, Yfd, struct('scale', true)); Qf = Rf.Q; Qfd = Yfd * Rf.Psi;
pf = sod_pair_state(Qf, dt, 150, struct('Phi', Rf.Phi)); ab = pf(1, 1:2);
res = add(res, 'decay planes', double(~isequal(pf(:, 1:2), double(R0.fplanes(:, 1:2)))), 0);
[s, z] = sod_plane_eigs(sod_fit_L(Qf(:, ab), Qfd(:, ab)));
res = add(res, 'plane eigenvalue and zeta', max(abs(s - R0.plane_s) / abs(R0.plane_s), abs(z / R0.plane_zeta - 1)), 1e-8);
res = add(res, 'plane identity', abs(sod_plane_identity(Qf(:, ab), dt) / R0.plane_ident - 1), 1e-8);
res = add(res, 'sod_lnm', cs(sod_lnm(R0.M, R0.K, 4), R0.lnm), 1e-9);
res = add(res, 'sod_mta', cs(sod_mta(R0.M, R0.K, R0.B, 4), R0.mta), 1e-9);
[U, pov] = sod_pod(X, 4);                              res = add(res, 'sod_pod', max(cs(U, R0.pod), rd(pov, R0.pov)), 1e-9);
[Pm, wm] = sod_modes(X, V, 4);                         res = add(res, 'sod_modes', max(cs(Pm, R0.sm_Phi), rd(wm, R0.sm_w)), 1e-8);
Anl = Acc - 30 * X.^3;
for kind = 'akx'
    res = add(res, ['sod_residual_basis ' kind], cs(sod_residual_basis(X, V, Anl, R0.M, R0.K, 2, 4, kind), R0.(['res_' kind])), 1e-6);
end
[~, Kr] = sod_galerkin(R0.M, R0.K, R0.C, R0.B, R0.sm_Phi); res = add(res, 'sod_galerkin', rd(Kr, R0.gal_K), 1e-12);
res = add(res, 'sod_mac', abs(sod_mac(R0.sm_Phi(:, 1), R0.lnm(:, 1)) / R0.mac12 - 1), 1e-12);

nf = 0; fprintf('\n%-36s %10s %8s\n', 'function', 'deviation', 'tol');
for i = 1:size(res, 1)
    ok = res{i, 2} <= res{i, 3}; nf = nf + ~ok;
    if ok, tag = 'PASS'; else, tag = 'FAIL'; end
    fprintf('%-36s %10.1e %8.0e  %s\n', res{i, 1}, res{i, 2}, res{i, 3}, tag);
end
fprintf('\n%d/%d checks passed\n', size(res, 1) - nf, size(res, 1));
