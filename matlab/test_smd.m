%TEST_SMD  Checks of the SMD functions: analytic properties, failure controls, and agreement with the Python toolbox
%   (reference values in smd_reference.mat, written by python/tests/make_smd_reference.py). Run from this folder.
clear; rows = [];
ref = load('smd_reference.mat');
names = {'b_fd', 'b_wh', 'b_sg', 'u_fd', 'u_wh', 'm_fd', 'm_lap', 'm_hes', 'm_sgg', 'h_wh', 'h_fd'};
Gb = sod_smd_grid(ref.xb); Gu = sod_smd_grid(ref.xn); Gm = sod_smd_grid(ref.xm, ref.ym);
Gh = sod_smd_grid(ref.xm, ref.ym, ~ref.hole);
spec = struct( ...
    'b_fd', {{ref.Wb, Gb, 'second', 'fd', struct()}}, ...
    'b_wh', {{ref.Wn, Gb, 'second', 'whittaker', struct('kc', 3 * ref.bl(end), 'rank', 3)}}, ...
    'b_sg', {{ref.Wn, Gb, 'second', 'savgol', struct('kc', 8 * ref.bl(end), 'rank', 3)}}, ...
    'u_fd', {{ref.Wu, Gu, 'grad', 'fd', struct()}}, ...
    'u_wh', {{ref.Wu, Gu, 'grad', 'whittaker', struct('kc', 30)}}, ...
    'm_fd', {{ref.Wm, Gm, 'grad', 'fd', struct()}}, ...
    'm_lap', {{ref.Wm, Gm, 'laplacian', 'fd', struct()}}, ...
    'm_hes', {{ref.Wm, Gm, 'second', 'savgol', struct('window', 9)}}, ...
    'm_sgg', {{ref.Wm, Gm, 'grad', 'savgol', struct('kc', 40)}}, ...
    'h_wh', {{ref.Wm, Gh, 'grad', 'whittaker', struct('kc', 60)}}, ...
    'h_fd', {{ref.Wm, Gh, 'grad', 'fd', struct()}});
ok = true;
for i = 1:numel(names)
    c = spec.(names{i});
    R = sod_smd(c{1}, c{2}, c{3}, c{4}, c{5});
    kp = ref.([names{i} '_k'])(:); Pp = ref.([names{i} '_Phi']);
    dk = max(abs(R.k(:) ./ kp - 1));
    Pm = R.Phi; Pm(isnan(Pm)) = 0;
    dm = 0;
    for j = 1:numel(kp), a = Pm(:, j); b = Pp(:, j); dm = max(dm, 1 - (a.' * b)^2 / ((a.' * a) * (b.' * b))); end
    % 1D Whittaker cases are ill-conditioned (condition ~1e12-1e13, see SOD_SMD_DERIVS): looser agreement there
    tk = 1e-8; tm = 1e-10;
    if any(strcmp(names{i}, {'b_wh', 'u_wh'})), tk = 1e-4; tm = 1e-8; end
    v1 = dk <= tk; v2 = dm <= tm; ok = ok && v1 && v2;
    fprintf('%s  %-58s %10.2e  <= %.0e\n', pf(v1), ['Python agreement, ' names{i} ': SMD wavenumbers'], dk, tk);
    fprintf('%s  %-58s %10.2e  <= %.0e\n', pf(v2), ['Python agreement, ' names{i} ': modes (1 - MAC)'], dm, tm);
    if isfield(ref, [names{i} '_G1'])
        W0 = c{1} - mean(c{1}, 1, 'omitnan');
        D = sod_smd_derivs(W0, c{2}, c{3}, c{4}, c{5});
        g = D.Gk{1}(1:5, :); g(isnan(g)) = 0; gp = ref.([names{i} '_G1']);
        dg = max(abs(g(:) - gp(:))) / max(abs(gp(:)));
        v3 = dg <= 1e-9 && isequal(logical(D.valid(:)), logical(ref.([names{i} '_valid'])(:)));
        ok = ok && v3;
        fprintf('%s  %-58s %10.2e  <= 1e-09\n', pf(v3), ['Python agreement, ' names{i} ': derivative field and valid set'], dg);
    end
end
% analytic checks and failure controls (beam)
bl = ref.bl(:); x = ref.xb;
R = sod_smd(ref.Wb, Gb, 'second', 'fd'); e = max(abs(R.k(:) ./ bl - 1));
v = e <= 1e-3; ok = ok && v; fprintf('%s  %-58s %10.2e  <= 1e-03\n', pf(v), 'cantilever, second, fd: k_j = beta_j', e);
Rs = sod_smd(ref.Wb(randperm(size(ref.Wb, 1)), :), Gb, 'second', 'fd'); e = max(abs(Rs.mu ./ R.mu - 1));
v = e <= 1e-10; ok = ok && v; fprintf('%s  %-58s %10.2e  <= 1e-10\n', pf(v), 'snapshot order irrelevant (shuffled)', e);
Rf = sod_smd(ref.Wn, Gb, 'second', 'fd', struct('rank', 3)); e = max(abs(Rf.k(:) ./ bl - 1));
v = e > 5e-2; ok = ok && v; fprintf('%s  %-58s %10.2e  >  5e-02\n', pf(v), 'FC 0.5% noise, plain fd second derivative: k error', e);
K = sod_smd_kc_scan(ref.Wn, Gb, 'second', 'whittaker', [3 6] * bl(end), struct('rank', 3));
e = max(abs(K(1, :).' ./ bl - 1)); v = e <= 1e-2; ok = ok && v;
fprintf('%s  %-58s %10.2e  <= 1e-02\n', pf(v), '0.5% noise, whittaker kc = 3 beta_3: k error', e);
e = max(abs(K(2, :) ./ K(1, :) - 1)); v = e <= 1e-2; ok = ok && v;
fprintf('%s  %-58s %10.2e  <= 1e-02\n', pf(v), '0.5% noise, whittaker kc 3 -> 6 beta_3 moves k by', e);
if ok, fprintf('\nALL CHECKS PASSED\n'); else, fprintf('\nSOME CHECKS FAILED\n'); end

function s = pf(v), if v, s = 'PASS'; else, s = 'FAIL'; end, end
function s = ifelse(c, a, b), if c, s = a; else, s = b; end, end
