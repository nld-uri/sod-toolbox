function [idx, pk, wsod] = sod_select_config(Q, mu, dt, wmax, opts)
%SOD_SELECT_CONFIG  Identification selection for configuration SOD (never by smoothness).
%   Keep coordinate i if its DCT peak lies within max(ftol, min(half-power width, wcap)) of sqrt(mu_i) and below wmax.
%   Coordinates sharing one peak: keep the one with most response energy ||phi_i||^2 var(q_i) (needs opts.Phi).
%   opts: cd (false: central-difference bias correction of sqrt(mu)), ftol (0.05), wcap (0.15), Phi ([]).
%   Returns kept indices (1-based) ordered by peak frequency, peaks and SOD frequencies.
if nargin < 5, opts = struct(); end
d = struct('cd', false, 'ftol', 0.05, 'wcap', 0.15, 'Phi', []); f = fieldnames(d);
for i = 1:numel(f), if ~isfield(opts, f{i}), opts.(f{i}) = d.(f{i}); end, end
[om, F] = sod_dct_spectrum(Q, dt); [pk, rel] = sod_peak_width(om, F);
wsod = sqrt(max(mu(:).', 0));
if opts.cd, wsod = asin(min(max(wsod * dt, -1), 1)) / dt; end
tol = max(opts.ftol, min(rel, opts.wcap));
ok = (abs(pk ./ max(wsod, 1e-12) - 1) < tol) & (pk < wmax);
cand = find(ok); [~, o] = sort(pk(cand)); idx = cand(o);
if ~isempty(opts.Phi) && ~isempty(idx)
    en = sum(opts.Phi.^2, 1) .* var(Q, 1, 1); keep = []; grp = idx(1);
    for i = idx(2:end)
        if abs(pk(i) / pk(grp(1)) - 1) < max(tol(i), tol(grp(1)))
            grp(end+1) = i; %#ok<AGROW>
        else
            [~, j] = max(en(grp)); keep(end+1) = grp(j); grp = i; %#ok<AGROW>
        end
    end
    [~, j] = max(en(grp)); keep(end+1) = grp(j); idx = keep;
end
end
