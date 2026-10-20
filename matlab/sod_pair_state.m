function planes = sod_pair_state(Q, dt, wmax, opts)
%SOD_PAIR_STATE  Pair state-SOD coordinates into modal planes by frequency and circularity (never by smoothness).
%   Candidates: DCT peak below wmax. Pair (a,b) if peaks agree within fgate*max(ftol, widths); score by the
%   one-sidedness c0 of the spectrum of q_a + i q_b; keep if c0 >= ctol. Rescue (needs opts.Phi): keep a pair with
%   c0 in [rescue(1), ctol) if its banded one-sidedness is >= rescue(2) and it carries >= rescue(3) of the energy.
%   Greedy by c0, one plane per frequency. Returns rows [a b f c0] (a, b 1-based) ordered by plane frequency f.
%   opts: ctol 0.85, ftol 0.05, fgate 4, Phi [], rescue [0.75 0.95 1e-2], band 0.3.
if nargin < 4, opts = struct(); end
d = struct('ctol', 0.85, 'ftol', 0.05, 'fgate', 4.0, 'Phi', [], 'rescue', [0.75 0.95 1e-2], 'band', 0.3);
f = fieldnames(d);
for i = 1:numel(f), if ~isfield(opts, f{i}), opts.(f{i}) = d.(f{i}); end, end
N = size(Q, 1);
[om, F] = sod_dct_spectrum(Q, dt); [pk, rel] = sod_peak_width(om, F);
kk = (0:N-1).'; kk(kk >= ceil(N/2)) = kk(kk >= ceil(N/2)) - N; om2 = 2*pi*kk / (N*dt);   % numpy fftfreq
en = [];
if ~isempty(opts.Phi), en = sum(opts.Phi.^2, 1) .* var(Q, 1, 1); en = en / sum(en); end
cand = find(pk < wmax);
Qn = (Q(:, cand) - mean(Q(:, cand), 1)) ./ std(Q(:, cand), 1, 1); FQ = fft(Qn);
pos = om2 > 0; neg = om2 < 0; S = zeros(0, 4);
for x = 1:numel(cand)
    a = cand(x);
    for y = x+1:numel(cand)
        b = cand(y);
        if abs(pk(a) / pk(b) - 1) < opts.fgate * max([opts.ftol, rel(a), rel(b)])
            Fz = abs(FQ(:, x) + 1i * FQ(:, y)).^2; Pp = sum(Fz(pos)); Pm = sum(Fz(neg));
            if Pp >= Pm, side = pos; else, side = neg; end
            c0 = max(Pp, Pm) / (Pp + Pm); o2 = om2(side); [~, j] = max(Fz(side)); fr = abs(o2(j));
            ok = c0 >= opts.ctol;
            if ~ok && ~isempty(en) && c0 >= opts.rescue(1) && fr > 0 && en(a) + en(b) >= opts.rescue(3)
                bd = abs(abs(om2) / fr - 1) < opts.band; P2 = sum(Fz(pos & bd)); M2 = sum(Fz(neg & bd));
                ok = max(P2, M2) / (P2 + M2) >= opts.rescue(2);
            end
            if ok, S(end+1, :) = [c0, a, b, fr]; end %#ok<AGROW>
        end
    end
end
S = sortrows(S, [-1 -2 -3 -4]); used = []; planes = zeros(0, 4);
for r = 1:size(S, 1)
    c = S(r, 1); a = S(r, 2); b = S(r, 3); fr = S(r, 4);
    if any(used == a) || any(used == b), continue; end
    if ~isempty(planes) && any(abs(fr ./ planes(:, 3) - 1) < 2 * opts.ftol), continue; end
    used = [used a b]; planes(end+1, :) = [a b fr c]; %#ok<AGROW>
end
planes = sortrows(planes, 3);
end
