function K = sod_smd_kc_scan(W, G, operator, method, kcs, opts)
%SOD_SMD_KC_SCAN  SMD wavenumbers for each cutoff in kcs (rows): keep a cutoff where they do not move.
if nargin < 6, opts = struct(); end
out = cell(numel(kcs), 1);
for i = 1:numel(kcs), opts.kc = kcs(i); R = sod_smd(W, G, operator, method, opts); out{i} = R.k(:).'; end
n = min(cellfun(@numel, out)); K = zeros(numel(kcs), n);
for i = 1:numel(kcs), K(i, :) = out{i}(1:n); end
end
