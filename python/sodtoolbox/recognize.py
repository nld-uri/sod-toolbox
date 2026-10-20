"""Recognising modes and modal planes among SOD coordinates, never by smoothness.

Rules: modes are identified, paired and ordered by their frequency or by the circularity of a pair,
not by the SOD value. Frequencies are read from the DCT-II periodogram (no wrap-around for non-periodic records),
smoothed over a constant relative bandwidth. The spectral tolerance is damping-aware: the larger of 5% and the
coordinate's relative half-power width, capped at 15%.

select_config : configuration SOD, spectral consistency of each coordinate with its SOD frequency
pair_state    : state SOD, pairs of coordinates forming one modal plane (one-sidedness of q_a + i q_b),
                with the energy-gated banded rescue for planes contaminated by a bend or a neighbour
"""
import numpy as np
from scipy.fft import dct

__all__ = ['dct_spectrum', 'peak_and_width', 'cd_correct', 'select_config', 'onesided', 'pair_state',
           'rom_order_config', 'rom_order_state']

def dct_spectrum(Q, dt, b=0.02):
    """DCT-II periodogram, smoothed over a constant relative bandwidth +-b (constant-Q moving average). A raw
       periodogram of a randomly forced coordinate has single-bin spikes; its arg-max is not the resonance and its
       bin-level half-power width misses the damping width 2 zeta omega."""
    F = dct(Q - Q.mean(0), type=2, axis=0, norm='ortho')**2; om = np.pi * np.arange(len(Q)) / (len(Q) * dt)
    if b:
        k = np.arange(len(F)); lo = np.floor(k * (1 - b)).astype(int); hi = np.minimum(np.ceil(k * (1 + b)).astype(int), len(F) - 1)
        cs = np.vstack([np.zeros((1, F.shape[1])), np.cumsum(F, axis=0)]); F = (cs[hi + 1] - cs[lo]) / (hi - lo + 1)[:, None]
    return om, F

def peak_and_width(om, F):
    """peak frequency and relative half-power width of each column's DCT periodogram"""
    k = np.argmax(F, axis=0); pk = om[k]; rel = np.zeros(F.shape[1])
    for j in range(F.shape[1]):
        f = F[:, j]; h = f[k[j]] / 2; lo = k[j]; hi = k[j]
        while lo > 0 and f[lo] > h: lo -= 1
        while hi < len(f) - 1 and f[hi] > h: hi += 1
        rel[j] = (om[hi] - om[lo]) / max(pk[j], 1e-12)
    return pk, rel

def cd_correct(w, dt):
    """invert the central-difference bias w_cd = sin(w dt)/dt; applied only when velocities came from central differences"""
    return np.arcsin(np.clip(w * dt, -1, 1)) / dt

def select_config(Q, mu, dt, wmax, cd=False, ftol=0.05, Phi=None, wcap=0.15):
    """ID selection, configuration SOD: keep coordinate i if its spectral peak lies within max(ftol, half-power width)
       of its SOD frequency sqrt(mu_i) (bias-corrected for central differences) and below wmax. The half-power width
       is capped at wcap (default 0.15, i.e. damping ratios up to 7.5%): without the cap a broadband or drifting
       coordinate, whose spectrum peaks near zero frequency, has a width above 1 and passes the test (c25_width_diag.py). If several kept
       coordinates share one peak (within the same tolerance), a real mode can own only one: keep the one carrying
       the most response energy, ||phi_i||^2 var(q_i) (needs Phi). Returns kept indices ordered by peak frequency,
       peaks, and SOD frequencies."""
    om, F = dct_spectrum(Q, dt); pk, rel = peak_and_width(om, F)
    wsod = np.sqrt(np.clip(mu, 0, None)); wsod = cd_correct(wsod, dt) if cd else wsod
    tol = np.maximum(ftol, np.minimum(rel, wcap)); ok = (abs(pk / np.maximum(wsod, 1e-12) - 1) < tol) & (pk < wmax)
    idx = list(np.where(ok)[0][np.argsort(pk[ok])])
    if Phi is not None and idx:
        en = (Phi**2).sum(0) * Q.var(0); keep = []; grp = [idx[0]]
        for i in idx[1:]:
            if abs(pk[i] / pk[grp[0]] - 1) < max(tol[i], tol[grp[0]]): grp.append(i)
            else: keep.append(max(grp, key=lambda j: en[j])); grp = [i]
        keep.append(max(grp, key=lambda j: en[j])); idx = keep
    return np.array(idx, int), pk, wsod

def onesided(qa, qb, om2):
    qa = (qa - qa.mean()) / qa.std(); qb = (qb - qb.mean()) / qb.std()
    F = np.abs(np.fft.fft(qa + 1j * qb))**2; Pp, Pm = F[om2 > 0].sum(), F[om2 < 0].sum()
    side = om2 > 0 if Pp >= Pm else om2 < 0
    return max(Pp, Pm) / (Pp + Pm), abs(om2[side][np.argmax(F[side])])

def pair_state(Q, dt, wmax, ctol=0.85, ftol=0.05, fgate=4.0, Phi=None, rescue=(0.75, 0.95, 1e-2), band=0.3):
    """ID pairing, state SOD: candidates are coordinates whose DCT peak is below wmax. Two coordinates may pair if their
       peaks agree within max(ftol, their half-power widths); among those, pairs are formed greedily by largest
       one-sidedness of z = q_a + i q_b, and kept if it exceeds ctol (0.85: modal planes gave 0.89-0.99, other pairs
       0.51-0.75 in c24_mode1_diag.py). The smoothness ranking is never used.
       The frequency gate is loose (fgate times the tolerance): peaks of randomly forced coordinates scatter by more
       than their measured widths, and circularity alone already rejects pairs from different planes.
       Rescue (needs Phi): a pair whose broadband one-sidedness c0 lies in [rescue[0], ctol) is also kept if its banded
       one-sidedness c1 (power within +-band of the pair's one-sided peak) is at least rescue[1] and the pair carries at
       least rescue[2] of the response energy ||phi||^2 var(q). This recovers a strong plane that a bend or another
       mode contaminates (COST beam at 30 N: mode-2 plane c0 = 0.836, c1 = 0.970; n11b_banded_circ.py), while weak
       narrowband noise pairs, which the banded test alone admits, stay out (c26_circ_rules.py). A banded test alone
       admitted up to seven times more non-modal pairs on noisy linear records.
       Returns a list of (a, b, plane frequency, broadband one-sidedness), ordered by plane frequency."""
    om, F = dct_spectrum(Q, dt); pk, rel = peak_and_width(om, F); om2 = 2 * np.pi * np.fft.fftfreq(len(Q), dt)
    en = None
    if Phi is not None: en = (Phi**2).sum(0) * Q.var(0); en = en / en.sum()
    cand = [i for i in range(Q.shape[1]) if pk[i] < wmax]; scores = []
    Qn = (Q[:, cand] - Q[:, cand].mean(0)) / Q[:, cand].std(0); FQ = np.fft.fft(Qn, axis=0)   # FFT is linear: FFT(z) = FFT(q_a) + i FFT(q_b)
    pos, neg = om2 > 0, om2 < 0
    for x, a in enumerate(cand):
        for y in range(x + 1, len(cand)):
            b = cand[y]
            if abs(pk[a] / pk[b] - 1) < fgate * max(ftol, rel[a], rel[b]):
                Fz = np.abs(FQ[:, x] + 1j * FQ[:, y])**2; Pp, Pm = Fz[pos].sum(), Fz[neg].sum(); side = pos if Pp >= Pm else neg
                c0 = max(Pp, Pm) / (Pp + Pm); f = abs(om2[side][np.argmax(Fz[side])]); ok = c0 >= ctol
                if not ok and en is not None and c0 >= rescue[0] and f > 0 and en[a] + en[b] >= rescue[2]:
                    bd = abs(abs(om2) / f - 1) < band; P2, M2 = Fz[pos & bd].sum(), Fz[neg & bd].sum(); ok = max(P2, M2) / (P2 + M2) >= rescue[1]
                if ok: scores.append((c0, a, b, f))
    used = set(); planes = []
    for c, a, b, f in sorted(scores, reverse=True):
        if a in used or b in used: continue
        if any(abs(f / p[2] - 1) < 2 * ftol for p in planes): continue      # one plane per frequency: keep the more circular
        used |= {a, b}; planes.append((a, b, f, c))
    return sorted(planes, key=lambda p: p[2])

def rom_order_config(idx_id, k_all):
    """ROM ordering, configuration: ID-selected modes by frequency, then the remaining coordinates by smoothness"""
    rest = [i for i in range(k_all) if i not in set(idx_id)]; return list(idx_id) + rest

def rom_order_state(planes, k_all):
    idx = [i for p in planes for i in p[:2]]; rest = [i for i in range(k_all) if i not in set(idx)]; return idx + rest
