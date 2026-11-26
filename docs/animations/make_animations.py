"""LaTeX-typeset animations for the SOD manual, slide deck and web page.

    python make_animations.py [names...] [--frames N] [--still-only]

Writes <name>.mp4 (H.264, 1600x700, 30 fps), ../web/<name>.gif (800 px, 15 fps) and <name>_still.pdf (vector
still for the manual). Every computed quantity comes from the toolbox.
"""
import os, sys, argparse, subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, '..', 'web')
sys.path.insert(0, os.path.join(HERE, '..', '..', 'python'))
import sodtoolbox as st
from sodtoolbox import systems as sy, plotting as pl

INK, PAPER, REF = '#132433', '#F7F6F2', '#8A8578'
C = pl.PALETTES['nld']
pl.set_style('nld', size=13)
plt.rcParams.update({'figure.facecolor': PAPER, 'savefig.facecolor': PAPER, 'text.color': INK,
                     'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
                     'axes.edgecolor': INK, 'savefig.bbox': None})
FIG = (8, 3.5); DPI = 200


def save(fig, anim, name, nframes, still_frame, still_only):
    anim._func(still_frame)
    fig.savefig(os.path.join(HERE, name + '_still.pdf'))
    if still_only:
        plt.close(fig); return
    mp4 = os.path.join(HERE, name + '.mp4')
    w = FFMpegWriter(fps=30, codec='libx264', bitrate=4000,
                     extra_args=['-pix_fmt', 'yuv420p', '-profile:v', 'high', '-movflags', '+faststart'])
    anim.save(mp4, writer=w, dpi=DPI)
    plt.close(fig)
    os.makedirs(WEB, exist_ok=True)
    gif = os.path.join(WEB, name + '.gif')
    pal = os.path.join(HERE, '_pal.png')
    vf = 'fps=15,scale=800:-1:flags=lanczos'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', mp4, '-vf', vf + ',palettegen=max_colors=96', pal], check=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', mp4, '-i', pal, '-lavfi',
                    vf + ' [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=4', gif], check=True)
    os.remove(pal)
    print('wrote', name, os.path.getsize(mp4) // 1024, 'KB mp4,', os.path.getsize(gif) // 1024, 'KB gif')


# ------------------------------------------------------------------ 1 geometry: the Rayleigh quotient
def rayleigh(nf, still_only):
    """SOD minimises the roughness of q = x psi over covectors psi; the extremes are the LNM frequencies."""
    M, K, Cd = sy.chain(2, m=[1, 3], k=[1, 2])
    w, P = sy.lnm_modes(M, K)
    y0 = sy.equal_energy(M, K, rng=np.random.default_rng(4))
    dt = 0.05; t, X, V, _ = sy.free_decay(M, K, Cd, y0, dt, 40000)
    Sx, Sv = st.covariances(X, V)
    R = st.sod(X, V)
    th = np.linspace(0, np.pi, 721); Psi = np.c_[np.cos(th), np.sin(th)].T
    rq = np.sqrt(np.einsum('ij,jk,ki->i', Psi.T, Sv, Psi) / np.einsum('ij,jk,ki->i', Psi.T, Sx, Psi))
    th_sod = np.mod(np.arctan2(R.Psi[1], R.Psi[0]), np.pi)
    U = st.pod(X, 2)[0]
    fig = plt.figure(figsize=FIG, dpi=DPI)
    a1 = fig.add_axes([0.075, 0.15, 0.26, 0.72]); a2 = fig.add_axes([0.43, 0.56, 0.2, 0.31])
    a3 = fig.add_axes([0.76, 0.15, 0.22, 0.72])
    seg = slice(0, 2400)
    a1.plot(X[seg, 0], X[seg, 1], lw=0.35, color=REF, alpha=0.7)
    L = 1.15 * np.abs(X[seg]).max()
    for j in range(2):
        u = U[:, j]; a1.plot([-L * u[0], L * u[0]], [-L * u[1], L * u[1]], ':', color=INK, lw=0.9)
        p = R.Phi[:, j] / np.linalg.norm(R.Phi[:, j]); a1.plot([-L * p[0], L * p[0]], [-L * p[1], L * p[1]], '-', color=C[j], lw=1.6)
    trial, = a1.plot([], [], '-', color=C[4], lw=2.4)
    a1.set_xlim(-L, L); a1.set_ylim(-L, L); a1.set_aspect('equal'); a1.set_xlabel(r'$x_1$'); a1.set_ylabel(r'$x_2$')
    a1.set_title(r'POD (dotted), smooth modes (solid)', fontsize=11)
    tt = t[:1200]
    qline, = a2.plot([], [], color=C[4], lw=1.0)
    a2.set_xlim(tt[0], tt[-1]); a2.set_xlabel(r'$t$', labelpad=1); a2.set_ylabel(r'$q=\mathbf{x}^{\mathsf T}\boldsymbol{\psi}$', labelpad=2); a2.set_title(r'coordinate read by $\boldsymbol{\psi}$', fontsize=11)
    a2.set_yticks([])
    a3.plot(np.degrees(th), rq, color=INK, lw=1.4)
    for j in range(2):
        a3.axhline(w[j], color=C[j], ls='--', lw=1); a3.plot(np.degrees(th_sod[j]), np.sqrt(R.mu[j]), 'o', color=C[j], ms=6)
    dot, = a3.plot([], [], 'o', color=C[4], mfc=PAPER, mew=2, ms=8)
    a3.set_xlim(0, 180); a3.set_xticks([0, 45, 90, 135, 180]); a3.set_xlabel(r'direction of $\boldsymbol{\psi}$ (deg)')
    a3.set_ylabel(r'$\sqrt{\rho}$', labelpad=2); a3.set_title(r'roughness $\rho=\langle\dot q^2\rangle/\langle q^2\rangle$', fontsize=11)
    txt = fig.text(0.43, 0.36, '', fontsize=12, color=INK, va='top', linespacing=1.6)

    # angle schedule: sweep 0 -> pi, easing into each extreme of the quotient and dwelling there (2 s at 30 fps),
    # so the smooth coordinate (a single mode) can be read
    t1, t2 = sorted(th_sod); hold = 60

    def ease(a, b, n):
        u = np.linspace(0, 1, n, endpoint=False); return a + (b - a) * (1 - np.cos(np.pi * u)) / 2
    seg = [ease(0, t1, nf // 4), np.full(hold, t1), ease(t1, t2, nf // 3), np.full(hold, t2), ease(t2, np.pi, nf - nf // 4 - nf // 3)]
    ang = np.concatenate(seg); tag = np.concatenate([np.zeros(len(seg[0])), np.ones(hold), np.zeros(len(seg[2])), 2 * np.ones(hold), np.zeros(len(seg[4]))])
    lab = {1: r'minimum $=\omega_1$: mode 1 alone', 2: r'maximum $=\omega_2$: mode 2 alone'}

    def upd(k):
        a = ang[k]; psi = np.array([np.cos(a), np.sin(a)])
        # a covector defines a coordinate; draw the direction it reads (its Sigma_x image, the coordinate's mode)
        d = Sx @ psi; d = d / np.linalg.norm(d)
        trial.set_data([-L * d[0], L * d[0]], [-L * d[1], L * d[1]])
        q = X[:1200] @ psi; qline.set_data(tt, q); a2.set_ylim(-1.15 * np.abs(q).max(), 1.15 * np.abs(q).max())
        r = np.sqrt(psi @ Sv @ psi / (psi @ Sx @ psi)); dot.set_data([np.degrees(a)], [r])
        txt.set_text(r'$\sqrt{\rho}=%.3f$' % r + ('\n' + lab[int(tag[k])] if tag[k] else ''))
        return trial, qline, dot, txt
    anim = FuncAnimation(fig, upd, frames=len(ang), blit=False)
    save(fig, anim, 'rayleigh_quotient', len(ang), len(seg[0]) + hold // 2, still_only)


# ------------------------------------------------------------------ 2 state SOD plane of a damped mode
def state_plane(nf, still_only):
    """Two state-SOD coordinates span one damped modal plane: their trajectory is a decaying circle."""
    M, K, Cd = sy.chain(4, zeta=0.02)
    s = sy.complex_eigs(M, K, Cd)
    y0 = sy.equal_energy(M, K, rng=np.random.default_rng(1))
    dt = 0.01; t, X, V, A = sy.free_decay(M, K, Cd, y0, dt, 60000)
    Y, Yd = np.c_[X, V], np.c_[V, A]
    R = st.state_sod(Y, Yd)
    planes = st.pair_state(R.Q, dt, 10.0, Phi=R.Phi)
    a_, b_, f, c = planes[0]
    Qd = Yd @ R.Psi
    L = st.fit_L(R.Q[:, [a_, b_]], Qd[:, [a_, b_]]); sh, zh = st.plane_eigs(L)
    q = R.Q[:, [a_, b_]] / np.abs(R.Q[:, [a_, b_]]).max()
    fig = plt.figure(figsize=FIG, dpi=DPI)
    a1 = fig.add_axes([0.09, 0.15, 0.3, 0.75]); a2 = fig.add_axes([0.47, 0.15, 0.5, 0.75])
    a1.set_xlim(-1.1, 1.1); a1.set_ylim(-1.1, 1.1); a1.set_aspect('equal'); a1.set_xlabel(r'$q_a$'); a1.set_ylabel(r'$q_b$')
    a1.set_title(r'circularity $%.3f$' % c, fontsize=11)
    trail, = a1.plot([], [], color=C[0], lw=0.6); head, = a1.plot([], [], 'o', color=C[1], ms=6)
    env = np.exp(sh.real * t); env = env / env[0] * np.abs(q[:2000]).max()
    a2.plot(t, env, '--', color=REF, lw=1); a2.plot(t, -env, '--', color=REF, lw=1)
    la, = a2.plot([], [], color=C[0], lw=0.8, label=r'$q_a$'); lb, = a2.plot([], [], color=C[1], lw=0.8, label=r'$q_b$')
    a2.set_xlim(0, t[-1]); a2.set_ylim(-1.1, 1.1); a2.set_xlabel(r'$t$'); a2.legend(loc='upper right', ncol=2)
    a2.set_title(r'mode 1 plane: $\hat\zeta=%.4f$ (exact $%.4f$)' % (zh, -s[0].real / abs(s[0])), fontsize=11)
    idx = np.linspace(200, len(t) - 1, nf).astype(int)

    def upd(k):
        i = idx[k]; j0 = max(0, i - 2500)
        trail.set_data(q[j0:i, 0], q[j0:i, 1]); head.set_data([q[i, 0]], [q[i, 1]])
        la.set_data(t[:i:3], q[:i:3, 0]); lb.set_data(t[:i:3], q[:i:3, 1])
        return trail, head, la, lb
    anim = FuncAnimation(fig, upd, frames=nf, blit=False)
    save(fig, anim, 'state_plane', nf, nf // 3, still_only)


# ------------------------------------------------------------------ 3 windowed SOD along a Duffing decay
def backbone(nf, still_only):
    """Sliding-window SOD of a hardening 2-DOF decay traces the frequency-amplitude dependence of mode 1."""
    M, K, Cd = sy.chain(2, m=[1, 1], k=[1, 1], zeta=0.005)
    w, P = sy.lnm_modes(M, K)
    y0 = np.r_[P[:, 0] / np.abs(P[:, 0]).max() * 2.5, 0, 0]
    dt = 0.05; t, X, V, A = sy.duffing_decay(M, K, Cd, 0.5, y0, dt, int(1500 / dt))
    nwin = int(12 * 2 * np.pi / w[0] / dt); step = nwin // 8
    tc, om, _ = st.sod_track(X, dt, nwin, step, boundary=True)   # 12 linear periods, symmetric boundary correction
    i0 = np.arange(0, len(X) - nwin - 1, step)
    amp = np.array([np.sqrt(2) * X[i:i + nwin, 0].std() for i in i0])
    x = X[:, 0]; zc = np.nonzero((x[:-1] < 0) & (x[1:] >= 0))[0]; tz = t[zc] - x[zc] / (x[zc + 1] - x[zc]) * dt
    fz = 2 * np.pi / np.diff(tz); az = np.array([np.abs(x[zc[k]:zc[k + 1]]).max() for k in range(len(zc) - 1)])
    fig = plt.figure(figsize=FIG, dpi=DPI)
    a1 = fig.add_axes([0.09, 0.15, 0.49, 0.75]); a2 = fig.add_axes([0.68, 0.15, 0.3, 0.75])
    a1.plot(t, x, color=INK, lw=0.4); a1.set_xlim(0, t[-1]); a1.set_xlabel(r'$t$'); a1.set_ylabel(r'$x_1$')
    a1.set_title(r'free decay, cubic spring at mass 1; shaded: one window (12 periods)', fontsize=11)
    span = a1.axvspan(0, 0, color=C[1], alpha=0.25)
    a2.plot(fz, az, '-', color=REF, lw=1.2, label='zero crossings')
    pts, = a2.plot([], [], 'o', color=C[0], ms=4, label='windowed SOD')
    a2.axvline(w[0], color=REF, ls='--', lw=0.8)
    a2.set_xlabel(r'$\omega$ of the smoothest coordinate'); a2.set_ylabel(r'amplitude of $x_1$'); a2.legend(loc='upper left', fontsize=10)
    a2.set_xlim(w[0] - 0.01, max(om.max(), fz.max()) + 0.01); a2.set_ylim(0, 1.05 * az.max())
    idx = np.linspace(0, len(tc) - 1, nf).astype(int)

    def upd(k):
        j = idx[k]; s0 = i0[j] * dt
        span.set_x(s0); span.set_width(nwin * dt)
        pts.set_data(om[:j + 1], amp[:j + 1])
        return span, pts
    anim = FuncAnimation(fig, upd, frames=nf, blit=False)
    save(fig, anim, 'backbone_tracking', nf, nf - 1, still_only)


# ------------------------------------------------------------------ 4 smooth trend extraction
def trend(nf, still_only):
    """Six observed channels mix a drift, a slow cycle and four fast sources; the smoothest SOD coordinate (step 15,
    lag 31) returns the drift with the error of the oracle unmixing A^-1, while POD 1 follows the fast sources."""
    rng = np.random.default_rng(3)
    N = 4000; dt = 1 / N; t = (np.arange(N) + 0.5) * dt
    k = np.array([1, 6, 80, 160, 300, 450]); amp = np.array([1.0, 0.5, 3.0, 2.0, 1.5, 1.0])
    S = amp[:, None] * np.cos(np.pi * k[:, None] * t)
    Q1, _ = np.linalg.qr(rng.normal(size=(6, 6))); Q2, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    Amix = Q1 @ np.diag(np.linspace(1, 3, 6)) @ Q2
    X0 = S.T @ Amix.T; E = rng.standard_normal(X0.shape)
    levels = np.r_[np.linspace(0, 0.03, nf // 2), np.linspace(0.03, 0, nf - nf // 2)]
    drift = S[0] - S[0].mean(); Ai = np.linalg.inv(Amix)

    def fit(z, est):
        return est * (z @ est) / (est @ est)
    fig = plt.figure(figsize=FIG, dpi=DPI)
    a1 = fig.add_axes([0.07, 0.15, 0.4, 0.75]); a2 = fig.add_axes([0.57, 0.15, 0.41, 0.75])
    ch, = a1.plot([], [], color=REF, lw=0.4); a1.set_xlim(0, 1); a1.set_xlabel(r'$t$'); a1.set_ylabel('channel 1')
    lim = 1.1 * np.abs(X0[:, 0]).max() * 1.05; a1.set_ylim(-lim, lim)
    a2.plot(t, drift, color=INK, lw=2.4, alpha=0.35, label='true drift')
    lo, = a2.plot([], [], color=C[2], lw=0.6, label=r'oracle $\mathbf{A}^{-1}$')
    ls, = a2.plot([], [], color=C[0], lw=1.0, label='SOD 1 (step 15, lag 31)')
    lp, = a2.plot([], [], color=C[1], lw=0.8, label='POD 1 (best scaled fit)')
    a2.set_xlim(0, 1); a2.set_ylim(-1.6, 1.6); a2.set_xlabel(r'$t$'); a2.legend(loc='lower left', fontsize=9.5)
    ttl = a1.set_title('', fontsize=11); ttl2 = a2.set_title('', fontsize=11)

    def upd(k):
        Xn = X0 + levels[k] * X0.std(0) * E
        R = st.sod(Xn, dt=dt, h=15, lag=31, center=True, tol=None)
        qs = fit(drift[15:-15], R.Q[:, 0] - R.Q[:, 0].mean())
        Xc = Xn - Xn.mean(0); U = st.pod(Xc, 1)[0]; qp = Xc @ U[:, 0]; qp = fit(drift, qp)
        qo = (Xn - Xn.mean(0)) @ Ai[0]; lo.set_data(t, qo)
        ch.set_data(t, Xn[:, 0]); ls.set_data(t[15:-15], qs); lp.set_data(t, qp)
        err = np.linalg.norm(qs - drift[15:-15]) / np.linalg.norm(drift[15:-15])
        eo = np.linalg.norm(qo - drift) / np.linalg.norm(drift)
        ttl.set_text(r'sensor noise %.1f\,\%%' % (100 * levels[k]))
        ttl2.set_text(r'drift error: SOD %.1e, oracle %.1e' % (err, eo))
        return ch, ls, lp
    anim = FuncAnimation(fig, upd, frames=nf, blit=False)
    save(fig, anim, 'trend_extraction', nf, nf // 2 - 1, still_only)


ALL = {'rayleigh_quotient': rayleigh, 'state_plane': state_plane, 'backbone_tracking': backbone,
       'trend_extraction': trend}

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('names', nargs='*'); p.add_argument('--frames', type=int, default=240)
    p.add_argument('--still-only', action='store_true'); a = p.parse_args()
    for n in (a.names or ALL):
        ALL[n](a.frames, a.still_only)
