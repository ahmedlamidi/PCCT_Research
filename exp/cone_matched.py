"""Cone-angle-matched version of the overlap test.

In an overlap band a voxel at depth d (mm above couch c's slab centre) is seen by
couch c at z1=+d and by couch c+1 at z2=d-15. Define the cone-angle asymmetry

    alpha = |z1| - |z2| = 2d - 15

alpha=0 at d=7.5 mm, where the two positions view the voxel at EQUAL cone-angle
magnitude. FDK cone-beam error grows with |alpha|, so extrapolating R to alpha=0
removes the cone-angle-mismatch term.

Caveat that cannot be removed: at alpha=0 the magnitudes match but the SIGNS are
still opposite (+7.5 vs -7.5), so the two axial PSFs are tilted oppositely. The
four bed positions are 15 mm apart and the axial FOV is 21.84 mm, so no pair ever
views the same voxel at the same signed cone angle. R(alpha=0) is therefore an
UPPER BOUND on genuine object-dependent disagreement, not a clean measurement.
"""
import os, sys, json
import numpy as np
from scipy.ndimage import gaussian_filter, shift as ndshift
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyse_matched import validity, fit_shift, NZ, OFF, DZ

NBIN_D = 9


def analyse(w, vdir):
    ok = validity()
    A = {c: np.load(f'{vdir}/A_c{c}.npy') for c in (1, 2, 3, 4)}
    Bh = {c: np.load(f'{vdir}/B_c{c}.npy') for c in (1, 2, 3, 4)}
    zc = (np.arange(NZ) - (NZ - 1) / 2) * DZ
    out = {}
    for c in (1, 2, 3):
        a1, b1 = A[c][:, :, OFF:], Bh[c][:, :, OFF:]
        a2, b2 = A[c + 1][:, :, :NZ - OFF], Bh[c + 1][:, :, :NZ - OFF]
        m = ok[:, :, OFF:] & ok[:, :, :NZ - OFF]
        s = fit_shift(a1, a2, m); h = s / 2
        sh = lambda v, d: ndshift(v, (0, 0, d), order=1, mode='nearest')
        a1s, b1s, a2s, b2s = sh(a1, h), sh(b1, h), sh(a2, -h), sh(b2, -h)

        ref = gaussian_filter((a1s + b1s + a2s + b2s) / 4., 1.0)
        g = np.sqrt(sum(x ** 2 for x in np.gradient(ref, DZ)))
        DC2 = (np.abs(a1s - b2s) ** 2 + np.abs(b1s - a2s) ** 2) / 2.
        DW1, DW2 = np.abs(a1s - b1s) ** 2, np.abs(a2s - b2s) ** 2

        gs = g[m]
        hi_thr = np.quantile(gs, 0.75)      # sharp: top quartile of gradient
        lo_thr = np.quantile(gs, 0.25)      # flat:  bottom quartile

        d_of_slice = zc[OFF:]
        rows = []
        edges = np.linspace(0, NZ - OFF, NBIN_D + 1).astype(int)
        for i in range(NBIN_D):
            k0, k1 = edges[i], edges[i + 1]
            if k1 <= k0: continue
            sub = np.zeros_like(m); sub[:, :, k0:k1] = True
            d = float(np.mean(d_of_slice[k0:k1]))
            alpha = 2 * d - 15.0
            rec = dict(d=d, alpha=alpha)
            for tag, sel in (('sharp', m & sub & (g >= hi_thr)),
                             ('flat', m & sub & (g <= lo_thr))):
                n = int(sel.sum())
                if n < 400:
                    rec[tag] = None; continue
                R = float(np.sqrt(DC2[sel].mean() /
                                  ((DW1[sel].mean() + DW2[sel].mean()) / 2)))
                rng = np.random.default_rng(i)
                idx = np.flatnonzero(sel.ravel())
                dc, w1, w2 = DC2.ravel(), DW1.ravel(), DW2.ravel()
                bs = []
                for _ in range(40):
                    j = rng.choice(idx, min(n, 20000), replace=True)
                    bs.append(np.sqrt(dc[j].mean() / ((w1[j].mean() + w2[j].mean()) / 2)))
                rec[tag] = dict(R=R, se=float(np.std(bs, ddof=1)), n=n)
            rows.append(rec)

        # extrapolate R(alpha) to alpha = 0 by quadratic fit (cone error is even-ish)
        fits = {}
        for tag in ('sharp', 'flat'):
            pts = [(r['alpha'], r[tag]['R'], r[tag]['se']) for r in rows if r.get(tag)]
            if len(pts) >= 4:
                al = np.array([p[0] for p in pts]); Rv = np.array([p[1] for p in pts])
                wt = 1.0 / np.maximum([p[2] for p in pts], 1e-6)
                co = np.polyfit(al, Rv, 2, w=wt)
                fits[tag] = dict(R_at_alpha0=float(np.polyval(co, 0.0)),
                                 curv=float(co[0]), n_pts=len(pts))
        out[f'{c}->{c+1}'] = dict(rows=rows, fits=fits, shift_um=float(s * DZ * 1000))
        sf = fits.get('sharp', {}); fl = fits.get('flat', {})
        print(f"W{w} pair {c}->{c+1}:  R_sharp(alpha=0) = {sf.get('R_at_alpha0', float('nan')):.3f}"
              f"   R_flat(alpha=0) = {fl.get('R_at_alpha0', float('nan')):.3f}"
              f"   curvature {sf.get('curv', float('nan')):+.4f}/mm^2", flush=True)
    return out


def plot(res, path, title):
    fig, ax = plt.subplots(figsize=(7.6, 5))
    cols = plt.cm.viridis(np.linspace(0.15, 0.8, len(res)))
    for (k, v), col in zip(res.items(), cols):
        for tag, mk, ls in (('sharp', 'o', '-'), ('flat', 's', '--')):
            pts = [(r['alpha'], r[tag]['R'], r[tag]['se']) for r in v['rows'] if r.get(tag)]
            if not pts: continue
            a = [p[0] for p in pts]
            ax.errorbar(a, [p[1] for p in pts], yerr=[p[2] for p in pts],
                        fmt=mk + ls, color=col, alpha=1.0 if tag == 'sharp' else .45,
                        label=f'{k} {tag}', ms=4)
        if 'sharp' in v['fits']:
            ax.plot(0, v['fits']['sharp']['R_at_alpha0'], '*', color=col, ms=17,
                    markeredgecolor='k', zorder=5)
    ax.axvline(0, color='k', ls=':', lw=1)
    ax.axhline(1, color='r', ls=':', lw=1)
    ax.set_xlabel(r'cone-angle asymmetry  $\alpha = |z_1|-|z_2|$  (mm)')
    ax.set_ylabel('R = observed / noise-predicted')
    ax.set_title(title + '\nstars = extrapolation to matched cone angle')
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print('wrote', path)


if __name__ == '__main__':
    w = sys.argv[1] if len(sys.argv) > 1 else '1'
    vdir = 'exp' if w == '1' else f'exp/w{w}'
    res = analyse(w, vdir)
    os.makedirs('exp/figs', exist_ok=True)
    json.dump(res, open(f'exp/cone_matched_w{w}.json', 'w'), indent=1)
    plot(res, f'exp/figs/cone_matched_w{w}.png',
         f'Walnut {w} — excess disagreement vs cone-angle asymmetry')
