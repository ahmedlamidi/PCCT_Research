"""Overlap-vs-gradient, with the resampling confound removed.

First pass applied ndshift() only to the cross arm to correct a -0.3 voxel
residual registration. Linear interpolation is not neutral: with weights
(1-d, d) it scales noise variance by (1-d)^2+d^2 (0.58 at d=0.3) and blurs
structure in proportion to local curvature. So the corrected arm got quieter in
flat regions and blurrier at edges -- which fabricates a rising cross/within
ratio with no physics at all. That is exactly the artefact this test exists to
avoid, and it showed up as cross < within (ratio 0.74) in the flat bins.

Matched design: a1 is shifted by +s/2 in BOTH arms, and each partner is shifted
by the same MAGNITUDE |s/2|:

    CROSS   |shift(a1,+s/2) - shift(a2,-s/2)|   aligned, both interpolated
    CONTROL |shift(a1,+s/2) - shift(b1,+s/2)|   aligned, both interpolated

b1 takes the same shift as a1, so the control stays perfectly registered while
still paying the identical interpolation cost. Every difference between the two
curves is now attributable to bed position, not to resampling.
"""
import os, sys, json
import numpy as np
from scipy.ndimage import gaussian_filter, shift as ndshift
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NXY, SXY, NZ = 256, 50.0, 123
DZ = 15.0 / 77.0
OFF = 77
DSD, DSO = 324.335, 140.24411
V0, DV, NV_DET = 253.99112, 0.1, 505


def validity():
    x = (np.arange(NXY) - (NXY - 1) / 2) * (SXY / NXY)
    z = (np.arange(NZ) - (NZ - 1) / 2) * DZ
    X, Y = np.meshgrid(x, x, indexing='ij')
    a_min = DSO - np.sqrt(X ** 2 + Y ** 2)
    vpos, vneg = (NV_DET - 1 - V0) * DV, V0 * DV
    ok = np.zeros((NXY, NXY, NZ), bool)
    for k, zz in enumerate(z):
        vmm = DSD * zz / a_min
        ok[:, :, k] = (a_min > 1.0) & (vmm <= vpos) & (vmm >= -vneg)
    return ok


def fit_shift(a, b, m):
    best, bs = 0.0, np.inf
    for s in np.arange(-1.0, 1.01, 0.05):
        r = float(np.abs(a[m] - ndshift(b, (0, 0, s), order=1, mode='nearest')[m]).mean())
        if r < bs:
            bs, best = r, float(s)
    return best


def run(tag, load):
    ok = validity()
    A = {c: load(c, 'A') for c in (1, 2, 3, 4)}
    Bh = {c: load(c, 'B') for c in (1, 2, 3, 4)}
    out = {}
    for c in (1, 2, 3):
        a1 = A[c][:, :, OFF:]; b1 = Bh[c][:, :, OFF:]
        a2 = A[c + 1][:, :, :NZ - OFF]; b2 = Bh[c + 1][:, :, :NZ - OFF]
        m = ok[:, :, OFF:] & ok[:, :, :NZ - OFF]

        s = fit_shift(a1, a2, m)
        h = s / 2.0
        sh = lambda v, d: ndshift(v, (0, 0, d), order=1, mode='nearest')
        a1s, b1s = sh(a1, +h), sh(b1, +h)      # same shift -> stay aligned
        a2s, b2s = sh(a2, -h), sh(b2, -h)      # opposite -> aligned to a1s

        ref = gaussian_filter((a1s + b1s + a2s + b2s) / 4.0, 1.0)
        g = np.sqrt(sum(x ** 2 for x in np.gradient(ref, DZ)))

        G = g[m]
        # View angles must match too. a = even views, b = odd views, so a cross
        # term |a1-a2| pairs two EVEN sets: their sparse-view streaks sit at the
        # same angles and partly cancel, while the control |a1-b1| pairs even
        # with odd so streaks add. That alone pushed the flat bins to R~0.76.
        # Use even-vs-odd on BOTH sides, averaging the two symmetric cross terms.
        DC = np.sqrt((np.abs(a1s - b2s)[m]**2 + np.abs(b1s - a2s)[m]**2) / 2.0)
        DW1 = np.abs(a1s - b1s)[m]             # control on couch c
        DW2 = np.abs(a2s - b2s)[m]             # control on couch c+1
        # The two arms have different noise levels (couch c sits at the TOP of its
        # axial FOV in the overlap, couch c+1 at the BOTTOM), so a plain
        # cross/within ratio is offset even under pure noise. Under independent
        # noise with per-arm sigmas s1,s2:
        #     DW1 ~ sqrt(2)s1 ,  DW2 ~ sqrt(2)s2 ,  DC ~ sqrt(s1^2+s2^2)
        # so the noise-only prediction for DC is sqrt((DW1^2+DW2^2)/2), and
        #     R = DC / sqrt((DW1^2+DW2^2)/2)
        # equals 1 EXACTLY under noise for any s1,s2. R>1 is excess disagreement.

        qs = np.quantile(G, np.linspace(0, 1, 13)); qs[-1] += 1e-9
        bins = []
        for i in range(12):
            sel = (G >= qs[i]) & (G < qs[i + 1])
            if sel.sum() < 50:
                continue
            # Mean-SQUARES, not medians. Per voxel E[DC^2]=s1^2+s2^2,
            # E[DW1^2]=2 s1^2, E[DW2^2]=2 s2^2, so
            #     R^2 = <DC^2> / ((<DW1^2>+<DW2^2>)/2) == 1
            # exactly under independent noise for ANY heterogeneous s1,s2.
            # Medians of |differences| do NOT combine this way when sigma varies
            # within a bin, which is what pinned the flat bins at ~0.8.
            cr = float(np.mean(DC[sel]**2)); w1 = float(np.mean(DW1[sel]**2))
            w2 = float(np.mean(DW2[sel]**2))
            pred = (w1 + w2) / 2.0
            n = int(sel.sum())
            rng = np.random.default_rng(i); idx = np.flatnonzero(sel)
            bs = []
            for _ in range(40):
                j = rng.choice(idx, min(n, 20000), replace=True)
                pp = (np.mean(DW1[j]**2) + np.mean(DW2[j]**2)) / 2.0
                bs.append(np.sqrt(np.mean(DC[j]**2) / max(pp, 1e-12)))
            bins.append(dict(g=float(np.median(G[sel])), n=n,
                             cross=float(np.sqrt(cr)), within=float(np.sqrt(w1)),
                             within2=float(np.sqrt(w2)), pred=float(np.sqrt(pred)),
                             ratio=float(np.sqrt(cr / max(pred, 1e-12))),
                             ratio_se=float(np.std(bs, ddof=1))))
        out[f'{c}->{c+1}'] = dict(shift_vox=float(s), shift_um=float(s * DZ * 1000),
                                  n=int(m.sum()), bins=bins)
        lo, hi = bins[0], bins[-1]
        print(f'{tag} pair {c}->{c+1}: shift {s*DZ*1000:+.0f} um | '
              f'flat ratio {lo["ratio"]:.3f}+-{lo["ratio_se"]:.3f}  '
              f'sharp ratio {hi["ratio"]:.3f}+-{hi["ratio_se"]:.3f}  '
              f'change {hi["ratio"]-lo["ratio"]:+.3f}', flush=True)
    return out


def plot(res, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    cols = plt.cm.viridis(np.linspace(0.15, 0.8, len(res)))
    for (k, v), col in zip(res.items(), cols):
        g = [b['g'] for b in v['bins']]
        axes[0].plot(g, [b['cross'] for b in v['bins']], 'o-', color=col, label=f'{k} cross')
        axes[0].plot(g, [b['pred'] for b in v['bins']], 's--', color=col, alpha=.55,
                     label=f'{k} noise prediction')
        axes[1].errorbar(g, [b['ratio'] for b in v['bins']],
                         yerr=[b['ratio_se'] for b in v['bins']], fmt='o-', color=col, label=k)
    axes[0].set_xlabel('local image gradient  |grad mu|  (mm$^{-1}$)')
    axes[0].set_ylabel('median |disagreement|')
    axes[0].set_title('disagreement vs edge sharpness'); axes[0].legend(fontsize=7); axes[0].grid(alpha=.3)
    axes[1].axhline(1.0, color='k', ls=':', lw=1)
    axes[1].set_xlabel('local image gradient  |grad mu|  (mm$^{-1}$)')
    axes[1].set_ylabel('R  =  observed / noise-predicted')
    axes[1].set_title('excess disagreement R\n(R=1 exactly under pure noise; R>1 = real)')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print('wrote', path)


if __name__ == '__main__':
    w = sys.argv[1] if len(sys.argv) > 1 else '1'
    d = 'exp' if w == '1' else f'exp/w{w}'
    res = run(f'W{w}', lambda c, k: np.load(f'{d}/{k}_c{c}.npy'))
    os.makedirs('exp/figs', exist_ok=True)
    json.dump(res, open(f'exp/matched_w{w}.json', 'w'), indent=1)
    plot(res, f'exp/figs/overlap_gradient_w{w}.png',
         f'Walnut {w} — bed-position disagreement vs local gradient (matched-resampling design)')
