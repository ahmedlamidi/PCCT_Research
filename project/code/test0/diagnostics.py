"""D3/D4/D5 figures: what does the leftover structure actually look like?"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import core, analysis

TILE = 129


def residuals_for(split, deg, crop, data, good, X, splits):
    tr, te = splits[split]
    out = {}
    for b in core.BINS:
        P, V = data[b]
        resid, W = core.fit_predict(X, P, tr, te)
        sd = core.predicted_resid_std(V, tr, te, W)
        r = resid[:, good].astype(np.float64); s = sd[:, good].astype(np.float64)
        r = r - r.mean(1, keepdims=True)          # drop per-acquisition offset
        out[b] = (r, s)
    return out, te


def to_map(vec, good, shape):
    m = np.full(good.size, np.nan); m[good] = vec
    return m.reshape(shape)


def chan_spectrum(mp):
    """Power spectrum along the channel axis, averaged over rows. Tile gaps -> row mean."""
    a = np.where(np.isfinite(mp), mp, np.nan)
    rm = np.nanmean(a, 1, keepdims=True)
    a = np.where(np.isfinite(a), a, rm)
    a = a - a.mean(1, keepdims=True)
    F = np.abs(np.fft.rfft(a, axis=1))**2
    return F.mean(0)


def main(crop=True, outdir='test0_results'):
    allc = core.parse_combos()
    T, H = core.load_raw(allc)
    good_full, _ = core.build_mask(T, H, verbose=False)
    i0 = [i for i, c in enumerate(allc) if c[0] == core.SELF_REF_COMBO][0]
    air_T, air_H = T[i0].copy(), H[i0].copy()
    keep = [i for i, c in enumerate(allc) if c[0] != core.SELF_REF_COMBO]
    combos = [allc[i] for i in keep]; T, H = T[keep], H[keep]
    if crop:
        r0, r1, c0, c1 = 124, 380, 775, 1287
        sel = np.zeros((core.NR, core.NC), bool); sel[r0:r1, c0:c1] = True; sel = sel.ravel()
        shape = (r1 - r0, c1 - c0); coff = c0
    else:
        sel = np.ones(core.NP, bool); shape = (core.NR, core.NC); coff = 0
    T, H, air_T, air_H = T[:, sel], H[:, sel], air_T[sel], air_H[sel]
    good = good_full[sel]
    tP = np.array([c[1] for c in combos]); tA = np.array([c[2] for c in combos])
    data = {b: core.attenuation_and_noise(T, H, air_T, air_H, b) for b in core.BINS}
    splits = analysis.make_splits(combos, data['total'][0][:, good].mean(1))
    X = core.design(tP, tA, 2)

    res = {}
    for sp in ('random_seed0', 'extrapolate_thick'):
        res[sp] = residuals_for(sp, 2, crop, data, good, X, splits)

    # ---- D2 spectra ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, sp in zip(axes, res):
        (rr, ss) = res[sp][0]['total']
        z = rr / ss; z = z - z.mean(1, keepdims=True)
        lam = analysis.gram_svals(z.T); lam /= lam.sum()
        n = np.random.default_rng(0).standard_normal(z.shape)
        n -= n.mean(1, keepdims=True)
        ln = analysis.gram_svals(n.T); ln /= ln.sum()
        ax.semilogy(np.arange(1, len(lam)+1), lam, 'o-', label='observed (whitened)')
        ax.semilogy(np.arange(1, len(ln)+1), ln, 's--', color='grey', label='iid null')
        ax.set_title(f'{sp} — total bin, deg 2'); ax.set_xlabel('component')
        ax.set_ylabel('fraction of residual energy'); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f'{outdir}/fig_D2_singular_values.png', dpi=130); plt.close(fig)

    # ---- D3 residual maps + leading spatial component ----
    (rr, ss) = res['extrapolate_thick'][0]['total']; te = res['extrapolate_thick'][1]
    z = rr / ss; z = z - z.mean(1, keepdims=True)
    lam, V = np.linalg.eigh((z @ z.T).astype(np.float64))
    u1 = (z.T @ V[:, -1]); u1 /= np.linalg.norm(u1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    for ax, k in zip(axes.ravel()[:3], [0, len(te)//2, len(te)-1]):
        mp = to_map(rr[k], good, shape)
        v = np.nanpercentile(np.abs(mp), 99)
        im = ax.imshow(mp, cmap='RdBu_r', vmin=-v, vmax=v, aspect='auto')
        ax.set_title(f'residual  {combos[te[k]][0]}  (deg2, offset removed)', fontsize=9)
        plt.colorbar(im, ax=ax, fraction=.03)
    mp = to_map(u1, good, shape)
    v = np.nanpercentile(np.abs(mp), 99)
    im = axes[1, 1].imshow(mp, cmap='RdBu_r', vmin=-v, vmax=v, aspect='auto')
    axes[1, 1].set_title('leading spatial component u1 (extrapolation)', fontsize=9)
    plt.colorbar(im, ax=axes[1, 1], fraction=.03)
    fig.tight_layout(); fig.savefig(f'{outdir}/fig_D3_residual_maps.png', dpi=130); plt.close(fig)

    # ---- D3 channel power spectrum: look for the 129-channel tile period ----
    fig, ax = plt.subplots(figsize=(9, 4))
    ps = chan_spectrum(to_map(u1, good, shape))
    f = np.arange(len(ps)) / shape[1]
    ax.semilogy(f[1:], ps[1:], label='u1 (leading component)')
    for k in (1, 2, 3):
        ax.axvline(k / TILE, color='r', ls='--', alpha=.6,
                   label=f'tile period {TILE} ch' if k == 1 else None)
    ax.set_xlabel('cycles / channel'); ax.set_ylabel('power'); ax.legend(); ax.grid(alpha=.3)
    ax.set_title('D3 — channel-direction power spectrum of the leading residual component')
    fig.tight_layout(); fig.savefig(f'{outdir}/fig_D3_power_spectrum.png', dpi=130); plt.close(fig)

    # ---- D4 total vs high ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    d4 = {}
    for ax, sp in zip(axes, res):
        rT = res[sp][0]['total'][0]; rH = res[sp][0]['high'][0]; rL = res[sp][0]['low'][0]
        k = rT.shape[0] // 2
        n = min(20000, rT.shape[1]); idx = np.random.default_rng(0).choice(rT.shape[1], n, False)
        ax.plot(rT[k][idx], rH[k][idx], '.', ms=1, alpha=.2)
        rTH = float(np.corrcoef(rT.ravel(), rH.ravel())[0, 1])
        rTL = float(np.corrcoef(rT.ravel(), rL.ravel())[0, 1])
        d4[sp] = dict(r_TH=rTH, r_TL_trivial=rTL)
        ax.set_title(f'{sp}\nr(Total,High) = {rTH:+.3f}   [r(Total,Low)={rTL:+.3f}, trivial]', fontsize=9)
        ax.set_xlabel('Total residual'); ax.set_ylabel('High residual'); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f'{outdir}/fig_D4_cross_bin.png', dpi=130); plt.close(fig)

    # ---- D5 residual vs thickness ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    rr2 = res['random_seed0'][0]['total'][0]; te2 = res['random_seed0'][1]
    for ax, tv, nm in ((axes[0], tP[te2], 'PMMA (mm)'), (axes[1], tA[te2], 'Al (mm)')):
        ax.plot(tv, np.sqrt((rr2**2).mean(1)), 'o')
        ax.set_xlabel(nm); ax.set_ylabel('residual RMS (offset removed)'); ax.grid(alpha=.3)
    fig.suptitle('D5 — residual vs thickness, held-out combos, deg 2')
    fig.tight_layout(); fig.savefig(f'{outdir}/fig_D5_thickness.png', dpi=130); plt.close(fig)

    json.dump(d4, open(f'{outdir}/d4_cross_bin.json', 'w'), indent=1)
    np.save(f'{outdir}/u1_map.npy', to_map(u1, good, shape))
    print('figures written to', outdir)
    return d4


if __name__ == '__main__':
    full = '--full' in sys.argv
    main(crop=not full, outdir='test0_results_full' if full else 'test0_results')
