"""Stage 2: per-pixel gain and threshold offset, with the spectrum shape frozen.

This is the physics-based replacement for STEPC. Where STEPC is a per-pixel
degree-2 polynomial in the two measured attenuations (5 free numbers per pixel,
no physical meaning), this carries TWO per-pixel numbers with physical content:

    g_i      flux/gain scale
    delta_i  threshold offset in keV, applied to BOTH thresholds

and everything else comes from the 9 global spectrum parameters.

Scored on Test 0's exact extrapolation split -- train on the thin half, predict
the thick half -- using Test 0's centred whitened z, so it is directly comparable
to the STEPC benchmark of 5.04 (deg 1) -> 2.41 (deg 2) -> 1.38 (deg 3).

Efficiency: the model is LINEAR in g, and for a scaled-Poisson likelihood the MLE
is g = sum(y)/sum(m). The shape m depends only on (combo, delta), never on the
pixel, so delta is a small 1-D grid search with g solved in closed form -- no
per-pixel nonlinear optimisation over a million pixels.
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slabdata
import model as M

NFRAMES = 600
DELTAS = np.linspace(-8.0, 8.0, 41)      # keV; widened after 10.7% of
                                         # pixels railed on a +-3 keV grid
CHUNK = 120_000


def load_pixels():
    """Per-pixel Low/High counts for all 56 combos, good pixels only."""
    combos = slabdata.parse_combos()
    NP = slabdata.NP
    T = np.empty((len(combos), NP), np.uint16)
    H = np.empty((len(combos), NP), np.uint16)
    for k, (d, _, _) in enumerate(combos):
        T[k] = np.fromfile(f'{slabdata.SLABS}/{d}/proj_total.raw', '<u2')
        H[k] = np.fromfile(f'{slabdata.SLABS}/{d}/proj_high.raw', '<u2')
    good = slabdata.good_mask(T, H)
    L = (T[:, good].astype(np.float32) - H[:, good].astype(np.float32))
    Hg = H[:, good].astype(np.float32)
    names = np.array([c[0] for c in combos])
    tP = np.array([c[1] for c in combos]); tA = np.array([c[2] for c in combos])
    return names, tP, tA, L, Hg, good


def shapes_for_deltas(p, tP, tA):
    """m_L, m_H of shape (n_delta, n_combo): model at unit gain for each offset."""
    mL = np.empty((len(DELTAS), len(tP))); mH = np.empty_like(mL)
    for j, dlt in enumerate(DELTAS):
        q = p.copy(); q[2] += dlt; q[3] += dlt          # shift E_L and E_H together
        a, b = M._shape(q, tP, tA)
        mL[j], mH[j] = a, b
    return mL, mH


def fit_stage2(L, H, mL, mH, train, per_bin_gain=True):
    """Per-pixel (g_L, g_H, delta): grid search on delta, gains in closed form.

    A separate gain per bin costs one extra per-pixel number but absorbs per-bin
    efficiency differences that a single shared gain cannot. Still 3 per pixel,
    against 6 for a degree-2 polynomial.
    """
    npix = L.shape[1]
    best_chi = np.full(npix, np.inf, np.float64)
    best_gL = np.zeros(npix); best_gH = np.zeros(npix); best_j = np.zeros(npix, np.int16)
    yL = L[train].sum(0).astype(np.float64)
    yH = H[train].sum(0).astype(np.float64)
    for j in range(len(DELTAS)):
        sL, sH = mL[j, train].sum(), mH[j, train].sum()
        if per_bin_gain:
            gL, gH = yL / max(sL, 1e-30), yH / max(sH, 1e-30)
        else:
            gL = gH = (yL + yH) / max(sL + sH, 1e-30)
        chi = np.zeros(npix)
        for c in train:
            for y, m, gg in ((L[c], mL[j, c], gL), (H[c], mH[j, c], gH)):
                lam = gg * m
                chi += (y - lam) ** 2 / np.maximum(lam, 1e-9)
        upd = chi < best_chi
        best_chi[upd] = chi[upd]; best_gL[upd] = gL[upd]
        best_gH[upd] = gH[upd]; best_j[upd] = j
    return best_gL, best_gH, best_j, best_chi


def centred_z(resid, sd):
    """Test 0's statistic: whiten by predicted sd, drop the per-combo offset, rms."""
    z = resid / sd
    z = z - z.mean(1, keepdims=True)          # per-combo global offset removed
    return float(np.sqrt((z.astype(np.float64) ** 2).mean()))


def main():
    p = np.load('spectrum/fit_p.npy')
    t0 = time.time()
    names, tP, tA, L, H, good = load_pixels()
    npix = L.shape[1]
    print(f'  {npix} good pixels x {len(names)} combos loaded in {time.time()-t0:.0f}s')

    keep = names != slabdata.SELF_REF
    ki = np.flatnonzero(keep)
    # Test 0's extrapolation split: rank by attenuation, train on the thin half
    tot = (L + H)[ki].mean(1)
    order = ki[np.argsort(-tot)]              # brightest (thinnest) first
    half = len(ki) // 2
    train, test = np.sort(order[:half]), np.sort(order[half:])
    print(f'  extrapolation split: train {len(train)} thin, test {len(test)} thick')

    mL, mH = shapes_for_deltas(p, tP, tA)
    t0 = time.time()
    g = np.zeros(npix); jj = np.zeros(npix, np.int16)
    for s in range(0, npix, CHUNK):
        e = min(s + CHUNK, npix)
        gg, j, _ = fit_stage2(L[:, s:e], H[:, s:e], mL, mH, train)
        g[s:e], jj[s:e] = gg, j
    dlt = DELTAS[jj]
    print(f'  stage-2 per-pixel fit in {time.time()-t0:.0f}s')
    print(f'  gain   : median {np.median(g):.4g}  IQR {np.subtract(*np.percentile(g,[75,25])):.3g}'
          f'  spread {100*g.std()/g.mean():.2f}%')
    print(f'  delta  : median {np.median(dlt):+.3f} keV  '
          f'IQR {np.subtract(*np.percentile(dlt,[75,25])):.3f}  '
          f'range [{dlt.min():+.2f},{dlt.max():+.2f}]')
    print(f'  at grid edge: {100*np.mean(np.abs(dlt)>2.9):.2f}% of pixels')

    # ---- held-out prediction on the thick half ----
    res, sds = [], []
    for c in test:
        for y, m, q in ((L[c], mL[jj, c], 2.0), (H[c], mH[jj, c], 1.0)):
            lam = g * m
            res.append(y - lam)
            sds.append(np.sqrt(np.maximum(lam, 1e-9) / NFRAMES + q / 12.0))
    res = np.array(res); sds = np.array(sds)
    z = centred_z(res, sds)
    raw_z = float(np.sqrt(((res / sds).astype(np.float64) ** 2).mean()))
    frac = float(np.sqrt(((res / np.maximum(np.array(
        [L[c] for c in test] + [H[c] for c in test]), 1e-9)) ** 2).mean()))
    print(f'\nHELD-OUT thick half:  centred whitened z = {z:.3f}   (raw z {raw_z:.3f})')
    print(f'  STEPC benchmark on the same split: 5.04 (deg1) -> 2.41 (deg2) -> 1.38 (deg3)')
    print(f'  physical model: 9 global + 2 per-pixel parameters')
    out = dict(z_centred=z, z_raw=raw_z, n_train=len(train), n_test=len(test),
               n_pix=int(npix), gain_spread_pct=float(100*g.std()/g.mean()),
               delta_median=float(np.median(dlt)),
               delta_iqr=float(np.subtract(*np.percentile(dlt, [75, 25]))),
               stepc_benchmark={'deg1': 5.04, 'deg2': 2.41, 'deg3': 1.38})
    json.dump(out, open('spectrum/stage2.json', 'w'), indent=1)
    np.savez_compressed('spectrum/stage2_pixels.npz', gain=g.astype(np.float32),
                        delta=dlt.astype(np.float32), good=good)
    print('wrote spectrum/stage2.json and stage2_pixels.npz')


if __name__ == '__main__':
    main()
