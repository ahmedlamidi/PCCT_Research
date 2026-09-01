"""Aggregate the PMMA/Al slab grid into the (Low, High) counts the spectrum fit uses.

Stage 1 fits GLOBAL spectrum parameters, so counts are aggregated spatially over
the good-pixel mask. That kills per-pixel gain variation, which stage 2 handles
separately.

Two choices worth stating:

  * PMMA_0_AL_0 is excluded from the fit, as specified. Note the usual reason
    (it is byte-identical to the air table) does not actually apply here -- this
    fit works on raw counts and never normalises by the air table, so there is no
    circularity. It is held out instead as a free zero-thickness validation point:
    the fitted spectrum has to predict the unattenuated counts it never saw.

  * Counts are 600-frame averages quantised to uint16, so the per-pixel variance
    is lam/600 + 1/12 (photon + quantisation), and Low = Total - High carries
    2/12 because it is a difference of two quantised bins. High is a subset of
    Total, so Var(T-H) = Var(T) - Var(H) and Low is Poisson in its own right.
"""
import os, re, json
import numpy as np

NR, NC = 505, 2063
NP = NR * NC
SLABS = 'data/CalibrationPhantomData/PMMA_AL_slabs'
CAL = 'data/CalibrationTable'
NFRAMES = 600
SELF_REF = 'PMMA_0_AL_0'
CACHE = 'spectrum/slab_aggregate.npz'


def parse_combos():
    out = []
    for d in sorted(os.listdir(SLABS)):
        m = re.fullmatch(r'PMMA_([\d.]+)_AL_([\d.]+)', d)
        if m and os.path.isdir(f'{SLABS}/{d}'):
            out.append((d, float(m.group(1)), float(m.group(2))))
    return out


def good_mask(T, H):
    good = np.ones(NP, bool)
    idx = np.round(np.fromfile(f'{CAL}/badchannelIndexAll.data', '<f4')).astype(np.int64) - 1
    good[idx[(idx >= 0) & (idx < NP)]] = False
    Ti, Hi = T.astype(np.int32), H.astype(np.int32)
    L = Ti - Hi
    for A in (Ti, Hi, L):
        good &= (A >= 5).all(0) & (A <= 4090).all(0)
    colbad = (~good).reshape(NR, NC).mean(0)
    g2 = good.reshape(NR, NC).copy()
    g2[:, np.where(colbad > 0.5)[0]] = False
    return g2.ravel()


def build(force=False):
    if os.path.exists(CACHE) and not force:
        d = np.load(CACHE, allow_pickle=True)
        return {k: d[k] for k in d.files}
    combos = parse_combos()
    T = np.empty((len(combos), NP), np.uint16)
    H = np.empty((len(combos), NP), np.uint16)
    for k, (d, _, _) in enumerate(combos):
        T[k] = np.fromfile(f'{SLABS}/{d}/proj_total.raw', '<u2')
        H[k] = np.fromfile(f'{SLABS}/{d}/proj_high.raw', '<u2')
    good = good_mask(T, H)
    n = int(good.sum())
    print(f'  {len(combos)} combos, {n}/{NP} good pixels ({100*n/NP:.2f}%)')

    Tg = T[:, good].astype(np.float64)
    Hg = H[:, good].astype(np.float64)
    Lg = Tg - Hg
    names = np.array([c[0] for c in combos])
    tP = np.array([c[1] for c in combos])
    tA = np.array([c[2] for c in combos])

    # spatial aggregate (mean over good pixels) and its variance
    lam_L, lam_H = Lg.mean(1), Hg.mean(1)
    var_L = (Lg / NFRAMES + 2 / 12.0).mean(1) / n     # variance OF THE MEAN
    var_H = (Hg / NFRAMES + 1 / 12.0).mean(1) / n
    # per-pixel-equivalent variance, i.e. the weight the brief specifies
    pp_L, pp_H = lam_L / NFRAMES + 2 / 12.0, lam_H / NFRAMES + 1 / 12.0

    out = dict(names=names, tP=tP, tA=tA, lam_L=lam_L, lam_H=lam_H,
               var_L=var_L, var_H=var_H, pp_L=pp_L, pp_H=pp_H,
               n_pix=np.array(n), good=good)
    os.makedirs('spectrum', exist_ok=True)
    np.savez_compressed(CACHE, **out)
    return out


def fit_split(d):
    """Boolean masks: which combos enter the fit, which is the held-out zero point."""
    keep = d['names'] != SELF_REF
    zero = d['names'] == SELF_REF
    return keep, zero


if __name__ == '__main__':
    d = build(force=True)
    keep, zero = fit_split(d)
    print(f"  fit set {keep.sum()} combos -> {2*keep.sum()} measurements; "
          f"held out: {d['names'][zero][0]}")
    print(f"\n{'combo':16s}{'tPMMA':>7}{'tAl':>6}{'lam_L':>11}{'lam_H':>11}"
          f"{'sd_L/lam':>10}{'H/(L+H)':>9}")
    order = np.argsort(d['tP'] * 100 + d['tA'])
    for i in order[:6]:
        print(f"{d['names'][i]:16s}{d['tP'][i]:>7.0f}{d['tA'][i]:>6.1f}"
              f"{d['lam_L'][i]:>11.2f}{d['lam_H'][i]:>11.2f}"
              f"{np.sqrt(d['pp_L'][i])/d['lam_L'][i]:>10.4f}"
              f"{d['lam_H'][i]/(d['lam_L'][i]+d['lam_H'][i]):>9.3f}")
    print('  ...')
    for i in order[-4:]:
        print(f"{d['names'][i]:16s}{d['tP'][i]:>7.0f}{d['tA'][i]:>6.1f}"
              f"{d['lam_L'][i]:>11.2f}{d['lam_H'][i]:>11.2f}"
              f"{np.sqrt(d['pp_L'][i])/d['lam_L'][i]:>10.4f}"
              f"{d['lam_H'][i]/(d['lam_L'][i]+d['lam_H'][i]):>9.3f}")
    print(f"\n  attenuation range: L {d['lam_L'].max()/d['lam_L'].min():.1f}x, "
          f"H {d['lam_H'].max()/d['lam_H'].min():.1f}x")
