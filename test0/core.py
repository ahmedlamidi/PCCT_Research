"""Test 0 — is there systematic structure left after a per-pixel polynomial calibration?

Core data handling: loading, masking, log-attenuation, noise model, fitting.
"""
import os, re, numpy as np

NR, NC = 505, 2063
NP = NR * NC
SLABS = 'data/CalibrationPhantomData/PMMA_AL_slabs'
CAL = 'data/CalibrationTable'
NFRAMES = 600           # AcqPara: nFramesNumPerView
BINS = ('total', 'high', 'low')

# air_table_*.raw is byte-identical to PMMA_0_AL_0 -> that combo is self-normalising
# (P == 0, zero noise). Excluded so it cannot anchor the fit or deflate the RMS.
SELF_REF_COMBO = 'PMMA_0_AL_0'


def parse_combos():
    """List slab folders and parse thicknesses from the names. No assumed grid."""
    out = []
    for d in sorted(os.listdir(SLABS)):
        m = re.fullmatch(r'PMMA_([\d.]+)_AL_([\d.]+)', d)
        if m and os.path.isdir(f'{SLABS}/{d}'):
            out.append((d, float(m.group(1)), float(m.group(2))))
    return out


def load_raw(combos):
    """Raw counts -> (n_combo, NP) uint16 for total and high."""
    T = np.empty((len(combos), NP), np.uint16)
    H = np.empty((len(combos), NP), np.uint16)
    for k, (d, _, _) in enumerate(combos):
        T[k] = np.fromfile(f'{SLABS}/{d}/proj_total.raw', '<u2')
        H[k] = np.fromfile(f'{SLABS}/{d}/proj_high.raw', '<u2')
    return T, H


def build_mask(T, H, verbose=True):
    """One mask, fixed across every combo, so D2's SVD is well posed."""
    good = np.ones(NP, bool)
    rep = {}

    idx = np.fromfile(f'{CAL}/badchannelIndexAll.data', '<f4')
    idx = np.round(idx).astype(np.int64) - 1          # stored 1-based (MATLAB)
    idx = idx[(idx >= 0) & (idx < NP)]
    good[idx] = False
    rep['bad_table'] = int(NP - good.sum())

    Ti, Hi = T.astype(np.int32), H.astype(np.int32)
    L = Ti - Hi

    prev = good.sum()
    for A in (Ti, Hi, L):
        good &= (A >= 5).all(0) & (A <= 4090).all(0)
    rep['counts_out_of_range'] = int(prev - good.sum())

    prev = good.sum()
    good &= (L > 0).all(0)
    rep['negative_low'] = int(prev - good.sum())

    # tile gaps / dead columns: any column mostly rejected already
    colbad = (~good).reshape(NR, NC).mean(0)
    dead = np.where(colbad > 0.5)[0]
    g2 = good.reshape(NR, NC).copy(); g2[:, dead] = False
    prev = good.sum(); good = g2.ravel()
    rep['tile_gap_cols'] = sorted(dead.tolist())
    rep['tile_gap_pixels'] = int(prev - good.sum())

    rep['n_good'] = int(good.sum()); rep['n_total'] = NP
    rep['frac_good'] = float(good.mean())
    if verbose:
        print(f"  mask: {rep['n_good']}/{NP} good ({100*rep['frac_good']:.2f}%)")
        for k in ('bad_table', 'counts_out_of_range', 'negative_low', 'tile_gap_pixels'):
            print(f"    -{rep[k]:8d}  {k}")
        print(f"    dead columns: {rep['tile_gap_cols']}")
    return good, rep


def attenuation_and_noise(T, H, air_T, air_H, bin_name):
    """P = -ln(I_obj/I_air) and the per-measurement variance of P.

    Stored values are 600-frame averages quantised to uint16, so
        Var(v) = v/600 + q/12        (photon + quantisation)
    with q=1 for total/high and q=2 for low (difference of two quantised bins).
    Low is Poisson with its own mean because High is a subset of Total, so
    Cov(T,H)=Var(H) and Var(T-H)=Var(T)-Var(H).

    The air term is identical across combos, so the per-pixel intercept absorbs
    it exactly and it does not enter the residual. Noise floor is object-only.
    """
    if bin_name == 'total':
        v, a, q = T.astype(np.float64), air_T.astype(np.float64), 1.0
    elif bin_name == 'high':
        v, a, q = H.astype(np.float64), air_H.astype(np.float64), 1.0
    else:
        v = T.astype(np.float64) - H.astype(np.float64)
        a = air_T.astype(np.float64) - air_H.astype(np.float64); q = 2.0
    P = -np.log(v / a[None, :])
    var_P = (v / NFRAMES + q / 12.0) / v**2
    return P.astype(np.float32), var_P.astype(np.float32)


def design(tP, tA, deg):
    """Design matrix; thicknesses scaled to [0,1] for conditioning at deg 3."""
    p, a = tP / 40.0, tA / 5.0
    cols = [np.ones_like(p), p, a]
    if deg >= 2:
        cols += [p*p, a*a, p*a]
    if deg >= 3:
        cols += [p**3, a**3, p*p*a, p*a*a]
    return np.stack(cols, 1)


def fit_predict(X, Y, train, test):
    """Per-pixel least squares on train combos; predict held-out combos.

    X is shared by every pixel, so one pinv serves all of them.
    Returns residuals (n_test, n_pix) and the leverage weights W (n_test, n_train).
    """
    Xtr, Xte = X[train], X[test]
    pinv = np.linalg.pinv(Xtr)                 # (n_terms, n_train)
    beta = pinv @ Y[train]                     # (n_terms, n_pix)
    resid = Y[test] - Xte @ beta
    W = Xte @ pinv                             # (n_test, n_train)
    return resid, W


def predicted_resid_std(var_P, train, test, W):
    """Exact residual std under independent noise: observation + prediction variance."""
    return np.sqrt(var_P[test] + (W**2) @ var_P[train])
