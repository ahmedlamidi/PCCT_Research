#!/usr/bin/env python3
"""
Week-1 kill test: spatial-spectral noise covariance of a photon-counting detector.

Measures, from 1440 repeat views of a static phantom:
  (a) Fano factor              var/mean per bin        1.0 = pure Poisson
  (b) non-trivial cross-bin    Cov(T,H) - Var(H)       0 = no charge sharing
  (c) neighbour correlations   lags 1-4, same+cross bin
  (d) how all of the above move with local count rate

Decision rule at the bottom of the output.

CPU only. Reads projections once, keeps only ROIs in RAM.

Usage:
    python kill_test.py --data /path/to/CalibrationPhantomData --out results/
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

NROW, NCOL = 505, 2063
FRAME_BYTES = NROW * NCOL * 2
NVIEWS = int(os.environ.get("NVIEWS", 1440))   # override only for short test runs

# Columns stuck at 4095 down their full length (measured from the archive).
STUCK_COLS = [127, 1804, 1805, 1806, 1935]

# ROIs: (name, row0, row1, col0, col1, detrend)
# Rows 190-310 = central slices, away from the V edges.
# Column ranges from the measured single-view profile. Air needs no detrending
# (open-beam pixels never see the phantom); everything else does.
ROIS = [
    ("air_L",       190, 310,   40,  168, False),
    ("air_R",       190, 310, 1770, 1898, False),
    ("edge_L",      190, 310,  330,  426, True),
    ("light_L",     190, 310,  450,  570, True),
    ("medium_L",    190, 310,  630,  750, True),
    ("deep_centre", 190, 310,  880, 1120, True),
    ("medium_R",    190, 310, 1260, 1380, True),
    ("light_R",     190, 310, 1480, 1600, True),
]

MAX_LAG = 4          # 5x5 neighbourhood
N_HARMONICS = 2      # sin/cos terms for the off-axis wobble
GO_THRESHOLD = 0.10
KILL_THRESHOLD = 0.05


# ----------------------------------------------------------------------------
# LOADING
# ----------------------------------------------------------------------------

def proj_paths(data_dir, phantom, energy):
    d = os.path.join(data_dir, phantom, energy)
    if not os.path.isdir(d):
        sys.exit(f"ERROR: not found: {d}")
    paths = [os.path.join(d, f"proj_{i:05d}.raw") for i in range(1, NVIEWS + 1)]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        sys.exit(f"ERROR: {len(missing)} projections missing, first: {missing[0]}")
    return paths


def load_rois(paths, rois, label=""):
    """One pass over the files; keep only ROI pixels.

    Returns {roi_name: (nviews, h, w) float32}
    """
    out = {}
    for name, r0, r1, c0, c1, _ in rois:
        out[name] = np.empty((len(paths), r1 - r0, c1 - c0), dtype=np.float32)

    t0 = time.time()
    for j, p in enumerate(paths):
        if os.path.getsize(p) != FRAME_BYTES:
            sys.exit(f"ERROR: {p} is {os.path.getsize(p)} bytes, expected {FRAME_BYTES}")
        frame = np.fromfile(p, dtype="<u2").reshape(NROW, NCOL)
        for name, r0, r1, c0, c1, _ in rois:
            out[name][j] = frame[r0:r1, c0:c1]
        if j % 200 == 0:
            el = time.time() - t0
            print(f"    {label} {j:4d}/{len(paths)}  ({el:5.1f}s)", flush=True)
    print(f"    {label} done in {time.time()-t0:.1f}s", flush=True)
    return out


def load_view_angles(data_dir, phantom):
    """Radians, one per view. Falls back to uniform if AcqPara is unreadable."""
    for energy in ("Total", "High"):
        f = os.path.join(data_dir, phantom, energy, "AcqPara.mat")
        if not os.path.exists(f):
            continue
        try:
            from scipy.io import loadmat
            m = loadmat(f, squeeze_me=True, struct_as_record=False)
            ang = np.asarray(m["AcqPara"].objViewAngle, dtype=float)
            if ang.size == NVIEWS:
                print(f"  view angles from {energy}/AcqPara.mat "
                      f"[{ang.min():.4f}, {ang.max():.4f}] rad")
                return ang
        except Exception as e:
            print(f"  could not read {f}: {e}")
    print("  WARNING: falling back to uniform 0..2pi view angles")
    return np.linspace(0, 2 * np.pi, NVIEWS, endpoint=False)


# ----------------------------------------------------------------------------
# BAD PIXELS
# ----------------------------------------------------------------------------

def build_bad_mask(cal_dir, total_frames_sample):
    """True = good. Combines the supplied table, stuck columns, and the
    authors' own <5 / >4090 criterion."""
    good = np.ones((NROW, NCOL), dtype=bool)

    tbl = os.path.join(cal_dir, "badchannelIndexAll.data") if cal_dir else None
    if tbl and os.path.exists(tbl):
        raw = np.fromfile(tbl, dtype=np.float32)
        if raw.size == NROW * NCOL:
            good &= (raw.reshape(NROW, NCOL) == 0)
            print(f"  bad-pixel table read as a mask ({(~good).sum()} flagged)")
        else:
            idx = raw.astype(np.int64)
            idx = idx[(idx >= 0) & (idx < NROW * NCOL)]
            flat = good.ravel()
            flat[idx] = False
            good = flat.reshape(NROW, NCOL)
            print(f"  bad-pixel table read as an index list ({idx.size} entries)")
    else:
        print("  WARNING: no bad-pixel table; using measured criteria only")

    good[:, STUCK_COLS] = False

    s = total_frames_sample
    good &= (s.min(axis=0) >= 5) & (s.max(axis=0) <= 4090)

    print(f"  good pixels: {good.sum()} / {good.size} ({100*good.mean():.2f}%)")
    return good


# ----------------------------------------------------------------------------
# PREPROCESSING
# ----------------------------------------------------------------------------

def drift_reference(air_stack):
    """Per-view common-mode level from an air ROI."""
    return air_stack.mean(axis=(1, 2))


def normalise_drift(stack, ref):
    """Remove tube output drift. Multiplicative, common-mode."""
    return stack * (ref.mean() / ref[:, None, None])


def harmonic_detrend(stack, theta, n_harm=N_HARMONICS):
    """Remove the off-axis wobble: object projection slides sinusoidally with
    gantry angle. Regress each pixel on {1, cos k*th, sin k*th} and keep the
    residual, added back onto the pixel mean so counts stay interpretable.
    """
    nv = stack.shape[0]
    cols = [np.ones(nv)]
    for k in range(1, n_harm + 1):
        cols += [np.cos(k * theta), np.sin(k * theta)]
    X = np.column_stack(cols)

    flat = stack.reshape(nv, -1)
    beta, *_ = np.linalg.lstsq(X, flat, rcond=None)
    resid = flat - X @ beta
    out = (resid + flat.mean(axis=0)).reshape(stack.shape)

    removed = 1.0 - resid.var(axis=0).mean() / max(flat.var(axis=0).mean(), 1e-12)
    return out, float(removed)


# ----------------------------------------------------------------------------
# STATISTICS
# ----------------------------------------------------------------------------

def fano(stack):
    return stack.var(axis=0) / np.maximum(stack.mean(axis=0), 1e-9)


def cross_bin_terms(T, H):
    """The counter-arithmetic cancellation.

    H counts above 30 keV, L between 15 and 30, T = H + L.
    If H and L were independent Poisson:  Cov(T,H) = Var(H) exactly.
    So Cov(T,H) - Var(H) == Cov(L,H), zero by construction in the ideal case.
    Anything nonzero is physics.
    """
    Tc = T - T.mean(axis=0)
    Hc = H - H.mean(axis=0)
    cov_TH = (Tc * Hc).mean(axis=0)
    var_H = H.var(axis=0)
    var_T = T.var(axis=0)

    nontrivial = cov_TH - var_H                       # == Cov(L,H)
    var_L = var_T + var_H - 2 * cov_TH
    denom = np.sqrt(np.maximum(var_L, 1e-12) * np.maximum(var_H, 1e-12))
    rho_LH = nontrivial / denom
    return nontrivial, rho_LH, var_L


def zscore(stack):
    """Zero-mean, unit-variance along the view axis. Computed once per ROI so
    the correlation at every lag is a single elementwise product."""
    z = stack - stack.mean(axis=0)
    s = z.std(axis=0)
    z /= np.maximum(s, 1e-9)
    return z.astype(np.float32), s


def lag_corr_z(Az, Bz, dy, dx, good):
    """Pearson correlation across views between pixel i in A and pixel i+(dy,dx)
    in B, median over all valid pairs. Pairs touching a bad pixel are dropped.
    Inputs must already be z-scored along axis 0.
    """
    h, w = Az.shape[1], Az.shape[2]
    m = MAX_LAG
    ys, xs = slice(m, h - m), slice(m, w - m)
    ysB, xsB = slice(m + dy, h - m + dy), slice(m + dx, w - m + dx)

    gm = good[ys, xs] & good[ysB, xsB]
    if gm.sum() < 50:
        return np.nan, 0

    r = np.einsum("vij,vij->ij", Az[:, ys, xs], Bz[:, ysB, xsB]) / Az.shape[0]
    return float(np.median(r[gm])), int(gm.sum())


def neighbourhood_map(Az, Bz, good, max_lag=MAX_LAG):
    """(2L+1, 2L+1) grid of correlations. Centre is the lag-0 value."""
    n = 2 * max_lag + 1
    out = np.full((n, n), np.nan)
    for dy in range(-max_lag, max_lag + 1):
        for dx in range(-max_lag, max_lag + 1):
            r, _ = lag_corr_z(Az, Bz, dy, dx, good)
            out[dy + max_lag, dx + max_lag] = r
    return out


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def analyse(data_dir, cal_dir, phantom, out_dir, rois=ROIS):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*70}\nKILL TEST — {phantom}\n{'='*70}")

    print("\n[1] view angles")
    theta = load_view_angles(data_dir, phantom)

    print("\n[2] loading ROIs (one pass per bin)")
    pT = proj_paths(data_dir, phantom, "Total")
    pH = proj_paths(data_dir, phantom, "High")
    T_raw = load_rois(pT, rois, "Total")
    H_raw = load_rois(pH, rois, "High")

    print("\n[3] bad-pixel mask")
    sample = np.stack([np.fromfile(p, dtype="<u2").reshape(NROW, NCOL)
                       for p in pT[::144]])
    good_full = build_bad_mask(cal_dir, sample)
    del sample

    # air reference for drift; each air ROI is referenced against the *other*
    # one so a region is never normalised by its own mean (which would
    # artificially anticorrelate its pixels)
    ref_L = drift_reference(T_raw["air_L"])
    ref_R = drift_reference(T_raw["air_R"])
    keep = np.abs(ref_L - np.median(ref_L)) < 5 * ref_L.std()
    print(f"\n[4] drift normalisation — dropping {(~keep).sum()} outlier views")

    results = {}
    for name, r0, r1, c0, c1, do_detrend in rois:
        print(f"\n[5] {name}")
        good = good_full[r0:r1, c0:c1]
        ref = ref_R if name == "air_L" else ref_L

        T = normalise_drift(T_raw[name], ref)[keep]
        H = normalise_drift(H_raw[name], ref)[keep]
        th = theta[keep]

        rate = float(T.mean())
        hard = float(H.mean() / T.mean())
        print(f"    count rate {rate:7.1f}   High/Total {hard:.3f}")

        if do_detrend:
            T, remT = harmonic_detrend(T, th)
            H, remH = harmonic_detrend(H, th)
            print(f"    harmonic detrend removed {100*remT:.1f}% (T) / "
                  f"{100*remH:.1f}% (H) of variance")
        else:
            remT = remH = 0.0
            print("    no detrend (air ROI — never sees the phantom)")

        fT, fH = fano(T), fano(H)
        nontrivial, rho_LH, _ = cross_bin_terms(T, H)
        print(f"    Fano   T {np.median(fT[good]):6.3f}   H {np.median(fH[good]):6.3f}")
        print(f"    rho_LH {np.median(rho_LH[good]):+.4f}")

        Tz, _ = zscore(T)
        Hz, _ = zscore(H)

        lags = {}
        for lab, A, B in (("TT", Tz, Tz), ("HH", Hz, Hz), ("TH", Tz, Hz)):
            for d in range(1, MAX_LAG + 1):
                lags[f"{lab}_x{d}"], _ = lag_corr_z(A, B, 0, d, good)
                lags[f"{lab}_y{d}"], _ = lag_corr_z(A, B, d, 0, good)
        lags["TH_x0"], _ = lag_corr_z(Tz, Hz, 0, 0, good)

        print("    neighbour corr:  " + "  ".join(
            f"{k}={lags[k]:+.3f}" for k in ("TT_x1", "HH_x1", "TH_x1", "TT_x2")))

        maps = {
            "TT": neighbourhood_map(Tz, Tz, good),
            "HH": neighbourhood_map(Hz, Hz, good),
            "TH": neighbourhood_map(Tz, Hz, good),
        }
        del Tz, Hz

        results[name] = dict(
            rate=rate, hardening=hard, detrended=bool(do_detrend),
            var_removed_T=remT, var_removed_H=remH,
            fano_T=float(np.median(fT[good])), fano_H=float(np.median(fH[good])),
            fano_T_iqr=float(np.subtract(*np.percentile(fT[good], [75, 25]))),
            rho_LH=float(np.median(rho_LH[good])),
            rho_LH_iqr=float(np.subtract(*np.percentile(rho_LH[good], [75, 25]))),
            nontrivial=float(np.median(nontrivial[good])),
            n_good=int(good.sum()),
            lags={k: (None if np.isnan(v) else float(v)) for k, v in lags.items()},
            maps={k: v.tolist() for k, v in maps.items()},
        )
        np.savez_compressed(os.path.join(out_dir, f"sigma_{name}.npz"),
                            **{f"map_{k}": v for k, v in maps.items()},
                            fano_T=fT, fano_H=fH, rho_LH=rho_LH,
                            good=good, rate=rate, hardening=hard)

    results["_meta"] = dict(
        phantom=phantom, n_views_kept=int(keep.sum()), max_lag=MAX_LAG,
        n_harmonics=N_HARMONICS, stuck_cols=STUCK_COLS,
        rois=[[r[0], r[1], r[2], r[3], r[4], r[5]] for r in rois],
        se_per_pair=float(1 / np.sqrt(keep.sum())),
    )
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    verdict(results, out_dir)
    return results


def verdict(res, out_dir):
    print(f"\n{'='*70}\nVERDICT\n{'='*70}")
    names = [k for k in res if not k.startswith("_")]
    se = res["_meta"]["se_per_pair"]
    print(f"per-pair standard error ~{se:.4f}; medians over thousands of pairs\n")

    print("(TH_x0, the same-pixel Total-High correlation, is excluded "
          "throughout:\n High is a subset of Total, so it is trivially large "
          "and carries no physics.)\n")
    print(f"{'ROI':<14}{'rate':>8}{'H/T':>7}{'FanoT':>8}{'rho_LH':>9}"
          f"{'TT_x1':>8}{'HH_x1':>8}{'TH_x1':>8}{'|max|':>8}")
    peak = 0.0
    air_peak = 0.0
    for n in names:
        r = res[n]
        vals = [abs(v) for k, v in r["lags"].items()
                if v is not None and k != "TH_x0"]   # TH_x0 = trivial nesting
        mx = max(vals) if vals else 0.0
        peak = max(peak, mx)
        if n.startswith("air"):
            air_peak = max(air_peak, mx)
        print(f"{n:<14}{r['rate']:8.0f}{r['hardening']:7.3f}{r['fano_T']:8.3f}"
              f"{r['rho_LH']:+9.4f}{r['lags']['TT_x1']:+8.3f}"
              f"{r['lags']['HH_x1']:+8.3f}"
              f"{r['lags']['TH_x1']:+8.3f}{mx:8.3f}")

    fanos = [abs(res[n]["fano_T"] - 1) for n in names]
    rhos = [abs(res[n]["rho_LH"]) for n in names]
    print(f"\npeak |correlation| overall : {peak:.4f}")
    print(f"peak |correlation| in air  : {air_peak:.4f}   (no detrending, "
          f"cleanest evidence)")
    print(f"max |Fano - 1|             : {max(fanos):.4f}")
    print(f"max |rho_LH|               : {max(rhos):.4f}")

    if peak >= GO_THRESHOLD or max(fanos) > 0.15 or max(rhos) >= GO_THRESHOLD:
        print("\n>>> GO. Effect is large enough to build the loss around.")
        print("    The measured Sigma is Figure 2. Proceed to Step 9 "
              "(analytic propagation).")
    elif peak < KILL_THRESHOLD and max(rhos) < KILL_THRESHOLD:
        print("\n>>> KILL. Everything below ~0.05.")
        print("    Before falling back: confirm detrending did not over-fit "
              "(check var_removed), and")
        print("    confirm the air ROIs agree with the object ROIs.")
    else:
        print("\n>>> AMBIGUOUS (0.05-0.10).")
        print("    Do not guess. Run Step 9: propagate Sigma through FDK and "
              "the 2x2 inverse, and")
        print("    compare predicted material-image noise against the "
              "diagonal-Sigma prediction.")
        print("    Ratio > ~1.2 means it matters anyway.")

    if air_peak < KILL_THRESHOLD <= peak:
        print("\n    ! Air is quiet but object ROIs are not. Suspect residual "
              "geometry, not physics.")
        print("      Increase N_HARMONICS and re-run before believing the "
              "object numbers.")
    print(f"\nartifacts written to {out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="dir containing Water_Phantom/ and HAP_Phantom/")
    ap.add_argument("--cal", default=None,
                    help="dir containing badchannelIndexAll.data")
    ap.add_argument("--phantom", default="Water_Phantom",
                    choices=["Water_Phantom", "HAP_Phantom"])
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    analyse(a.data, a.cal, a.phantom, a.out)
