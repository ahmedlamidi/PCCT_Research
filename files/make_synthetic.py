#!/usr/bin/env python3
"""
Generate a synthetic stand-in for Water_Phantom with KNOWN properties, so the
analysis script can be validated before the real 9 GB download finishes.

Injects, deliberately:
  - a realistic attenuation profile (matched to the measured single view)
  - charge sharing: a fraction of counts moved to a random neighbour, and
    demoted from High into Low  ->  known positive cross-pixel correlation
                                    and known negative Cov(L,H)
  - an off-axis wobble of +/-27.5 channels (the real value)
  - tube drift
  - fixed-pattern gain non-uniformity
  - five stuck-high columns

If kill_test.py recovers the injected correlation, it works.
Set SHARE_FRAC = 0.0 for a null test: everything should come back ~0.
"""
import os, sys, numpy as np

NROW, NCOL, NVIEWS = 505, 2063, 1440
STUCK = [127, 1804, 1805, 1806, 1935]

SHARE_FRAC = float(os.environ.get("SHARE_FRAC", 0.20))  # 20% of events shared
WOBBLE_CH  = 27.5     # +/- channels, measured from the real data
DRIFT_AMP  = 0.02     # 2% tube drift
SEED       = int(os.environ.get("SEED", 0))

# ---------------------------------------------------------------------------
# Spectral spill at the 30 keV threshold  (Gate A1)
#
# P_SPILL controls a KNOWN, non-cancelling Cov(Low, High). It is independent of
# SHARE_FRAC, which stays the spatial (cross-pixel) injection.
#
# Why a plain spill does not work -- all three of these give Cov(L,H) == 0 EXACTLY,
# confirmed both algebraically and by Monte Carlo:
#     symmetric two-way swap :  L' = L-A+B, H' = H-B+A,  A~Bin(L,p), B~Bin(H,p)
#     one-way spill          :  L' = L-A,   H' = H+A
#     existing SHARE_FRAC    :  shared~Bin(H,f); H-=shared; L+=shared
# The reason is the same in every case: binomial thinning of a Poisson variable
# produces *independent* Poisson parts, so no covariance can survive. The spec
# warns only about the symmetric swap; the one-way spill cancels just as exactly.
#
# What does survive is a FLUCTUATING split fraction -- physically, per-event
# threshold and gain jitter smearing the 30 keV boundary (spectral response
# overlap). For the near-threshold sub-population of rate lam_n, with the
# High-assignment probability q = frac_H + delta,  delta ~ N(0, p^2):
#
#     Cov(L,H) = E[Cov(.|q)] + Cov(E[L|q], E[H|q])
#              = 0 + Cov(lam_n(1-q), lam_n q)  =  -lam_n^2 * p^2
#     Var(L)   = lam_L + lam_n^2 p^2 ,   Var(H) = lam_H + lam_n^2 p^2
#     rho_LH   = -lam_n^2 p^2 / sqrt(Var(L)*Var(H))            <-- NEGATIVE
#
# so |rho_LH| rises monotonically (quadratically) with p_spill. With lam ~ 2900
# in air and NEAR_FRAC = 0.25 (lam_n ~ 725) this predicts roughly
#     p=0.00 -> 0.000     p=0.01 -> -0.035
#     p=0.05 -> -0.476    p=0.10 -> -0.784
# ---------------------------------------------------------------------------
P_SPILL   = float(os.environ.get("P_SPILL", 0.0))
NEAR_FRAC = float(os.environ.get("NEAR_FRAC", 0.25))  # fraction near threshold


def profile():
    """Attenuation profile matched to the measured view-1 values."""
    c = np.arange(NCOL)
    air = 2900.0
    p = np.full(NCOL, air)
    lo, hi = 302, 1634
    x = (c - (lo + hi) / 2) / ((hi - lo) / 2)
    inside = np.abs(x) < 1
    chord = np.sqrt(np.clip(1 - x**2, 0, None))
    p[inside] = air * np.exp(-1.585 * chord[inside])   # -> ~599 at centre
    return p


def main(out_dir, nviews=NVIEWS):
    rng = np.random.default_rng(SEED)
    base = profile()
    theta = np.linspace(0, 2 * np.pi, nviews, endpoint=False)

    # fixed-pattern gain: the ~10x-Poisson spatial non-uniformity we measured
    gain = rng.normal(1.0, 0.055, size=(NROW, NCOL)).astype(np.float32)
    row_fall = np.cos(np.linspace(-0.6, 0.6, NROW))[:, None].astype(np.float32)

    for energy in ("Total", "High"):
        d = os.path.join(out_dir, "Water_Phantom", energy)
        os.makedirs(d, exist_ok=True)

    print(f"generating {nviews} views  SHARE_FRAC={SHARE_FRAC}")
    for j in range(nviews):
        shift = WOBBLE_CH * np.cos(theta[j])            # off-axis wobble
        drift = 1.0 + DRIFT_AMP * np.sin(2 * np.pi * j / nviews * 1.3)

        c = np.arange(NCOL)
        lam_obj = np.interp(c - shift, c, base)      # object attenuation, NO drift
        lam_row = lam_obj * drift                    # tube output applied after
        lam = (lam_row[None, :] * gain * row_fall).astype(np.float32)
        lam = np.clip(lam, 1e-3, None)

        # ---- true independent Poisson counts, split into two energy bands ----
        # High band fraction rises with attenuation (beam hardening).
        #
        # NOTE: this must be computed from lam_obj, NOT lam_row. Beam hardening
        # depends on how much object the beam crossed, not on tube output. Using
        # the drift-modulated lam_row makes frac_H fluctuate view-to-view, which
        # is exactly the fluctuating-split-fraction mechanism used below for
        # P_SPILL, and it injected a spurious Cov(L,H) = -lam^2 Var(frac_H)
        # ~ -54 counts^2 in air (rho_LH ~ -0.035) with zero p_spill -- i.e. a
        # false positive large enough to swamp p_spill = 0.01. It is negligible
        # behind the phantom (lam small), which is why only the air ROIs showed it.
        frac_H = 0.545 + 0.18 * (1 - lam_obj / 2900.0)
        frac_H = np.clip(frac_H, 0.3, 0.85)[None, :]

        if P_SPILL > 0:
            # near-threshold sub-population gets a jittered High/Low split
            lam_near = NEAR_FRAC * lam
            lam_rest = lam - lam_near
            n_H = rng.poisson(lam_rest * frac_H).astype(np.float32)
            n_L = rng.poisson(lam_rest * (1 - frac_H)).astype(np.float32)
            q = np.clip(frac_H + rng.normal(0.0, P_SPILL, size=lam.shape), 0.0, 1.0)
            n_near = rng.poisson(lam_near)
            A = rng.binomial(n_near.astype(np.int64), q).astype(np.float32)
            n_H += A
            n_L += n_near.astype(np.float32) - A
        else:
            n_H = rng.poisson(lam * frac_H).astype(np.float32)
            n_L = rng.poisson(lam * (1 - frac_H)).astype(np.float32)

        if SHARE_FRAC > 0:
            # Charge sharing: a photon near a border deposits in two pixels.
            # Take SHARE_FRAC of High counts; each becomes a Low count in this
            # pixel AND a Low count in a horizontal neighbour.
            shared = rng.binomial(n_H.astype(np.int64), SHARE_FRAC).astype(np.float32)
            n_H -= shared                       # demoted out of High
            n_L += shared                       # ... into Low here
            spill = np.zeros_like(shared)
            spill[:, 1:]  += 0.5 * shared[:, :-1]
            spill[:, :-1] += 0.5 * shared[:, 1:]
            n_L += spill                        # ... and into the neighbour

        T = n_H + n_L
        H = n_H

        for arr, energy in ((T, "Total"), (H, "High")):
            a = np.clip(np.rint(arr), 0, 4095).astype(np.uint16)
            a[:, STUCK] = 4095
            a.tofile(os.path.join(out_dir, "Water_Phantom", energy,
                                  f"proj_{j+1:05d}.raw"))
        if j % 200 == 0:
            print(f"  {j}/{nviews}", flush=True)

    np.save(os.path.join(out_dir, "theta.npy"), theta)
    print("done ->", out_dir)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "synth",
         int(sys.argv[2]) if len(sys.argv) > 2 else NVIEWS)
