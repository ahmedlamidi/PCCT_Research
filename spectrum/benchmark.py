"""Head-to-head: physical forward model vs the per-pixel polynomial (STEPC-style).

The stage-2 number cannot be compared to Test 0's 5.04/2.41/1.38 directly -- Test 0
scored in log-attenuation space with a leverage-corrected noise term, stage 2
scored in count space with observation-only noise. So both models are re-scored
HERE in one place, on the same pixels, the same split, the same space and the
same noise model:

    P      = -ln(y / air)                 air = PMMA_0_AL_0, as Test 0 used
    sigma  = sqrt(y/600 + q/12) / y       count noise propagated to log space
    z      = rms( (P_obs - P_pred)/sigma ), per-combo offset removed

Leverage is reported both ways. Including it enlarges the denominator by the
prediction variance, which helps the model with MORE parameters -- so the
observation-only column is the one that does not flatter the polynomials.

    physical : 9 global + 2 per pixel   (gain, threshold offset)
    poly deg1: 3 per pixel     deg2: 6 per pixel     deg3: 10 per pixel
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slabdata, model as M
from stage2 import load_pixels, shapes_for_deltas, DELTAS, NFRAMES, CHUNK, fit_stage2


def design(tP, tA, deg):
    p, a = tP / 40.0, tA / 5.0
    cols = [np.ones_like(p), p, a]
    if deg >= 2: cols += [p * p, a * a, p * a]
    if deg >= 3: cols += [p ** 3, a ** 3, p * p * a, p * a * a]
    return np.stack(cols, 1)


def zstat(resid, sd):
    z = resid / sd
    z = z - z.mean(1, keepdims=True)
    return float(np.sqrt((z.astype(np.float64) ** 2).mean()))


def main():
    p = np.load('spectrum/fit_p.npy')
    names, tP, tA, L, H, good = load_pixels()
    npix = L.shape[1]
    keep = names != slabdata.SELF_REF
    ki = np.flatnonzero(keep)
    air_i = np.flatnonzero(~keep)[0]
    airL, airH = L[air_i].astype(np.float64), H[air_i].astype(np.float64)

    tot = (L + H)[ki].mean(1)
    order = ki[np.argsort(-tot)]
    half = len(ki) // 2
    train, test = np.sort(order[:half]), np.sort(order[half:])
    print(f'{npix} pixels | train {len(train)} thin, test {len(test)} thick')

    def P_and_sd(counts, air, q):
        y = np.maximum(counts.astype(np.float64), 1e-6)
        return -np.log(y / np.maximum(air, 1e-6)), np.sqrt(y / NFRAMES + q / 12.0) / y

    res = {}

    # ---------------- physical model ----------------
    mL, mH = shapes_for_deltas(p, tP, tA)
    gL = np.zeros(npix); gH = np.zeros(npix); jj = np.zeros(npix, np.int16)
    for s in range(0, npix, CHUNK):
        e = min(s + CHUNK, npix)
        a, b, j, _ = fit_stage2(L[:, s:e], H[:, s:e], mL, mH, train)
        gL[s:e], gH[s:e], jj[s:e] = a, b, j
    rr, ss = [], []
    for c in test:
        for cnt, m, air, q, gg in ((L[c], mL[jj, c], airL, 2.0, gL),
                                   (H[c], mH[jj, c], airH, 1.0, gH)):
            Pobs, sd = P_and_sd(cnt, air, q)
            Ppred = -np.log(np.maximum(gg * m, 1e-9) / np.maximum(air, 1e-6))
            rr.append(Pobs - Ppred); ss.append(sd)
    rr = np.array(rr); ss = np.array(ss)
    npar_phys = 3
    lev = np.sqrt(1.0 + npar_phys / len(train))
    res['physical'] = dict(z_obs_only=zstat(rr, ss), z_leverage=zstat(rr, ss * lev),
                           n_par_per_pixel=npar_phys)
    print(f"physical (2/pixel):  z {res['physical']['z_obs_only']:.3f}  "
          f"(with leverage {res['physical']['z_leverage']:.3f})")

    # ---------------- polynomial baselines ----------------
    for deg in (1, 2, 3):
        X = design(tP, tA, deg)
        Xtr, pinv = X[train], np.linalg.pinv(X[train])
        rr, ss = [], []
        for cnt, air, q in ((L, airL, 2.0), (H, airH, 1.0)):
            Ptr = np.stack([P_and_sd(cnt[c], air, q)[0] for c in train])
            beta = pinv @ Ptr
            for c in test:
                Pobs, sd = P_and_sd(cnt[c], air, q)
                rr.append(Pobs - X[c] @ beta); ss.append(sd)
        rr = np.array(rr); ss = np.array(ss)
        lev = np.sqrt(1.0 + X.shape[1] / len(train))
        res[f'poly_deg{deg}'] = dict(z_obs_only=zstat(rr, ss),
                                     z_leverage=zstat(rr, ss * lev),
                                     n_par_per_pixel=X.shape[1])
        print(f"poly deg{deg} ({X.shape[1]}/pixel): z {res[f'poly_deg{deg}']['z_obs_only']:.3f}  "
              f"(with leverage {res[f'poly_deg{deg}']['z_leverage']:.3f})")

    res['test0_reference'] = {'deg1': 5.035, 'deg2': 2.411, 'deg3': 1.379,
                              'note': 'Test 0 numbers; different noise convention'}
    json.dump(res, open('spectrum/benchmark.json', 'w'), indent=1)
    print('\nwrote spectrum/benchmark.json')


if __name__ == '__main__':
    main()
