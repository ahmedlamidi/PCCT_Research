"""Error bars on the beam, by bootstrap over slab combos.

Two things get uncertainties here:

  1. the 9 stage-1 spectrum parameters -- resample the 55 combos with replacement
     and refit. Resampling COMBOS (not pixels) is the right unit: the residual is
     dominated by model error that is coherent within a combo, so a pixel
     bootstrap would report absurdly small errors.

  2. the per-acquisition threshold offset that the water check needed (+1.30 keV)
     -- resample detector columns with replacement and refit the offset alone.
     Downstream identifiability claims inherit BOTH of these.
"""
import sys, json
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, 'spectrum')
import slabdata, model as M
from physics import mu_lin
import water_check as W

NBOOT = 300


def boot_spectrum(d, keep, nboot=NBOOT):
    tP, tA = d['tP'][keep], d['tA'][keep]
    yL, yH = d['lam_L'][keep], d['lam_H'][keep]
    p_full = np.load('spectrum/fit_p.npy')
    n = len(tP)
    rng = np.random.default_rng(0)
    out = []
    for b in range(nboot):
        idx = rng.integers(0, n, n)
        r = least_squares(M.residuals, p_full, bounds=(M.BOUNDS_LO, M.BOUNDS_HI),
                          args=(tP[idx], tA[idx], yL[idx], yH[idx]),
                          x_scale='jac', max_nfev=4000)
        out.append(r.x)
    return np.array(out), p_full


def boot_water_offset(nboot=120):
    """Refit the acquisition threshold offset on bootstrap resamples of columns."""
    p = np.load('spectrum/fit_p.npy')
    T, L, H = W.load_profile()
    cols = np.arange(W.NC, dtype=float)
    mu_w = mu_lin('Water')

    def frac(delta, Lw):
        q = p.copy(); q[2] += delta; q[3] += delta
        SL, SH = M.spectra_shape(q)
        A = np.exp(-np.outer(Lw, mu_w))
        a, b = A @ SL, A @ SH
        return b / (a + b)

    win = (cols >= 200) & (cols <= 1750) & (T > 0)

    def geom(q):
        R, s0, k = q
        Lw, _ = W.path_len(cols, R, s0)
        SL, SH = M.spectra_shape(p)
        A = np.exp(-np.outer(Lw, mu_w))
        pt = k * ((A @ SL) + (A @ SH))
        return (T[win] - pt[win]) / np.sqrt(np.maximum(pt[win], 1e-9))

    R, s0, _ = least_squares(geom, [18., 0., 1e-5],
                             bounds=([5, -20, 1e-9], [45, 20, 1e-1])).x
    Lw, inside = W.path_len(cols, R, s0)
    m = np.flatnonzero(win & inside & (Lw > 1.) & (L > 0) & (H > 0))
    obs = H[m] / (L[m] + H[m])
    rng = np.random.default_rng(1)
    vals = []
    for b in range(nboot):
        j = rng.integers(0, len(m), len(m))
        r = least_squares(lambda x: obs[j] - frac(x[0], Lw[m][j]), [1.3],
                          bounds=([-8], [8]))
        vals.append(float(r.x[0]))
    return np.array(vals), R, s0


def main():
    d = slabdata.build(); keep, _ = slabdata.fit_split(d)
    S, p_full = boot_spectrum(d, keep)
    print(f'spectrum bootstrap: {len(S)} resamples of {keep.sum()} combos\n')
    print(f"{'parameter':14s}{'fit':>10}{'boot mean':>11}{'sd':>9}{'2.5%':>9}{'97.5%':>9}")
    stats = {}
    for i, nm in enumerate(M.PARAM_NAMES):
        c = S[:, i]
        lo, hi = np.percentile(c, [2.5, 97.5])
        print(f'{nm:14s}{p_full[i]:>10.4f}{c.mean():>11.4f}{c.std(ddof=1):>9.4f}'
              f'{lo:>9.4f}{hi:>9.4f}')
        stats[nm] = dict(fit=float(p_full[i]), mean=float(c.mean()),
                         sd=float(c.std(ddof=1)), lo=float(lo), hi=float(hi))
    cov = np.cov(S.T)

    off, R, s0 = boot_water_offset()
    print(f'\nwater acquisition threshold offset: {off.mean():+.3f} +- {off.std(ddof=1):.3f} keV'
          f'   95% CI [{np.percentile(off,2.5):+.3f}, {np.percentile(off,97.5):+.3f}]')
    print(f'  (cylinder R = {R:.2f} mm, axis offset {s0:+.2f} mm)')
    print(f'  -> nominal E_H 30 keV; slab-fitted {p_full[3]:.2f}; '
          f'water acquisition {p_full[3]+off.mean():.2f}')

    np.savez('spectrum/bootstrap.npz', samples=S, cov=cov,
             water_offset=off, param_names=np.array(M.PARAM_NAMES, dtype=object))
    json.dump(dict(params=stats,
                   water_offset_keV=float(off.mean()),
                   water_offset_sd=float(off.std(ddof=1)),
                   water_offset_ci=[float(np.percentile(off, 2.5)),
                                    float(np.percentile(off, 97.5))],
                   n_boot=len(S)),
              open('spectrum/bootstrap.json', 'w'), indent=1)
    print('\nwrote spectrum/bootstrap.npz and bootstrap.json')


if __name__ == '__main__':
    main()
