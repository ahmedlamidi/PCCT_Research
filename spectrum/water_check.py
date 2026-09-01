"""Cross-material validation: predict water transmission from the PMMA/Al-fitted spectrum.

This is the real pass/fail. Spectrum estimation through only two basis materials
is weakly identified -- many wrong spectra reproduce the PMMA/Al grid because
spectrum error is absorbed into effective attenuation. Water is a third material
and breaks the degeneracy.

Guarding against circularity: the cylinder geometry (radius, axis offset) is fitted
using the TOTAL bin ONLY, then the spectrum predicts the Low/High SPLIT, which the
geometry fit never saw. The split is a pure spectral quantity -- it is almost
insensitive to path length errors but very sensitive to getting the spectrum wrong.
"""
import sys, json
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, 'spectrum')
from physics import mu_lin, E_GRID
import model as M

NR, NC = 505, 2063
DATA = 'data/CalibrationPhantomData/Water_Phantom'
DSD, DSO, DU, U0 = 324.335, 140.24411, 0.1, 983.5178
ROWS = (190, 310)          # central slices, as the kill test used
NVIEW = 120                # views averaged; the cylinder is ~rotationally symmetric


def load_profile():
    """View-averaged Low/High counts per detector column, central rows."""
    accT = np.zeros((NR, NC)); accH = np.zeros((NR, NC))
    idx = np.linspace(1, 1440, NVIEW).astype(int)
    for i in idx:
        accT += np.fromfile(f'{DATA}/Total/proj_{i:05d}.raw', '<u2').reshape(NR, NC)
        accH += np.fromfile(f'{DATA}/High/proj_{i:05d}.raw', '<u2').reshape(NR, NC)
    accT /= len(idx); accH /= len(idx)
    r0, r1 = ROWS
    T = accT[r0:r1].mean(0); H = accH[r0:r1].mean(0)
    return T, T - H, H


def path_len(cols, R, s0):
    """Water chord length (mm) for each detector column, cone-beam fan geometry."""
    u = (cols - U0) * DU                      # mm on the detector from piercing point
    theta = np.arctan(u / DSD)                # fan angle
    s = DSO * np.sin(theta) - s0              # perpendicular distance to rotation axis
    inside = np.abs(s) < R
    L = np.zeros_like(s)
    L[inside] = 2.0 * np.sqrt(R ** 2 - s[inside] ** 2)
    return L, inside


def main():
    p = np.load('spectrum/fit_p.npy')
    rep = json.load(open('spectrum/fit_stage1.json'))
    T, L, H = load_profile()
    cols = np.arange(NC, dtype=float)

    # air reference from the outer columns (open beam either side of the cylinder)
    air = np.r_[np.arange(60, 200), np.arange(1750, 1900)]
    airL, airH = L[air].mean(), H[air].mean()

    S_L, S_H = M.spectra_shape(p)
    mu_w = mu_lin('Water')

    def pred(Lw, scale):
        A = np.exp(-np.outer(Lw * scale, mu_w))
        return A @ S_L, A @ S_H

    # ---- stage 1: geometry from the TOTAL bin only ----
    # FIXED evaluation window: the residual vector must not change length as the
    # fitted radius moves the 'inside' mask, or least_squares cannot run.
    win = (cols >= 200) & (cols <= 1750) & (T > 0)

    def geom_resid(q):
        R, s0, k = q
        Lw, _ = path_len(cols, R, s0)
        pL, pH = pred(Lw, 1.0)
        predT = k * (pL + pH)
        return (T[win] - predT[win]) / np.sqrt(np.maximum(predT[win], 1e-9))

    g = least_squares(geom_resid, [18.0, 0.0, 1e-5],
                      bounds=([5, -20, 1e-9], [45, 20, 1e-1]))
    R, s0, k = g.x
    Lw, inside = path_len(cols, R, s0)
    print(f'geometry fitted on TOTAL bin only: R = {R:.2f} mm, axis offset = {s0:+.2f} mm')
    print(f'  water path range over the cylinder: 0 - {Lw.max():.1f} mm')

    # ---- stage 2: the actual test, on the split the geometry fit never saw ----
    pL, pH = pred(Lw, 1.0)
    predL, predH = k * pL, k * pH
    m = win & inside & (Lw > 1.0) & (L > 0) & (H > 0)
    obs_frac = H[m] / (L[m] + H[m])
    pred_frac = predH[m] / (predL[m] + predH[m])
    d = obs_frac - pred_frac

    atten = (airL + airH) / (L[m] + H[m])
    print(f'\nattenuation range probed: {atten.min():.2f}x - {atten.max():.2f}x')
    print(f'\nHigh fraction H/(L+H) -- pure spectral prediction:')
    print(f"{'path mm':>9}{'observed':>11}{'predicted':>11}{'diff':>9}")
    q = np.quantile(Lw[m], np.linspace(0.05, 0.95, 7))
    for lo, hi in zip(q[:-1], q[1:]):
        s = (Lw[m] >= lo) & (Lw[m] < hi)
        if s.sum() < 5: continue
        print(f'{np.median(Lw[m][s]):>9.1f}{obs_frac[s].mean():>11.4f}'
              f'{pred_frac[s].mean():>11.4f}{d[s].mean():>+9.4f}')
    # noise floor on the split, from Poisson on 120-view averages of 600-frame means
    nL, nH = L[m], H[m]
    sd = np.sqrt((nH**2 * nL + nL**2 * nH) / (nL + nH)**4 / (600.0 * NVIEW * (ROWS[1]-ROWS[0])))
    print(f'\nsplit residual: rms {np.sqrt((d**2).mean()):.5f}   '
          f'noise floor {sd.mean():.5f}   ratio {np.sqrt((d**2).mean())/sd.mean():.1f}x')
    out = dict(R_mm=float(R), axis_offset_mm=float(s0),
               atten_min=float(atten.min()), atten_max=float(atten.max()),
               split_rms=float(np.sqrt((d**2).mean())), noise_floor=float(sd.mean()),
               ratio=float(np.sqrt((d**2).mean()) / sd.mean()),
               split_bias=float(d.mean()))
    json.dump(out, open('spectrum/water_check.json', 'w'), indent=1)
    print('wrote spectrum/water_check.json')


if __name__ == '__main__':
    main()
