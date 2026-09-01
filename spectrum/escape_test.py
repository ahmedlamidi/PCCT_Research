"""Escape switch-on test: is the energy-loss tail K-escape, or generic sharing?

The stage-1 fit needed 41% of events to register below their true energy. Two
mechanisms could do that, and they make DIFFERENT predictions:

  charge sharing  charge splits into a neighbouring pixel. Possible at ANY energy,
                  and it necessarily correlates neighbouring pixels.
  K-escape        a Cd or Te K-fluorescence photon leaves the sensor. Energy is
                  lost with NO neighbour involvement -- but it is impossible below
                  the K-edge, and the lost energy is a FIXED quantum, not a smear.

    Cd K-edge 26.71 keV, Cd K-alpha 23.17 keV
    Te K-edge 31.81 keV, Te K-alpha 27.47 keV

So the escape model is a step, not a smear:

    R_eff(E) = (1-f(E)) R(E)  +  f(E) R(E - E_fluor),
    f(E) = f0 * 0.5 erfc((E_K - E)/(w sqrt2))          w = 1 keV

E_K is left FREE over 15-45 keV. If the tail is really escape, the fit has to put
E_K at a K-edge on its own -- nothing in the data or the parameterisation points
there. That is the whole test. Landing at 26.7 keV would be the payoff; landing
anywhere else, or no improvement in fit, would say the tail is not escape.
"""
import os, sys, json
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erfc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slabdata, model as M
from physics import E_GRID, mu_lin, eta_cdte, threshold_response

NFRAMES = 600
W_SWITCH = 1.0        # keV, width of the switch-on (fixed; the test is on E_K)

# uniform-tail model (current stage 1): 9 params
# escape model: 8 shared + f0, E_K, E_fluor = 11 params
ESC_NAMES = ['extra_al_mm', 'd_cdte_mm', 'E_L', 'E_H', 'sigma_L', 'sigma_H',
             'scale_P', 'scale_A', 'f0', 'E_K', 'E_fluor']
ESC_LO = [0.0, 0.30, 5.0, 20.0, 0.20, 0.20, 0.80, 0.80, 0.0, 15.0, 10.0]
ESC_HI = [12.0, 3.00, 28.0, 45.0, 12.0, 12.0, 1.25, 1.25, 0.95, 45.0, 35.0]
ESC_P0 = [0.5, 1.5, 14.0, 30.0, 1.0, 2.0, 0.92, 1.00, 0.4, 27.0, 24.0]


def esc_shape(p):
    extra_al, d, E_L, E_H, sL, sH, _, _, f0, E_K, E_fl = p
    b = M.base_phi()
    phi = b['phi'] * np.exp(-b['mu_al'] * max(extra_al, 0.0))
    eta = eta_cdte(d)
    f = f0 * 0.5 * erfc((E_K - E_GRID) / (W_SWITCH * np.sqrt(2.0)))
    out = []
    for Eb, sb in ((E_L, sL), (E_H, sH)):
        R = threshold_response(Eb, sb)
        R_esc = threshold_response(Eb, sb, E_GRID - E_fl)   # degraded deposit
        out.append((1 - f) * R + f * R_esc)
    R_L, R_H = out
    band = np.clip(R_L - R_H, 0.0, None)
    return phi * eta * band, phi * eta * R_H


def esc_predict(p, tP, tA, yL, yH):
    S_L, S_H = esc_shape(p)
    b = M.base_phi()
    A = np.exp(-np.outer(tP * p[6], b['mu_pmma']) - np.outer(tA * p[7], b['mu_al']))
    mL, mH = A @ S_L, A @ S_H
    t = (yL.sum() + yH.sum()) / max(mL.sum() + mH.sum(), 1e-30)
    return t * mL, t * mH, t


def esc_resid(p, tP, tA, yL, yH):
    lL, lH, _ = esc_predict(p, tP, tA, yL, yH)
    lL = np.maximum(lL, 1e-9); lH = np.maximum(lH, 1e-9)
    return np.concatenate([(yL - lL) / np.sqrt(lL / NFRAMES + 2 / 12.0),
                           (yH - lH) / np.sqrt(lH / NFRAMES + 1 / 12.0)])


def main():
    d = slabdata.build(); keep, zero = slabdata.fit_split(d)
    tP, tA = d['tP'][keep], d['tA'][keep]
    yL, yH = d['lam_L'][keep], d['lam_H'][keep]

    # ---- model A: uniform tail (stage 1) ----
    pA = np.load('spectrum/fit_p.npy')
    rA = M.residuals(pA, tP, tA, yL, yH)
    chiA = float((rA ** 2).sum())

    # ---- model B: escape, E_K free ----
    best = None
    for s in range(40):
        rng = np.random.default_rng(s)
        p0 = ESC_P0 if s == 0 else [
            float(np.clip(v + rng.normal(0, sc), lo, hi)) for v, sc, lo, hi in
            zip(ESC_P0, [0.6, 0.5, 2.0, 2.0, 1.0, 1.0, 0.04, 0.04, 0.15, 6.0, 5.0],
                ESC_LO, ESC_HI)]
        r = least_squares(esc_resid, p0, bounds=(ESC_LO, ESC_HI),
                          args=(tP, tA, yL, yH), x_scale='jac', max_nfev=20000)
        if best is None or r.cost < best.cost:
            best = r
    pB = best.x
    chiB = float((best.fun ** 2).sum())
    lL, lH, t = esc_predict(pB, tP, tA, yL, yH)
    frac = np.concatenate([(yL - lL) / yL, (yH - lH) / yH])

    nA, nB = len(pA), len(pB)
    dofA, dofB = 2 * len(tP) - nA, 2 * len(tP) - nB
    print(f'{"model":28s}{"k":>4}{"chi2":>12}{"chi2/dof":>11}{"frac rms %":>12}{"AIC":>10}')
    frA = np.concatenate([(yL - M.predict(pA, tP, tA, yL=yL, yH=yH)[0]) / yL,
                          (yH - M.predict(pA, tP, tA, yL=yL, yH=yH)[1]) / yH])
    print(f'{"A uniform tail":28s}{nA:>4}{chiA:>12.1f}{chiA/dofA:>11.2f}'
          f'{100*np.sqrt((frA**2).mean()):>12.3f}{chiA+2*nA:>10.1f}')
    print(f'{"B K-escape (E_K free)":28s}{nB:>4}{chiB:>12.1f}{chiB/dofB:>11.2f}'
          f'{100*np.sqrt((frac**2).mean()):>12.3f}{chiB+2*nB:>10.1f}')

    print(f'\n--- THE TEST ---')
    print(f'  fitted switch-on E_K   = {pB[9]:.2f} keV')
    print(f'  Cd K-edge              = 26.71 keV   (miss {abs(pB[9]-26.71):.2f})')
    print(f'  Te K-edge              = 31.81 keV   (miss {abs(pB[9]-31.81):.2f})')
    print(f'  fitted escape quantum  = {pB[10]:.2f} keV')
    print(f'  Cd K-alpha 23.17 / Te K-alpha 27.47 keV')
    print(f'  escape fraction f0     = {pB[8]:.3f}')
    print('\nfitted parameters (escape model):')
    for k, v in zip(ESC_NAMES, pB):
        print(f'  {k:14s}{v:>10.4f}')

    np.save('spectrum/fit_p_escape.npy', pB)
    json.dump(dict(chi2_uniform=chiA, chi2_escape=chiB, k_uniform=nA, k_escape=nB,
                   aic_uniform=chiA + 2 * nA, aic_escape=chiB + 2 * nB,
                   E_K=float(pB[9]), E_fluor=float(pB[10]), f0=float(pB[8]),
                   cd_k_edge=26.71, te_k_edge=31.81,
                   params={k: float(v) for k, v in zip(ESC_NAMES, pB)}, t=float(t)),
              open('spectrum/escape_test.json', 'w'), indent=1)
    print('\nwrote spectrum/escape_test.json')


if __name__ == '__main__':
    main()
