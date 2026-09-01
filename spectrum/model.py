"""Forward model and stage-1 global spectrum fit.

    lam_b(a) = SUM_E  S_b(E) * exp(-a_PMMA*mu_PMMA(E) - a_Al*mu_Al(E))

with a physical parameterisation, not a free per-energy vector:

    Phi(E)   80 kVp W bremsstrahlung (spekpy) through 0.5 mm Al, times a fitted
             extra-Al term for housing/window/air path. Filtration is exactly
             multiplicative, so spekpy is called ONCE and the extra term applied
             analytically as exp(-mu_Al*extra).
    eta(E)   1 - exp(-mu_CdTe(E)*d)          sensor absorption
    R_b(E)   0.5*erfc((E_b-E)/(sigma_b*sqrt2))   smoothed threshold

The two bins are DISJOINT, which the band response has to respect:

    S_H(E) = t*Phi*eta*R_H(E)                counts above the high threshold
    S_L(E) = t*Phi*eta*(R_L(E) - R_H(E))     counts BETWEEN the thresholds

7 free parameters against 110 measurements.

Likelihood: the slabs are 600-frame averages quantised to uint16, so
Var = lam/600 + q/12 -- a scaled Poisson, not Poisson. Treating them as raw
Poisson counts would inflate every weight by 600x.
"""
import numpy as np
from scipy.optimize import least_squares

from physics import E_GRID, mu_lin, phi_spekpy, eta_cdte, threshold_response

# The model is LINEAR in the flux scale t, so t is profiled out analytically
# (Poisson MLE for a common scale: t = sum(y)/sum(shape)) rather than searched.
# That removes a parameter whose dynamic range spans orders of magnitude and
# whose bounds are easy to set wrongly -- the first attempt railed on them.
# scale_P / scale_A are effective-thickness factors: the slabs' true density and
# thickness need not equal nominal, and a mis-scaled attenuation path shows up as a
# residual that is MONOTONE in that material's thickness -- which is exactly what
# the 6-parameter fit left behind (-4.9% at 0 mm PMMA rising to +8.5% at 40 mm).
# f_tail is the incomplete-charge-collection fraction: in a real CdTe PCD a
# photon of energy E frequently deposits LESS than E in the struck pixel (charge
# shared with neighbours, or escape), so the effective bin response is softer than
# any threshold-on-true-energy model can produce. Without it the fit compensated
# by railing d_cdte at an unphysical 6 mm and softening sigma_H to 11 keV.
# Tail model: a fraction f of events register at an energy uniform on [0, E], so
#   R_eff(E) = (1-f) R(E) + f * mean_{E' <= E} R(E')
# d_cdte is now bounded to a physically sensible 0.3-3.0 mm for CdTe.
PARAM_NAMES = ['extra_al_mm', 'd_cdte_mm', 'E_L', 'E_H', 'sigma_L', 'sigma_H',
               'scale_P', 'scale_A', 'f_tail']
BOUNDS_LO = [0.0, 0.30, 5.0, 20.0, 0.20, 0.20, 0.80, 0.80, 0.0]
BOUNDS_HI = [12.0, 3.00, 28.0, 45.0, 12.0, 12.0, 1.25, 1.25, 0.9]
P0 = [1.0, 1.0, 15.0, 30.0, 2.0, 2.0, 1.0, 1.0, 0.2]
NFRAMES = 600

_BASE = {}


def base_phi():
    """spekpy 80 kVp W through the documented 0.5 mm Al, computed once."""
    if 'phi' not in _BASE:
        _BASE['phi'] = phi_spekpy(kvp=80.0, al_mm=0.5, extra_al_mm=0.0)
        _BASE['mu_al'] = mu_lin('Al')
        _BASE['mu_pmma'] = mu_lin('PMMA')
    return _BASE


def spectra_shape(p):
    """S_L(E), S_H(E) at unit flux scale (t = 1)."""
    extra_al, d, E_L, E_H, sL, sH = p[:6]
    b = base_phi()
    phi = b['phi'] * np.exp(-b['mu_al'] * max(extra_al, 0.0))
    eta = eta_cdte(d)
    R_L = threshold_response(E_L, sL)
    R_H = threshold_response(E_H, sH)
    f = p[8] if len(p) > 8 else 0.0
    if f > 0:
        n = np.arange(1, len(R_L) + 1)
        R_L = (1 - f) * R_L + f * (np.cumsum(R_L) / n)
        R_H = (1 - f) * R_H + f * (np.cumsum(R_H) / n)
    band = np.clip(R_L - R_H, 0.0, None)          # Low = between the thresholds
    return phi * eta * band, phi * eta * R_H


def _shape(p, tP, tA):
    S_L, S_H = spectra_shape(p)
    b = base_phi()
    sP, sA = (p[6], p[7]) if len(p) > 6 else (1.0, 1.0)
    A = np.exp(-np.outer(tP * sP, b['mu_pmma']) - np.outer(tA * sA, b['mu_al']))
    return A @ S_L, A @ S_H


def t_hat(p, tP, tA, yL, yH):
    mL, mH = _shape(p, tP, tA)
    den = mL.sum() + mH.sum()
    return (yL.sum() + yH.sum()) / max(den, 1e-30)


def predict(p, tP, tA, t=None, yL=None, yH=None):
    """lam_L, lam_H for each (PMMA, Al) thickness pair, in mm."""
    mL, mH = _shape(p, tP, tA)
    if t is None:
        t = t_hat(p, tP, tA, yL, yH)
    return t * mL, t * mH


def spectra(p, t):
    S_L, S_H = spectra_shape(p)
    return t * S_L, t * S_H


def residuals(p, tP, tA, yL, yH):
    lL, lH = predict(p, tP, tA, yL=yL, yH=yH)
    lL = np.maximum(lL, 1e-9); lH = np.maximum(lH, 1e-9)
    vL = lL / NFRAMES + 2 / 12.0        # Low is a difference of two quantised bins
    vH = lH / NFRAMES + 1 / 12.0
    return np.concatenate([(yL - lL) / np.sqrt(vL), (yH - lH) / np.sqrt(vH)])


def fit(tP, tA, yL, yH, p0=None, seed=0):
    p0 = list(P0 if p0 is None else p0)
    if seed:
        rng = np.random.default_rng(seed)
        p0 = [np.clip(v + rng.normal(0, s), lo, hi) for v, s, lo, hi in
              zip(p0, [0.8, 0.4, 2.0, 2.5, 1.0, 1.0, 0.04, 0.04, 0.12],
                  BOUNDS_LO, BOUNDS_HI)]
    r = least_squares(residuals, p0, bounds=(BOUNDS_LO, BOUNDS_HI),
                      args=(tP, tA, yL, yH), method='trf', x_scale='jac',
                      max_nfev=20000)
    return r


def report(r, tP, tA, yL, yH):
    p = r.x
    lL, lH = predict(p, tP, tA, yL=yL, yH=yH)
    n = 2 * len(tP)
    chi2 = float((r.fun ** 2).sum())
    fr = np.concatenate([(yL - lL) / yL, (yH - lH) / yH])
    return dict(params={k: float(v) for k, v in zip(PARAM_NAMES, p)},
                t=float(t_hat(p, tP, tA, yL, yH)),
                chi2=chi2, dof=n - len(p), chi2_red=chi2 / (n - len(p)),
                frac_rms=float(np.sqrt((fr ** 2).mean())),
                frac_max=float(np.abs(fr).max()))
