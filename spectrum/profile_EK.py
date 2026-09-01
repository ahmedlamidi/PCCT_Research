"""Profile likelihood in the escape switch-on energy E_K.

The single-shot fit put E_K at 27.02 keV, 0.31 keV from the Cd K-edge -- but it
also drove E_L and E_fluor to the same value (18.13), which is the signature of a
degeneracy rather than a measurement. A profile settles it: FIX E_K, refit
everything else, and see whether the data actually prefers the K-edge or whether
chi2 is flat and 27.02 was an accident of the optimiser.

E_K is removed from the parameter vector rather than pinned by collapsing its
bounds -- the latter breaks the jacobian scaling and produces numerical garbage.
"""
import sys, json
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, 'spectrum')
import slabdata
import escape_test as ET

IDX_EK = 9
IDX_FL = 10
SCALES = [0.6, 0.5, 2.0, 2.0, 1.0, 1.0, 0.04, 0.04, 0.15, 0.0, 5.0]


def fit_at(EK, tP, tA, yL, yH, fix_fluor=None, nstart=8):
    free = [i for i in range(11) if i != IDX_EK
            and not (fix_fluor is not None and i == IDX_FL)]
    lo = [ET.ESC_LO[i] for i in free]
    hi = [ET.ESC_HI[i] for i in free]
    base = list(ET.ESC_P0)

    def expand(x):
        q = list(base)
        q[IDX_EK] = EK
        if fix_fluor is not None:
            q[IDX_FL] = fix_fluor
        for k, i in enumerate(free):
            q[i] = x[k]
        return np.array(q)

    def res(x):
        return ET.esc_resid(expand(x), tP, tA, yL, yH)

    best = None
    for s in range(nstart):
        rng = np.random.default_rng(s)
        x0 = [float(np.clip(base[i] + (0.0 if s == 0 else rng.normal(0, SCALES[i])),
                            a, b)) for i, a, b in zip(free, lo, hi)]
        r = least_squares(res, x0, bounds=(lo, hi), max_nfev=6000)
        if best is None or r.cost < best.cost:
            best = r
    return float((best.fun ** 2).sum()), expand(best.x)


def main():
    d = slabdata.build()
    keep, _ = slabdata.fit_split(d)
    tP, tA = d['tP'][keep], d['tA'][keep]
    yL, yH = d['lam_L'][keep], d['lam_H'][keep]
    grid = np.arange(16.0, 42.1, 1.0)
    out = {}
    for tag, ff in (('E_fluor_free', None), ('E_fluor_fixed_CdKa_23.17', 23.17)):
        chi = np.empty(len(grid))
        pars = []
        for j, EK in enumerate(grid):
            chi[j], pj = fit_at(EK, tP, tA, yL, yH, ff)
            pars.append(pj)
            print(f'  [{tag}] E_K={EK:5.1f}  chi2={chi[j]:9.1f}', flush=True)
        i = int(chi.argmin())
        p = pars[i]
        print(f'\n=== {tag} ===')
        print(f'  MINIMUM at E_K = {grid[i]:.1f} keV   chi2 = {chi[i]:.1f}')
        print(f'  Cd K-edge 26.71   Te K-edge 31.81')
        print(f'  chi2 range over scan = {chi.max()-chi.min():.1f} '
              f'(flat would mean E_K is not measured)')
        print(f'  at minimum: E_L {p[2]:.2f}  E_H {p[3]:.2f}  scale_A {p[7]:.3f} '
              f' f0 {p[8]:.3f}  E_fluor {p[10]:.2f}')
        out[tag] = dict(grid=grid.tolist(), chi2=chi.tolist(),
                        argmin=float(grid[i]), chi2_min=float(chi[i]),
                        chi2_range=float(chi.max() - chi.min()),
                        params_at_min=[float(v) for v in p])
    json.dump(out, open('spectrum/profile_EK.json', 'w'), indent=1)
    print('\nwrote spectrum/profile_EK.json')


if __name__ == '__main__':
    main()
