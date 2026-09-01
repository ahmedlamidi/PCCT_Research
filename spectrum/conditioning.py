"""How much can the two bins actually distinguish? Fitted detector vs nominal thresholds.

This is a test with a prediction, not a survey. Threshold blur and the energy-loss
tail both move counts across the 30 keV boundary in the wrong direction, which
makes the two bins spectrally MORE alike than nominal 15/30 keV thresholds imply.
So the prediction going in is:

    the real detector separates two materials WORSE than its nominal thresholds suggest.

Measured by the Cramer-Rao bound on a two-material decomposition. For one
measurement of (L, H) at basis thicknesses a, with Poisson counts:

    J      = d(lam_L, lam_H)/da
    Jw     = diag(1/sqrt(lam)) J          whitened
    F      = Jw' Jw                        Fisher information
    sd     = sqrt(diag(F^-1))              CRLB per material
    kappa  = cond(Jw)                      how close the two bins are to redundant

kappa is the honest headline: kappa -> infinity means the two bins carry the same
information and the decomposition is singular. The comparison is fitted-vs-ideal
at IDENTICAL beam, sensor and flux -- only the threshold sharpness and the tail
differ -- so the ratio isolates exactly the effect being tested.
"""
import sys, json
import numpy as np

sys.path.insert(0, 'spectrum')
from forward_model import ForwardModel, ideal_model

BASIS = ['Water', 'HA']
# walnut-scale operating points (mm of each basis material)
POINTS = [('kernel, thin', {'Water': 10.0, 'HA': 0.0}),
          ('kernel, mid', {'Water': 25.0, 'HA': 0.0}),
          ('kernel, thick', {'Water': 40.0, 'HA': 0.0}),
          ('shell, thin', {'Water': 20.0, 'HA': 0.5}),
          ('shell, mid', {'Water': 25.0, 'HA': 1.5}),
          ('shell, thick', {'Water': 30.0, 'HA': 3.0})]


def evaluate(fm, points=POINTS, basis=BASIS):
    out = []
    for name, a in points:
        sd, kappa = fm.crlb(a, basis)
        lam = fm.predict(a)
        out.append(dict(point=name, a=a, kappa=float(kappa),
                        sd_water=float(sd[0]), sd_ha=float(sd[1]),
                        lam_L=lam[0], lam_H=lam[1]))
    return out


def main():
    fm = ForwardModel.load('spectrum/forward_model.npz')
    idl = ideal_model(fm)
    real, ideal = evaluate(fm), evaluate(idl)

    print(f'basis: {BASIS[0]} / {BASIS[1]}   (CRLB sd per single (L,H) measurement, mm)')
    print(f"\n{'operating point':16s}{'kappa real':>12}{'kappa ideal':>13}{'ratio':>8}"
          f"{'sd_HA real':>12}{'sd_HA ideal':>13}{'ratio':>8}")
    kr, ki = [], []
    for r, i in zip(real, ideal):
        kr.append(r['kappa']); ki.append(i['kappa'])
        print(f"{r['point']:16s}{r['kappa']:>12.1f}{i['kappa']:>13.1f}"
              f"{r['kappa']/i['kappa']:>8.2f}{r['sd_ha']:>12.4g}"
              f"{i['sd_ha']:>13.4g}{r['sd_ha']/i['sd_ha']:>8.2f}")
    kr, ki = np.array(kr), np.array(ki)
    print(f"\nkappa penalty from real thresholds: "
          f"median {np.median(kr/ki):.2f}x   range {(kr/ki).min():.2f}-{(kr/ki).max():.2f}x")
    sdr = np.array([r['sd_ha'] for r in real]); sdi = np.array([i['sd_ha'] for i in ideal])
    print(f"HA-thickness noise penalty:          "
          f"median {np.median(sdr/sdi):.2f}x   range {(sdr/sdi).min():.2f}-{(sdr/sdi).max():.2f}x")
    verdict = 'CONFIRMED' if np.median(kr / ki) > 1.05 else \
              ('REFUTED' if np.median(kr / ki) < 0.95 else 'NO DIFFERENCE')
    print(f"\nprediction was: real worse than nominal.  -> {verdict}")

    json.dump(dict(basis=BASIS, real=real, ideal=ideal,
                   kappa_ratio_median=float(np.median(kr / ki)),
                   sd_ratio_median=float(np.median(sdr / sdi)),
                   verdict=verdict),
              open('spectrum/conditioning.json', 'w'), indent=1)
    print('wrote spectrum/conditioning.json')


if __name__ == '__main__':
    main()
