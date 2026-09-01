"""Physics inputs for the two-bin spectrum fit.

Energy grid is fixed to the one the dataset itself ships: the vendor's
H2O/HA mass-attenuation tables are 160 points, and matching them against NIST
identifies the grid as 1..160 keV in 1 keV steps (water agrees to within 3.4%,
and the HA Ca K-edge lands between 4 and 5 keV as it must).

mu/rho comes from xraylib (NIST-derived, includes coherent scattering, as the
NIST mu/rho tables do). Phi(E) comes from spekpy: a real Tungsten-anode
bremsstrahlung model at the measured 80 kVp rather than an analytic stand-in.
"""
import numpy as np

E_GRID = np.arange(1, 161, dtype=float)          # keV, matches the shipped tables

DENSITY = {          # g/cm3
    'PMMA': 1.19,
    'Al': 2.699,
    'CdTe': 5.85,
    'Water': 1.000,
    'HA': 3.16,      # hydroxyapatite
}
FORMULA = {
    'PMMA': 'C5H8O2',
    'Al': 'Al',
    'CdTe': 'CdTe',
    'Water': 'H2O',
    'HA': 'Ca10P6O26H2',   # Ca10(PO4)6(OH)2
}

_MU_CACHE = {}


def mu_rho(material, E=E_GRID):
    """Mass attenuation coefficient [cm2/g] on the energy grid."""
    key = (material, len(E), float(E[0]), float(E[-1]))
    if key in _MU_CACHE:
        return _MU_CACHE[key]
    import xraylib
    f = FORMULA[material]
    out = np.array([xraylib.CS_Total_CP(f, float(e)) for e in E])
    _MU_CACHE[key] = out
    return out


def mu_lin(material, E=E_GRID):
    """Linear attenuation coefficient [1/mm] (slab thicknesses are in mm)."""
    return mu_rho(material, E) * DENSITY[material] / 10.0


def phi_spekpy(kvp=80.0, al_mm=0.5, extra_al_mm=0.0, E=E_GRID, th=12.0):
    """Tungsten bremsstrahlung fluence per keV bin, filtered.

    al_mm is the documented inherent filtration (AcqPara: '0.5mm Al');
    extra_al_mm is the fitted term standing in for housing, window and air path.
    """
    import spekpy as sp
    s = sp.Spek(kvp=kvp, th=th, dk=1.0, char=True)
    s.filter('Al', al_mm + max(extra_al_mm, 0.0))
    k, f = s.get_spectrum(edges=False)
    return np.interp(E, k, f, left=0.0, right=0.0)


def eta_cdte(d_mm, E=E_GRID):
    """Sensor absorption efficiency for a CdTe slab of thickness d (mm)."""
    return 1.0 - np.exp(-mu_lin('CdTe', E) * d_mm)


def threshold_response(E_thr, sigma, E=E_GRID):
    """Smoothed threshold: 0.5*erfc((E_thr-E)/(sigma*sqrt2)).

    sigma is the electronic-noise / charge-sharing dispersion of the
    discriminator, so the transition is an erfc rather than a step.
    """
    from scipy.special import erfc
    s = max(float(sigma), 1e-6)
    return 0.5 * erfc((E_thr - E) / (s * np.sqrt(2.0)))


if __name__ == '__main__':
    import h5py
    with h5py.File('data/CalibrationTable/H2O_massAttenuationCoeff.mat', 'r') as h:
        shipped = np.array(h['H2O_massAttenuationCoeff']).ravel()
    ours = mu_rho('Water')
    m = (E_GRID >= 10) & (E_GRID <= 120)
    r = ours[m] / shipped[m]
    print('xraylib water vs shipped table, 10-120 keV:')
    print(f'  ratio mean {r.mean():.4f}  min {r.min():.4f}  max {r.max():.4f}')
    print(f"{'E':>5}{'xraylib':>10}{'shipped':>10}{'ratio':>8}")
    for e in (15, 20, 30, 50, 80, 120):
        i = int(e) - 1
        print(f'{e:>5}{ours[i]:>10.4f}{shipped[i]:>10.4f}{ours[i]/shipped[i]:>8.3f}')
    print('\nlinear mu (1/mm) at 30 keV:')
    for mat in ('PMMA', 'Al', 'CdTe', 'Water', 'HA'):
        print(f'  {mat:6s} {mu_lin(mat)[29]:.5f}')
    print('\nspekpy 80 kVp W + 0.5 mm Al:')
    p = phi_spekpy()
    print(f'  total fluence {p.sum():.4g},  mean energy '
          f'{(p*E_GRID).sum()/p.sum():.2f} keV,  peak at {E_GRID[p.argmax()]:.0f} keV')
    print(f'  CdTe eta at 2 mm: 20 keV {eta_cdte(2.0)[19]:.3f}, '
          f'60 keV {eta_cdte(2.0)[59]:.3f}, 100 keV {eta_cdte(2.0)[99]:.3f}')
