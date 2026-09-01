"""Does the mandated harmonic detrend REMOVE the effect rather than clean it?

The concern is specific. The phantom is ~1.2 mm off-axis, so its shadow slides
sinusoidally with gantry angle. That sinusoid is the ONLY thing making the object
vary view-to-view -- so path length through water is modulated at the gantry
frequency, and any genuinely object-dependent detector error is modulated at the
SAME frequency. Regressing out {1, cos k.theta, sin k.theta} therefore removes
the geometric wobble and any real object-dependent spectral effect together.

Symptom already on record: at deep_centre the 2-harmonic detrend removes only
1.3% of variance yet moves rho_LH by 23 sigma (+0.0056 -> +0.0007). A tiny but
COHERENT, cross-bin-correlated component is being taken out. Geometry and physics
both look like that.

Discriminator: sweep the number of harmonics. Pure geometry is low-order in
theta, so rho_LH should drop and then PLATEAU once the wobble is captured. If it
keeps sliding with every extra harmonic, the regression is eating signal.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'files'))
os.environ.setdefault('NVIEWS', '1440')
import kill_test as K

DATA = 'data/CalibrationPhantomData'
CAL = 'data/CalibrationTable'
PHANTOM = 'Water_Phantom'
HARMS = [0, 1, 2, 3, 4, 6, 8]
ROI_NAMES = ('air_L', 'deep_centre', 'medium_L', 'edge_L')


def main():
    rois = [r for r in K.ROIS if r[0] in ROI_NAMES] + \
           [r for r in K.ROIS if r[0] == 'air_R']
    theta = K.load_view_angles(DATA, PHANTOM)
    pT = K.proj_paths(DATA, PHANTOM, 'Total')
    pH = K.proj_paths(DATA, PHANTOM, 'High')
    T_raw = K.load_rois(pT, rois, 'Total')
    H_raw = K.load_rois(pH, rois, 'High')
    sample = np.stack([np.fromfile(p, dtype='<u2').reshape(K.NROW, K.NCOL)
                       for p in pT[::144]])
    good_full = K.build_mask if False else K.build_bad_mask(CAL, sample)
    del sample
    ref_L = K.drift_reference(T_raw['air_L'])
    ref_R = K.drift_reference(T_raw['air_R'])
    keep = np.abs(ref_L - np.median(ref_L)) < 5 * ref_L.std()

    out = {}
    print(f"{'ROI':13s}{'nharm':>6s}{'var_removed%':>14s}{'rho_LH':>11s}{'d(rho)':>10s}")
    for name, r0, r1, c0, c1, do_detrend in rois:
        if name not in ROI_NAMES:
            continue
        good = good_full[r0:r1, c0:c1]
        ref = ref_R if name == 'air_L' else ref_L
        T0 = K.normalise_drift(T_raw[name], ref)[keep]
        H0 = K.normalise_drift(H_raw[name], ref)[keep]
        th = theta[keep]
        rows, prev = [], None
        for nh in HARMS:
            if nh == 0:
                T, H, rem = T0, H0, 0.0
            else:
                T, remT = K.harmonic_detrend(T0, th, n_harm=nh)
                H, _ = K.harmonic_detrend(H0, th, n_harm=nh)
                rem = remT
            _, rho, _ = K.cross_bin_terms(T, H)
            r = float(np.median(rho[good]))
            d = '' if prev is None else f'{r-prev:+.5f}'
            print(f'{name:13s}{nh:>6d}{100*rem:>14.2f}{r:>+11.5f}{d:>10s}')
            rows.append(dict(n_harm=nh, var_removed=float(rem), rho_LH=r))
            prev = r
        out[name] = rows
        print()
    os.makedirs('gateA/results', exist_ok=True)
    json.dump(out, open('gateA/results/harmonic_sweep.json', 'w'), indent=1)
    print('wrote gateA/results/harmonic_sweep.json')


if __name__ == '__main__':
    main()
