"""Gate A3 -- read the calibrated cross-bin instrument on real Water_Phantom data.

Two pixel populations, as specified:
    A: open-beam air              (air_L, air_R)   -- no detrending needed
    B: behind the water cylinder centre (deep_centre) -- harmonic detrend, 2 harmonics

Preprocessing is NOT reimplemented: the functions are imported from kill_test.py
so the numbers are produced by exactly the pipeline that was calibrated in A2.

Standard errors come from a bootstrap over VIEWS (not pixels). Pixels within an
ROI share the same views and the same drift reference, so a pixel-wise SE would
be optimistic; resampling views propagates that shared randomness.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'files'))
os.environ.setdefault('NVIEWS', '1440')
import kill_test as K

DATA = 'data/CalibrationPhantomData'
CAL = 'data/CalibrationTable'
PHANTOM = 'Water_Phantom'
NBOOT = int(os.environ.get('NBOOT', 200))

POPULATIONS = {
    'A_air': [r for r in K.ROIS if r[0] in ('air_L', 'air_R')],
    'B_water_centre': [r for r in K.ROIS if r[0] == 'deep_centre'],
}


def stats_for(T, H, good):
    """rho_LH, Cov(T,H)-Var(H), and Fano per bin -- medians over good pixels."""
    fT, fH = K.fano(T), K.fano(H)
    nontrivial, rho_LH, _ = K.cross_bin_terms(T, H)
    return dict(rho_LH=float(np.median(rho_LH[good])),
                nontrivial=float(np.median(nontrivial[good])),
                fano_T=float(np.median(fT[good])),
                fano_H=float(np.median(fH[good])))


def main():
    rois = list(K.ROIS)
    theta = K.load_view_angles(DATA, PHANTOM)
    pT = K.proj_paths(DATA, PHANTOM, 'Total')
    pH = K.proj_paths(DATA, PHANTOM, 'High')
    T_raw = K.load_rois(pT, rois, 'Total')
    H_raw = K.load_rois(pH, rois, 'High')

    sample = np.stack([np.fromfile(p, dtype='<u2').reshape(K.NROW, K.NCOL)
                       for p in pT[::144]])
    good_full = K.build_bad_mask(CAL, sample)
    del sample

    ref_L = K.drift_reference(T_raw['air_L'])
    ref_R = K.drift_reference(T_raw['air_R'])
    keep = np.abs(ref_L - np.median(ref_L)) < 5 * ref_L.std()
    print(f'drift: dropping {(~keep).sum()} outlier views; {keep.sum()} kept')

    out = {}
    rng = np.random.default_rng(0)
    for pop, plist in POPULATIONS.items():
        for name, r0, r1, c0, c1, do_detrend in plist:
            good = good_full[r0:r1, c0:c1]
            ref = ref_R if name == 'air_L' else ref_L
            T = K.normalise_drift(T_raw[name], ref)[keep]
            H = K.normalise_drift(H_raw[name], ref)[keep]
            th = theta[keep]

            # --- preprocessing-sensitivity: with and without the hard-gated detrend ---
            variants = {}
            base = stats_for(T, H, good)
            variants['no_detrend'] = base
            if do_detrend:
                Td, remT = K.harmonic_detrend(T, th)
                Hd, remH = K.harmonic_detrend(H, th)
                variants['detrend_2harm'] = stats_for(Td, Hd, good)
                use_T, use_H = Td, Hd
                var_removed = (float(remT), float(remH))
            else:
                use_T, use_H = T, H
                var_removed = (0.0, 0.0)

            main_stats = variants.get('detrend_2harm', base)

            # --- bootstrap over views ---
            nv = use_T.shape[0]
            boot = {k: [] for k in ('rho_LH', 'nontrivial', 'fano_T', 'fano_H')}
            for b in range(NBOOT):
                idx = rng.integers(0, nv, nv)
                s = stats_for(use_T[idx], use_H[idx], good)
                for k in boot:
                    boot[k].append(s[k])
            se = {k: float(np.std(v, ddof=1)) for k, v in boot.items()}

            out[name] = dict(population=pop, n_views=int(nv),
                             n_good=int(good.sum()),
                             rate=float(use_T.mean()),
                             hardening=float(use_H.mean() / use_T.mean()),
                             detrended=bool(do_detrend),
                             var_removed_T=var_removed[0], var_removed_H=var_removed[1],
                             **main_stats,
                             se={k: se[k] for k in se},
                             variants=variants)
            print(f"{name:14s} [{pop}]  rho_LH {main_stats['rho_LH']:+.5f} "
                  f"+- {se['rho_LH']:.5f}   Cov(T,H)-Var(H) {main_stats['nontrivial']:+.4f} "
                  f"+- {se['nontrivial']:.4f}   FanoT {main_stats['fano_T']:.4f} "
                  f"+- {se['fano_T']:.4f}   FanoH {main_stats['fano_H']:.4f} "
                  f"+- {se['fano_H']:.4f}", flush=True)
            if do_detrend:
                d = main_stats['rho_LH'] - base['rho_LH']
                print(f"               detrend changed rho_LH by {d:+.5f} "
                      f"({abs(d)/max(se['rho_LH'],1e-12):.1f} SE); "
                      f"var removed {100*var_removed[0]:.1f}% T / {100*var_removed[1]:.1f}% H")

    os.makedirs(f'{HERE}/results', exist_ok=True)
    json.dump(out, open(f'{HERE}/results/A3_real_data.json', 'w'), indent=1)
    print('\nwrote gateA/results/A3_real_data.json')


if __name__ == '__main__':
    main()
