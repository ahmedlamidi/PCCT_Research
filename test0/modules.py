"""Is the leftover structure organised by detector module? And a corrected D4.

D4 note: the brief says only Total-vs-High is informative, but High is a SUBSET
of Total, so Cov(T,H)=Var(H) and corr(T,H)=sqrt(H/T) even under pure counting
statistics. The genuinely informative pair is Low vs High: they are disjoint
count sets, hence independent under the null, so any correlation is physics.
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import core, analysis

TILE = 129
MIDROW = 252          # detector midline seen in the residual maps


def module_id(shape, coff=0, roff=0):
    rr, cc = np.meshgrid(np.arange(shape[0]) + roff, np.arange(shape[1]) + coff, indexing='ij')
    return ((cc // TILE) * 2 + (rr >= MIDROW)).ravel()


def main(full=True):
    outdir = 'test0_results_full' if full else 'test0_results'
    allc = core.parse_combos(); T, H = core.load_raw(allc)
    good_full, _ = core.build_mask(T, H, verbose=False)
    i0 = [i for i, c in enumerate(allc) if c[0] == core.SELF_REF_COMBO][0]
    air_T, air_H = T[i0].copy(), H[i0].copy()
    keep = [i for i, c in enumerate(allc) if c[0] != core.SELF_REF_COMBO]
    combos = [allc[i] for i in keep]; T, H = T[keep], H[keep]
    shape = (core.NR, core.NC); good = good_full
    tP = np.array([c[1] for c in combos]); tA = np.array([c[2] for c in combos])
    data = {b: core.attenuation_and_noise(T, H, air_T, air_H, b) for b in core.BINS}
    splits = analysis.make_splits(combos, data['total'][0][:, good].mean(1))
    mid = module_id(shape)[good]
    nmod = mid.max() + 1
    M = np.zeros((nmod, mid.size), np.float32)
    M[mid, np.arange(mid.size)] = 1.0
    cnt = M.sum(1)

    out = {}
    for deg in (2, 3):
        X = core.design(tP, tA, deg)
        for sp in ('random_seed0', 'extrapolate_thick', 'leave_AL_5'):
            tr, te = splits[sp]
            rec = {}
            R = {}
            for b in core.BINS:
                P, V = data[b]
                resid, W = core.fit_predict(X, P, tr, te)
                sd = core.predicted_resid_std(V, tr, te, W)
                r = resid[:, good].astype(np.float64); s = sd[:, good].astype(np.float64)
                r = r - r.mean(1, keepdims=True)              # per-acquisition offset
                z0 = float(np.sqrt(((r / s)**2).mean()))
                mmean = (r @ M.T) / cnt                        # (n_test, nmod)
                rmod = r - mmean[:, mid]                       # remove per-module offset
                z1 = float(np.sqrt(((rmod / s)**2).mean()))
                var_mod = float((r**2).mean() - (rmod**2).mean())
                rec[b] = dict(z_rms_offset_removed=z0, z_rms_module_removed=z1,
                              module_var_frac=var_mod / float((r**2).mean()))
                R[b] = (r, rmod, s)
            # corrected D4: Low vs High (independent under the null)
            for tag, (a, c) in (('offset_removed', (0, 0)), ('module_removed', (1, 1))):
                rL = R['low'][a].ravel(); rH = R['high'][c].ravel(); rT = R['total'][a].ravel()
                rec['D4_' + tag] = dict(
                    r_LowHigh=float(np.corrcoef(rL, rH)[0, 1]),
                    r_TotalHigh=float(np.corrcoef(rT, rH)[0, 1]),
                    r_TotalHigh_noise_only=float(np.sqrt(
                        (H[:, good][te].astype(np.float64) /
                         T[:, good][te].astype(np.float64)).mean())))
            out[f'{sp}_deg{deg}'] = rec
        print(f'  deg{deg} done')
    json.dump(out, open(f'{outdir}/modules_and_D4.json', 'w'), indent=1)
    return out


if __name__ == '__main__':
    main()
