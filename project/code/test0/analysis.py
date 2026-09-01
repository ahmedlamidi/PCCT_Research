"""Test 0 diagnostics D1-D5 with a pre-registered decision rule."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import core

RNG_NULL = np.random.default_rng(12345)


def gram_svals(R):
    """Singular values of a tall (n_pix, n_test) matrix via its small Gram matrix."""
    G = (R.T @ R).astype(np.float64)
    ev = np.linalg.eigvalsh(G)[::-1]
    return np.clip(ev, 0, None)          # these are s**2


def d2_metrics(R):
    lam = gram_svals(R)
    tot = lam.sum()
    if tot <= 0:
        return dict(top5_frac=float('nan'), eff_rank=float('nan'), svals=[])
    top5 = float(lam[:5].sum() / tot)
    eff = float(tot**2 / (lam**2).sum())          # participation ratio
    return dict(top5_frac=top5, eff_rank=eff, svals=np.sqrt(lam).tolist())


def run_split(P, var, X, train, test, good):
    """Reports raw metrics and metrics after removing the per-combo global offset.

    Each slab combo is a separate step-and-shoot acquisition, so tube-output drift
    appears as one flat number per combo. That is an acquisition artifact, not
    spatial detector physics, and it would trivially satisfy a low-rank test, so
    the spatially-informative metrics are the centred ones. The null is centred
    identically so the comparison is fair.
    """
    resid, W = core.fit_predict(X, P, train, test)
    sd = core.predicted_resid_std(var, train, test, W)
    r, s = resid[:, good], sd[:, good]
    f64 = lambda a: a.astype(np.float64)
    off_vec = f64(r).mean(1, keepdims=True)
    rc = f64(r) - off_vec
    # whiten by the predicted noise std: under the null z is exactly iid N(0,1),
    # which removes the heteroscedasticity that otherwise inflates the D2 null
    # (thick combos are far noisier than thin ones).
    z = f64(r) / f64(s)
    zc = rc / f64(s)
    zc = zc - zc.mean(1, keepdims=True)
    nullz = RNG_NULL.standard_normal(zc.shape)
    nullz = nullz - nullz.mean(1, keepdims=True)
    rms = float(np.sqrt((f64(r)**2).mean()))
    rms_c = float(np.sqrt((rc**2).mean()))
    floor = float(np.sqrt((f64(s)**2).mean()))
    off = float(np.sqrt((off_vec**2).mean()))
    return dict(rms=rms, rms_centred=rms_c, floor=floor, offset_rms=off,
                ratio=rms / floor, ratio_centred=rms_c / floor,
                z_rms=float(np.sqrt((z**2).mean())),
                z_rms_centred=float(np.sqrt((zc**2).mean())),
                offset_var_frac=float((off_vec**2).mean() / (f64(r)**2).mean()),
                d2_white=d2_metrics(zc.T), d2_white_null=d2_metrics(nullz.T)), r, s


def make_splits(combos, P_mean):
    """combos: list of (name,tP,tA) already excluding the self-referential one."""
    n = len(combos)
    sp = {}
    for seed in range(5):
        rng = np.random.default_rng(seed)
        te = np.sort(rng.choice(n, 12, replace=False))
        sp[f'random_seed{seed}'] = (np.setdiff1d(np.arange(n), te), te)
    order = np.argsort(P_mean)                      # rank by actual attenuation
    half = n // 2
    sp['extrapolate_thick'] = (np.sort(order[:half]), np.sort(order[half:]))
    tA = np.array([c[2] for c in combos])
    for a in sorted(set(tA[tA > 0])):
        te = np.where(tA == a)[0]
        sp[f'leave_AL_{a:g}'] = (np.setdiff1d(np.arange(n), te), te)
    return sp


def main(crop=True, outdir='test0_results'):
    os.makedirs(outdir, exist_ok=True)
    allc = core.parse_combos()
    T, H = core.load_raw(allc)
    good_full, mrep = core.build_mask(T, H)

    keep = [i for i, c in enumerate(allc) if c[0] != core.SELF_REF_COMBO]
    combos = [allc[i] for i in keep]
    print(f"  using {len(combos)} combos (excluded self-referential {core.SELF_REF_COMBO})")

    air_T = T[[i for i, c in enumerate(allc) if c[0] == core.SELF_REF_COMBO][0]].copy()
    air_H = H[[i for i, c in enumerate(allc) if c[0] == core.SELF_REF_COMBO][0]].copy()
    T, H = T[keep], H[keep]

    if crop:
        r0, r1, c0, c1 = 124, 380, 775, 1287
        sel = np.zeros((core.NR, core.NC), bool); sel[r0:r1, c0:c1] = True
        sel = sel.ravel()
        shape = (r1 - r0, c1 - c0)
    else:
        sel = np.ones(core.NP, bool); shape = (core.NR, core.NC)

    T, H, air_T, air_H = T[:, sel], H[:, sel], air_T[sel], air_H[sel]
    good = good_full[sel]
    print(f"  region {shape}, good pixels {good.sum()}/{good.size}")

    tP = np.array([c[1] for c in combos]); tA = np.array([c[2] for c in combos])
    res = dict(region=list(shape), n_combos=len(combos), n_good=int(good.sum()),
               mask=mrep, crop=crop, splits={}, combos=[list(c) for c in combos])

    data = {}
    for b in core.BINS:
        P, V = core.attenuation_and_noise(T, H, air_T, air_H, b)
        data[b] = (P, V)
    splits = make_splits(combos, data['total'][0][:, good].mean(1))
    res['split_defs'] = {k: [v[0].tolist(), v[1].tolist()] for k, v in splits.items()}

    store = {}
    for deg in (1, 2, 3):
        X = core.design(tP, tA, deg)
        for sname, (tr, te) in splits.items():
            for b in core.BINS:
                P, V = data[b]
                out, r, s = run_split(P, V, X, tr, te, good)
                res['splits'].setdefault(sname, {}).setdefault(f'deg{deg}', {})[b] = out
                if deg == 2 and sname == 'random_seed0':
                    store[b] = (r, s, te)
        print(f"  deg{deg} done")

    np.savez_compressed(f'{outdir}/resid_deg2_seed0.npz',
                        **{f'r_{b}': store[b][0] for b in core.BINS},
                        **{f's_{b}': store[b][1] for b in core.BINS},
                        test=store['total'][2], good=good, shape=np.array(shape),
                        tP=tP, tA=tA)

    # D4 cross-bin: Total vs High residual, same pixel (Total vs Low is trivial)
    rT, rH, rL = store['total'][0], store['high'][0], store['low'][0]
    d4 = []
    for k in range(rT.shape[0]):
        d4.append(dict(combo=combos[store['total'][2][k]][0],
                       r_TH=float(np.corrcoef(rT[k], rH[k])[0, 1]),
                       r_TL=float(np.corrcoef(rT[k], rL[k])[0, 1])))
    res['D4'] = d4

    # D5 residual vs thickness, over held-out combos
    te = store['total'][2]
    res['D5'] = {b: [dict(tP=float(tP[c]), tA=float(tA[c]),
                          mean=float(store[b][0][k].mean()),
                          rms=float(np.sqrt((store[b][0][k].astype(np.float64)**2).mean())))
                     for k, c in enumerate(te)] for b in core.BINS}

    json.dump(res, open(f'{outdir}/results.json', 'w'), indent=1)
    return res


if __name__ == '__main__':
    main(crop='--full' not in sys.argv,
         outdir='test0_results' if '--full' not in sys.argv else 'test0_results_full')
