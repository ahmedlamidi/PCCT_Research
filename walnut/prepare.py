"""Python port of ProjDataPrepare.m (Zhou et al., Sci Data 12, 1955).

Pipeline, in their order:
  1. load Total/High, derive Low = Total - High
  2. air correction      P = -ln(I) + ln(I_air)          per bin
  3. STEPC non-uniformity  err = c1*Pl + c2*Ph + c3*Pl^2 + c4*Ph^2 + c5*Pl*Ph
                           P  = P_bin - err              (poly_corr_proj, poly_type=2)
  4. bad pixel correction  table + counts<=5 / >=4090, linear interp along channel;
                           detector gap slices 252:254 (1-based) interp along slice
  5. ring artifact correction (RingArtifactCorrection.m, Method_type=1)

Deviation: ring correction is applied after detector binning with proportionally
scaled kernel/window, rather than at full resolution. Everything else is faithful.
"""
import os, sys, time, numpy as np
from scipy.ndimage import median_filter, gaussian_filter

NR, NC = 505, 2063           # slices (v), channels (u)
LOW_T, HIGH_T = 5, 4090
GAP_SLICES = [251, 252, 253]  # 0-based; MATLAB gap_index = [252;253;254]


def load(path):
    return np.fromfile(path, '<u2').reshape(NR, NC)


def build_mask(root, nviews, step, cal):
    """Union of the shipped bad-channel table and any pixel out of range in any view."""
    bad = np.zeros((NR, NC), bool)
    idx = np.round(np.fromfile(f'{cal}/badchannelIndexAll.data', '<f4')).astype(np.int64) - 1
    idx = idx[(idx >= 0) & (idx < NR * NC)]
    bad.ravel()[idx] = True
    n_tab = int(bad.sum())
    for i in range(1, nviews + 1, step):
        T = load(f'{root}/Total/proj_{i:05d}.raw').astype(np.int32)
        H = load(f'{root}/High/proj_{i:05d}.raw').astype(np.int32)
        L = T - H
        for A in (T, H, L):
            bad |= (A <= LOW_T) | (A >= HIGH_T)
    print(f'  bad pixels: {bad.sum()} ({n_tab} from table, {bad.sum()-n_tab} added by thresholds)')
    return bad


def interp_plan(bad):
    """Precompute linear interpolation along the channel axis for a fixed mask."""
    plan = []
    for v in range(NR):
        b = np.where(bad[v])[0]
        g = np.where(~bad[v])[0]
        if b.size == 0 or g.size < 2:
            plan.append(None); continue
        j = np.searchsorted(g, b)
        lo = g[np.clip(j - 1, 0, g.size - 1)]
        hi = g[np.clip(j, 0, g.size - 1)]
        same = lo == hi
        w = np.where(same, 1.0, (hi - b) / np.where(same, 1, hi - lo))
        plan.append((b, lo, hi, w.astype(np.float32)))
    return plan


def apply_interp(P, plan):
    for v, pl in enumerate(plan):
        if pl is None: continue
        b, lo, hi, w = pl
        P[v, b] = w * P[v, lo] + (1 - w) * P[v, hi]
    good = [s for s in range(NR) if s not in GAP_SLICES]
    for s in GAP_SLICES:
        lo = max([g for g in good if g < s], default=good[0])
        hi = min([g for g in good if g > s], default=good[-1])
        f = 0.5 if hi == lo else (s - lo) / (hi - lo)
        P[s] = (1 - f) * P[lo] + f * P[hi]
    return P


def bin2(A, B):
    r, c = (A.shape[0] // B) * B, (A.shape[1] // B) * B
    return A[:r, :c].reshape(r // B, B, c // B, B).mean((1, 3))


def ring_correct(P, U0b, B, nsub=10):
    """RingArtifactCorrection.m Method_type=1, at binned resolution."""
    nv = P.shape[0]
    sub = nv // nsub
    w = max(2, int(round(10 / B)))
    coeff = np.empty((nsub,) + P.shape[1:], np.float32)
    for k in range(nsub):
        m = P[k * sub:(k + 1) * sub].mean(0)
        coeff[k] = m - median_filter(m, size=(w, w), mode='nearest')
    c = np.median(coeff, 0)
    c = c - gaussian_filter(c, 2.0 / B)
    half = int(round(250 / B))
    keep = np.zeros_like(c)
    a, b = max(0, int(U0b) - half), min(c.shape[1], int(U0b) + half)
    keep[:, a:b] = c[:, a:b]
    return P - keep


def prepare(root, cal, energy='Total', nviews=1440, step=1, B=2, out=None):
    t0 = time.time()
    bad = build_mask(root, nviews, max(step, 1), cal)
    plan = interp_plan(bad)

    air = {k: load(f'{cal}/air_table_{k}.raw').astype(np.float32)
           for k in ('total', 'high', 'low')}
    lnair = {k: np.log(np.maximum(v, 1.0)) for k, v in air.items()}
    # MATLAB: reshape(tbl, nChannel, nSlice, []) column-major puts element (u,v,k)
    # at flat offset u + NC*v + NC*NR*k, which is exactly C-order (5, NR, NC)[k,v,u].
    tab = np.fromfile(f'{cal}/STEPC_table_{energy.lower()}.data', '<f4').reshape(5, NR, NC)

    idx = list(range(1, nviews + 1, step))
    nb_r, nb_c = NR // B, NC // B
    outP = np.empty((len(idx), nb_r, nb_c), np.float32)
    for n, i in enumerate(idx):
        T = load(f'{root}/Total/proj_{i:05d}.raw').astype(np.float32)
        H = load(f'{root}/High/proj_{i:05d}.raw').astype(np.float32)
        L = T - H
        Pt = -np.log(np.maximum(T, 1.0)) + lnair['total']
        Ph = -np.log(np.maximum(H, 1.0)) + lnair['high']
        Pl = -np.log(np.maximum(L, 1.0)) + lnair['low']
        # STEPC is itself an OBJECT-DEPENDENT correction: a per-pixel polynomial in
        # the measured attenuation of both bins, vendor-fitted on the PMMA/Al slabs to
        # cancel how detector response varies with the object. Set NO_STEPC=1 to skip
        # it and ask whether it is removing the very effect we are hunting.
        if os.environ.get('NO_STEPC'):
            err = np.zeros_like(Pt)
        else:
            err = (tab[0]*Pl + tab[1]*Ph + tab[2]*Pl**2 + tab[3]*Ph**2 + tab[4]*Pl*Ph)
        P = {'Total': Pt, 'High': Ph, 'Low': Pl}[energy] - err
        outP[n] = bin2(apply_interp(P, plan), B)
        if n % 200 == 0:
            print(f'  {n}/{len(idx)}  {time.time()-t0:.0f}s', flush=True)
    U0b = (983.5178 - (B - 1) / 2) / B
    outP = ring_correct(outP, U0b, B)
    if out:
        np.save(out, outP)
    print(f'  prepared {outP.shape} in {time.time()-t0:.0f}s')
    return outP


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/Walnut_1/couch_1')
    ap.add_argument('--cal', default='data/CalibrationTable')
    ap.add_argument('--energy', default='Total')
    ap.add_argument('--nviews', type=int, default=1440)
    ap.add_argument('--step', type=int, default=2)
    ap.add_argument('--bin', type=int, default=2)
    ap.add_argument('--out', default='walnut/P_total.npy')
    a = ap.parse_args()
    prepare(a.root, a.cal, a.energy, a.nviews, a.step, a.bin, a.out)
