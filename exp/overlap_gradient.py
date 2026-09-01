"""Do overlapping bed positions disagree MORE at sharp edges than in flat regions?

Claim under test: object-dependent detector error should make two bed positions
that image the same voxel through different ray paths disagree, and disagree more
where the object has sharp structure.

The naive version of this test cannot work. Two failure modes both manufacture a
rising trend with zero underlying physics:
  (1) sub-voxel misregistration:  v_a - v_b ~ grad(v).delta, i.e. disagreement is
      EXACTLY proportional to gradient;
  (2) noise: FDK noise and streaks are elevated near sharp edges, and if the
      gradient is estimated from the same noisy data the correlation is spurious.

So every pair is measured twice, on the SAME voxels against the SAME gradient:

  TEST    |A_c - A_c+1|   two independent half-scans, DIFFERENT bed positions
  CONTROL |A_c - B_c|     two independent half-scans, SAME bed position

Both are differences of two statistically independent 180-view reconstructions,
so noise is matched by construction. The control shares the geometry, the cone
angle, the registration (identically zero) and the streak structure -- it can
contain everything except genuine cross-position disagreement. Only the GAP
between the two curves is evidence. A test curve that rises in parallel with the
control is noise; a test curve that rises ABOVE it is the effect.

Sub-voxel z-registration is fitted and removed per pair before differencing, and
reported, so a residual rigid shift cannot masquerade as the effect.
"""
import os, sys, json, time
import numpy as np
from scipy.ndimage import gaussian_filter, shift as ndshift

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'walnut'))
import scipy.io as sio
from prepare import prepare
from fdk import fdk

B, STEP = 2, 4
CAL = 'data/CalibrationTable'
COUCH = {1: 480.0, 2: 495.0, 3: 510.0, 4: 525.0}
NXY, SXY = 256, 50.0
# choose dz so the 15 mm couch step is an EXACT integer number of voxels (77),
# otherwise every pair needs resampling just to align and that blurs the test
NZ = 123
DZ = 15.0 / 77.0
SZ = NZ * DZ                     # 23.96 mm slab
OFF_VOX = 77                     # 15 mm in voxels
DSD, DSO = 324.335, 140.24411
V0, DV, NV_DET = 253.99112, 0.1, 505
OUT = 'exp'


def halves(c):
    """Two independent half-scan reconstructions of bed position c."""
    fa, fb = f'{OUT}/A_c{c}.npy', f'{OUT}/B_c{c}.npy'
    if os.path.exists(fa) and os.path.exists(fb):
        return np.load(fa), np.load(fb)
    root = f'data/Walnut_1/couch_{c}'
    m = sio.loadmat(f'{root}/Total/AcqPara.mat', squeeze_me=True,
                    struct_as_record=False)['AcqPara']
    P = prepare(root, CAL, 'Total', nviews=1440, step=STEP, B=B, out=None)
    ang = np.asarray(m.objViewAngle, float)[::STEP] - np.pi
    geo = dict(DSD=float(m.SDD), DSO=float(m.SID), du=0.1 * B, dv=0.1 * B,
               U0=(float(m.U0) - (B - 1) / 2) / B, V0=(float(m.V0) - (B - 1) / 2) / B,
               roll=np.deg2rad(float(m.InpRot)))
    out = []
    for sl, f in ((slice(0, None, 2), fa), (slice(1, None, 2), fb)):
        t = time.time()
        v = fdk(P[sl], ang[sl], geo, (NXY, NXY, NZ), (SXY, SXY, SZ),
                offOrigin=(0, 0, 0), verbose=False)
        print(f'  couch_{c} {os.path.basename(f)} {v.shape} in {time.time()-t:.0f}s', flush=True)
        np.save(f, v)
        out.append(v)
    del P
    return out[0], out[1]


def validity():
    """Voxels this bed position actually measures (inside detector for EVERY view)."""
    x = (np.arange(NXY) - (NXY - 1) / 2) * (SXY / NXY)
    z = (np.arange(NZ) - (NZ - 1) / 2) * DZ
    X, Y = np.meshgrid(x, x, indexing='ij')
    a_min = DSO - np.sqrt(X ** 2 + Y ** 2)
    vpos, vneg = (NV_DET - 1 - V0) * DV, V0 * DV
    ok = np.zeros((NXY, NXY, NZ), bool)
    for k, zz in enumerate(z):
        vmm = DSD * zz / a_min
        ok[:, :, k] = (a_min > 1.0) & (vmm <= vpos) & (vmm >= -vneg)
    return ok


def fit_zshift(a, b, m):
    """Residual sub-voxel z shift between two aligned slabs (minimises |a-b|)."""
    best, bs = 0.0, np.inf
    for s in np.arange(-1.0, 1.01, 0.1):
        bb = ndshift(b, (0, 0, s), order=1, mode='nearest')
        r = float(np.abs(a[m] - bb[m]).mean())
        if r < bs:
            bs, best = r, float(s)
    return best, bs


def main():
    os.makedirs(OUT, exist_ok=True)
    ok = validity()
    A, Bh = {}, {}
    for c in COUCH:
        A[c], Bh[c] = halves(c)

    res, rows = {}, []
    for c in (1, 2, 3):
        # overlap: voxel k of couch c+1 is voxel k+OFF_VOX of couch c
        lo, hi = OFF_VOX, NZ                      # index range in couch c
        a1 = A[c][:, :, lo:hi]; b1 = Bh[c][:, :, lo:hi]; o1 = ok[:, :, lo:hi]
        a2 = A[c + 1][:, :, 0:hi - lo]; b2 = Bh[c + 1][:, :, 0:hi - lo]
        o2 = ok[:, :, 0:hi - lo]
        m = o1 & o2
        print(f'pair {c}->{c+1}: overlap voxels {m.sum()} '
              f'({100*m.sum()/m.size:.1f}% of the {hi-lo}-slice band)')

        # remove any residual rigid z shift before differencing
        sft, _ = fit_zshift(a1, a2, m)
        if abs(sft) > 1e-6:
            a2 = ndshift(a2, (0, 0, sft), order=1, mode='nearest')
            b2 = ndshift(b2, (0, 0, sft), order=1, mode='nearest')
        print(f'  residual z-registration {sft*DZ*1000:+.0f} um ({sft:+.2f} voxel)')

        # gradient from the average of ALL FOUR half-scans, smoothed:
        # identical for TEST and CONTROL, so any noise-gradient coupling is shared
        ref = gaussian_filter((a1 + b1 + a2 + b2) / 4.0, 1.0)
        g = np.sqrt(sum(x ** 2 for x in np.gradient(ref, DZ)))

        d_cross = np.abs(a1 - a2)          # different bed positions
        d_within = np.abs(a1 - b1)         # same bed position  (control)

        G = g[m]; DC = d_cross[m]; DW = d_within[m]
        qs = np.quantile(G, np.linspace(0, 1, 13))
        qs[-1] += 1e-9
        binned = []
        for i in range(12):
            s = (G >= qs[i]) & (G < qs[i + 1])
            if s.sum() < 50:
                continue
            binned.append(dict(g=float(np.median(G[s])), n=int(s.sum()),
                               cross=float(np.median(DC[s])),
                               within=float(np.median(DW[s])),
                               ratio=float(np.median(DC[s]) / max(np.median(DW[s]), 1e-12))))
        res[f'{c}->{c+1}'] = dict(z_shift_um=float(sft * DZ * 1000),
                                  n_overlap=int(m.sum()), bins=binned)
        rows.append((c, binned))
        lo_b, hi_b = binned[0], binned[-1]
        print(f'  flat  bin: cross {lo_b["cross"]:.5f}  within {lo_b["within"]:.5f}  '
              f'ratio {lo_b["ratio"]:.3f}')
        print(f'  sharp bin: cross {hi_b["cross"]:.5f}  within {hi_b["within"]:.5f}  '
              f'ratio {hi_b["ratio"]:.3f}')
        print(f'  ratio change flat->sharp: {hi_b["ratio"]-lo_b["ratio"]:+.3f}')

    json.dump(res, open(f'{OUT}/overlap_gradient.json', 'w'), indent=1)
    print(f'wrote {OUT}/overlap_gradient.json')
    return res


if __name__ == '__main__':
    main()
