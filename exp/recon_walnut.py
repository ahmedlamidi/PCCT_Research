"""Reconstruct the two independent half-scans per bed position for any walnut."""
import os, sys, time
import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'walnut'))
from prepare import prepare
from fdk import fdk

B, STEP = 2, 4
CAL = 'data/CalibrationTable'
NXY, SXY, NZ = 256, 50.0, 123
DZ = 15.0 / 77.0
SZ = NZ * DZ


def main(w):
    out = 'exp' if str(w) == '1' else f'exp/w{w}'
    os.makedirs(out, exist_ok=True)
    for c in (1, 2, 3, 4):
        fa, fb = f'{out}/A_c{c}.npy', f'{out}/B_c{c}.npy'
        if os.path.exists(fa) and os.path.exists(fb):
            print(f'  walnut{w} couch_{c} cached'); continue
        root = f'data/Walnut_{w}/couch_{c}'
        m = sio.loadmat(f'{root}/Total/AcqPara.mat', squeeze_me=True,
                        struct_as_record=False)['AcqPara']
        P = prepare(root, CAL, 'Total', nviews=1440, step=STEP, B=B, out=None)
        ang = np.asarray(m.objViewAngle, float)[::STEP] - np.pi
        geo = dict(DSD=float(m.SDD), DSO=float(m.SID), du=0.1 * B, dv=0.1 * B,
                   U0=(float(m.U0) - (B - 1) / 2) / B,
                   V0=(float(m.V0) - (B - 1) / 2) / B,
                   roll=np.deg2rad(float(m.InpRot)))
        for sl, f in ((slice(0, None, 2), fa), (slice(1, None, 2), fb)):
            t = time.time()
            v = fdk(P[sl], ang[sl], geo, (NXY, NXY, NZ), (SXY, SXY, SZ),
                    offOrigin=(0, 0, 0), verbose=False)
            np.save(f, v)
            print(f'  walnut{w} couch_{c} {os.path.basename(f)} in {time.time()-t:.0f}s',
                  flush=True)
        del P
    print(f'WALNUT_{w}_RECON_COMPLETE', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
