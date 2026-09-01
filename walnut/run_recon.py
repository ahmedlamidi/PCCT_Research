"""Drive the full Walnut_1 couch_1 reconstruction: prepare -> FDK -> figures."""
import sys, time, numpy as np
sys.path.insert(0, 'walnut')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from prepare import prepare
from fdk import fdk
import scipy.io as sio

B, STEP = 2, 2
ROOT, CAL = 'data/Walnut_1/couch_1', 'data/CalibrationTable'

def main(energy='Total', nvox=(256,256,96), svox=(50,50,18), zoff=5.0):
    m = sio.loadmat(f'{ROOT}/Total/AcqPara.mat', squeeze_me=True, struct_as_record=False)['AcqPara']
    ang_all = np.asarray(m.objViewAngle, float)
    P = prepare(ROOT, CAL, energy, nviews=1440, step=STEP, B=B,
                out=f'walnut/P_{energy.lower()}.npy')
    angles = ang_all[::STEP] - np.pi          # their convention
    geo = dict(DSD=float(m.SDD), DSO=float(m.SID), du=0.1*B, dv=0.1*B,
               U0=(float(m.U0)-(B-1)/2)/B, V0=(float(m.V0)-(B-1)/2)/B,
               roll=np.deg2rad(float(m.InpRot)))
    print(f'  geo {geo}', flush=True)
    t = time.time()
    vol = fdk(P, angles, geo, nvox, svox, offOrigin=(0, 0, zoff), window='hann')
    print(f'  FDK {vol.shape} in {time.time()-t:.0f}s', flush=True)
    np.save(f'walnut/vol_{energy.lower()}.npy', vol)
    return vol

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'Total')
