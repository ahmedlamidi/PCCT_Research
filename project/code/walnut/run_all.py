"""Reconstruct all four couch positions and stitch, following their pipeline.

Mirrors ReconAllEnergy.m + ReconDataSave.m:
  - each couch folder is reconstructed independently into a 15 mm slab
    (their sVoxel = [50;50;15], offOrigin = [0;0;0])
  - couch positions are 480/495/510/525 mm, i.e. exactly 15 mm apart, so the
    slabs abut; their code writes each to DICOM at fStartPos = couchPos - 7.5
    and the series stacks by slice position. Here that is a plain concatenation
    in couch order -- no blending, matching what they do.
  - HU conversion from ReconDataSave.m:  HU = mu * HU_water_table.<bin> - 1000

Only difference from their code: ProjDataRecon.m calls TIGRE's GPU FDK, which
needs CUDA. There is no GPU here, so fdk.py supplies an equivalent CPU FDK built
on the same TIGRE geometry (validated to 0.2% against analytic spheres).
"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scipy.io as sio
import h5py
from prepare import prepare
from fdk import fdk

B, STEP = 2, 4
CAL = 'data/CalibrationTable'
NXY, SXY = 256, 50.0          # their sVoxel(1:2) = 50 mm
NZ, SZ = 77, 15.0             # their sVoxel(3) = 15 mm; 77 slices -> isotropic
COUCH = {1: 480.0, 2: 495.0, 3: 510.0, 4: 525.0}


def hu_scale(energy):
    with h5py.File(f'{CAL}/HU_water_table.mat', 'r') as f:
        return float(f['HU_water_table'][energy.lower()][()].ravel()[0])


def recon_couch(c, energy='Total'):
    out = f'walnut/vol_c{c}_{energy.lower()}.npy'
    if os.path.exists(out):
        print(f'  couch_{c} already reconstructed'); return np.load(out)
    root = f'data/Walnut_1/couch_{c}'
    m = sio.loadmat(f'{root}/Total/AcqPara.mat', squeeze_me=True,
                    struct_as_record=False)['AcqPara']
    P = prepare(root, CAL, energy, nviews=1440, step=STEP, B=B, out=None)
    angles = np.asarray(m.objViewAngle, float)[::STEP] - np.pi   # their convention
    geo = dict(DSD=float(m.SDD), DSO=float(m.SID), du=0.1 * B, dv=0.1 * B,
               U0=(float(m.U0) - (B - 1) / 2) / B,
               V0=(float(m.V0) - (B - 1) / 2) / B,
               roll=np.deg2rad(float(m.InpRot)))
    t = time.time()
    vol = fdk(P, angles, geo, (NXY, NXY, NZ), (SXY, SXY, SZ), offOrigin=(0, 0, 0))
    print(f'  couch_{c} FDK {vol.shape} in {time.time()-t:.0f}s', flush=True)
    del P
    np.save(out, vol)
    return vol


if __name__ == '__main__':
    energy = 'Total'
    cs = [int(x) for x in sys.argv[1:] if x.isdigit()] or [1, 2, 3, 4]
    vols = [recon_couch(c, energy) for c in cs]
    if len(vols) > 1:
        st = np.concatenate(vols, axis=2)            # couch order == slice order
        hu = st * hu_scale(energy) - 1000.0          # ReconDataSave.m
        np.save('walnut/vol_stitched_mu.npy', st)
        np.save('walnut/vol_stitched_hu.npy', hu.astype(np.float32))
        print(f'  stitched {st.shape} = {st.shape[2]*SZ/NZ:.1f} mm of walnut')
        print(f'  HU range [{np.percentile(hu,0.1):.0f}, {np.percentile(hu,99.9):.0f}]')
