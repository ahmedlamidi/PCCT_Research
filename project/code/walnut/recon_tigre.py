"""CUDA reconstruction of the Walnut PCCT dataset via TIGRE — literal port of
ProjDataRecon.m / WalnutDataRecon.m / ReconAllEnergy.m (Zhou et al., Sci Data 12, 1955).

Preprocessing is shared with the CPU path (prepare.py, the port of ProjDataPrepare.m).
This module only replaces the reconstruction call, which in their MATLAB is:

    rawdata_proj = permute(rawdata_proj, [2,1,3]);
    geo.DSD = AcqPara.SDD;  geo.DSO = AcqPara.SID;
    geo.nDetector = [nChannelNum; nSliceNum];      % [U; V] in MATLAB TIGRE
    geo.dDetector = [fDetU; fDetV];
    geo.nVoxel    = [1000;1000;300];               % [x; y; z] in MATLAB TIGRE
    geo.sVoxel    = [50;50;15];
    geo.offDetector = [(nChannelNum/2-U0)/10; (nSliceNum/2-V0)/10];
    geo.rotDetector = [deg2rad(InpRot); 0; 0];
    angles = unwrap(AcqPara.objViewAngle' - pi);
    ReconData = FDK(rawdata_proj, geo, angles, 'filter', 'hann');

!! AXIS ORDER — the one real hazard in this port !!
MATLAB TIGRE and Python TIGRE use OPPOSITE orderings:
    MATLAB : geo.nDetector = [nU; nV]      proj shape (nV, nU, nAngles)
             geo.nVoxel    = [nx; ny; nz]
    Python : geo.nDetector = [nV, nU]      proj shape (nAngles, nV, nU)
             geo.nVoxel    = [nz, ny, nx]
So nDetector/dDetector/offDetector are reversed relative to their MATLAB, and
nVoxel/sVoxel/dVoxel are z-first. Everything below is written in PYTHON order.

Their offDetector divides pixel counts by 10, which is only correct because
fDetU == fDetV == 0.1 mm. We multiply by the actual pixel size instead — identical
here, but it does not silently break if the detector is binned.

Requires: pip install tigre  (needs CUDA toolkit + NVIDIA GPU, >=8 GB per their README)
Untested on this machine — there is no GPU here. Verify the first reconstruction
against walnut/figs/*_ortho.png from the CPU path before trusting it.
"""
import os, sys, time, argparse
import numpy as np
import scipy.io as sio
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import prepare

CAL = 'data/CalibrationTable'
COUCH = (1, 2, 3, 4)


def build_geo(m, nVoxel, sVoxel, B=1):
    """Map AcqPara -> TIGRE geometry, in PYTHON axis order."""
    import tigre
    nU, nV = int(m.nChannelNum) // B, int(m.nSliceNum) // B
    dU, dV = float(m.fDetU) * B, float(m.fDetV) * B
    # piercing point in (possibly binned) pixel units
    U0 = (float(m.U0) - (B - 1) / 2) / B
    V0 = (float(m.V0) - (B - 1) / 2) / B

    geo = tigre.geometry(mode='cone')
    geo.DSD = float(m.SDD)
    geo.DSO = float(m.SID)
    geo.nDetector = np.array([nV, nU])                 # python order [V, U]
    geo.dDetector = np.array([dV, dU])
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.nVoxel = np.array(nVoxel)                      # python order [z, y, x]
    geo.sVoxel = np.array(sVoxel, dtype=float)
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    geo.offOrigin = np.array([0.0, 0.0, 0.0])
    # their [(nU/2 - U0)/10 ; (nV/2 - V0)/10], reversed to [V, U] and using real pixel size
    geo.offDetector = np.array([(nV / 2 - V0) * dV, (nU / 2 - U0) * dU])
    geo.rotDetector = np.array([np.deg2rad(float(m.InpRot)), 0.0, 0.0])
    geo.COR = 0.0
    geo.accuracy = 1.0
    return geo


def hu_scale(energy):
    with h5py.File(f'{CAL}/HU_water_table.mat', 'r') as f:
        return float(f['HU_water_table'][energy.lower()][()].ravel()[0])


def recon_couch(c, energy='Total', step=1, B=1, nVoxel=(300, 1000, 1000),
                sVoxel=(15.0, 50.0, 50.0), filt='hann', tv_iter=0, tv_lambda=20.0):
    import tigre
    from tigre.algorithms import fdk as tigre_fdk

    root = f'data/Walnut_1/couch_{c}'
    m = sio.loadmat(f'{root}/Total/AcqPara.mat', squeeze_me=True,
                    struct_as_record=False)['AcqPara']

    # --- shared preprocessing (port of ProjDataPrepare.m) ---
    P = prepare(root, CAL, energy, nviews=int(m.nViewTotal), step=step, B=B, out=None)
    P = np.ascontiguousarray(P.astype(np.float32))     # (nAngles, nV, nU) == python TIGRE

    # --- angles: unwrap(objViewAngle - pi), as in ProjDataRecon.m ---
    angles = np.unwrap(np.asarray(m.objViewAngle, float)[::step] - np.pi)

    geo = build_geo(m, nVoxel, sVoxel, B=B)
    print(f'  couch_{c}: proj {P.shape}  nDetector {geo.nDetector}  '
          f'offDetector {geo.offDetector}  nVoxel {geo.nVoxel}', flush=True)

    t = time.time()
    vol = tigre_fdk(P, geo, angles, filter=filt)
    print(f'  couch_{c}: TIGRE FDK {vol.shape} in {time.time()-t:.0f}s', flush=True)

    if tv_iter:                                        # recon_type = 2 (FDK + TV)
        from tigre.utilities.im_3d_denoise import im3ddenoise
        t = time.time()
        vol = im3ddenoise(vol, 'TV', tv_iter, tv_lambda)
        print(f'  couch_{c}: TV denoise in {time.time()-t:.0f}s', flush=True)
    return vol.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--energy', default='Total', choices=['Total', 'High', 'Low'])
    ap.add_argument('--couches', default='1,2,3,4')
    ap.add_argument('--step', type=int, default=1, help='their dose_ratio')
    ap.add_argument('--bin', type=int, default=1, help='detector binning (1 = native)')
    ap.add_argument('--nz', type=int, default=300)
    ap.add_argument('--nxy', type=int, default=1000)
    ap.add_argument('--filter', default='hann')
    ap.add_argument('--tv-iter', type=int, default=0, help='100 reproduces their recon_type=2')
    ap.add_argument('--tv-lambda', type=float, default=20.0)
    ap.add_argument('--out', default='walnut/tigre')
    a = ap.parse_args()

    try:
        import tigre  # noqa: F401
    except ImportError:
        sys.exit('TIGRE not installed. On a CUDA machine:  pip install tigre\n'
                 '(see https://github.com/CERN/TIGRE)')

    os.makedirs(a.out, exist_ok=True)
    cs = [int(x) for x in a.couches.split(',')]
    vols = []
    for c in cs:
        v = recon_couch(c, a.energy, step=a.step, B=a.bin,
                        nVoxel=(a.nz, a.nxy, a.nxy), sVoxel=(15.0, 50.0, 50.0),
                        filt=a.filter, tv_iter=a.tv_iter, tv_lambda=a.tv_lambda)
        np.save(f'{a.out}/vol_c{c}_{a.energy.lower()}.npy', v)
        vols.append(v)

    # couch positions are 15 mm apart and each slab is 15 mm, so slabs abut;
    # ReconAllEnergy.m/ReconDataSave.m write them as one DICOM series in couch order.
    # Python TIGRE volumes are z-first, so stitching concatenates axis 0.
    st = np.concatenate(vols, axis=0)
    hu = st * hu_scale(a.energy) - 1000.0              # ReconDataSave.m
    np.save(f'{a.out}/vol_stitched_mu_{a.energy.lower()}.npy', st)
    np.save(f'{a.out}/vol_stitched_hu_{a.energy.lower()}.npy', hu.astype(np.float32))
    print(f'  stitched {st.shape}; HU range '
          f'[{np.percentile(hu,0.1):.0f}, {np.percentile(hu,99.9):.0f}]')


if __name__ == '__main__':
    main()
