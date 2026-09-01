"""Gate B -- overlap sufficiency between adjacent bed positions.

Important subtlety: their WalnutDataRecon.m sets sVoxel(3) = 15 mm, exactly the
couch step, so reconstructing "as they do" TRUNCATES every slab to abutting
blocks and yields 0% overlap by construction. That is a reconstruction choice,
not a property of the acquisition. The physically available overlap is set by the
detector's axial FOV at isocentre versus the couch step:

    axial FOV at iso = 505 px * 0.1 mm / (SDD/SID) = 50.5 / 2.3127 = 21.84 mm
    couch step       = 15 mm
    -> adjacent positions genuinely share ~6.8 mm at the rotation axis

So each couch is reconstructed here over a WIDER 24 mm slab, and a per-voxel
validity mask is computed from the cone geometry: a voxel counts as covered by a
bed position only if it projects inside the detector for EVERY view.

For a voxel at radius r and height z (relative to that position's centre plane),
the binding view is the one where the source is closest, a_min = DSO - r, so

    covered  <=>  DSD*z/(DSO - r)  within  [-(V0*dv), ((nV-1-V0)*dv)]
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'walnut'))
import scipy.io as sio
from prepare import prepare
from fdk import fdk

B, STEP = 2, 4
CAL = 'data/CalibrationTable'
COUCH = {1: 480.0, 2: 495.0, 3: 510.0, 4: 525.0}
NXY, SXY = 256, 50.0
NZ, SZ = 123, 24.0          # 24 mm slab, ~0.195 mm voxels (wider than the 21.84 mm FOV)
DSD, DSO = 324.335, 140.24411
V0, DV, NV_DET = 253.99112, 0.1, 505


def recon_wide(c):
    out = f'gateB/vol_wide_c{c}.npy'
    if os.path.exists(out):
        return np.load(out)
    root = f'data/Walnut_1/couch_{c}'
    m = sio.loadmat(f'{root}/Total/AcqPara.mat', squeeze_me=True,
                    struct_as_record=False)['AcqPara']
    P = prepare(root, CAL, 'Total', nviews=1440, step=STEP, B=B, out=None)
    angles = np.asarray(m.objViewAngle, float)[::STEP] - np.pi
    geo = dict(DSD=float(m.SDD), DSO=float(m.SID), du=0.1 * B, dv=0.1 * B,
               U0=(float(m.U0) - (B - 1) / 2) / B, V0=(float(m.V0) - (B - 1) / 2) / B,
               roll=np.deg2rad(float(m.InpRot)))
    v = fdk(P, angles, geo, (NXY, NXY, NZ), (SXY, SXY, SZ), offOrigin=(0, 0, 0),
            verbose=False)
    del P
    os.makedirs('gateB', exist_ok=True)
    np.save(out, v)
    return v


def validity_mask():
    """Per-voxel coverage for ONE bed position, on the local 24 mm grid."""
    x = (np.arange(NXY) - (NXY - 1) / 2) * (SXY / NXY)
    z = (np.arange(NZ) - (NZ - 1) / 2) * (SZ / NZ)
    X, Y = np.meshgrid(x, x, indexing='ij')
    r = np.sqrt(X**2 + Y**2)
    a_min = DSO - r                                   # closest approach of source
    vpos = (NV_DET - 1 - V0) * DV                     # +25.00 mm
    vneg = V0 * DV                                    # -25.40 mm
    ok = np.zeros((NXY, NXY, NZ), bool)
    with np.errstate(divide='ignore', invalid='ignore'):
        for k, zz in enumerate(z):
            vmm = DSD * zz / a_min
            ok[:, :, k] = (a_min > 1.0) & (vmm <= vpos) & (vmm >= -vneg)
    return ok, z


def main():
    os.makedirs('gateB', exist_ok=True)
    vols = {c: recon_wide(c) for c in COUCH}
    ok, zloc = validity_mask()

    dz = SZ / NZ
    zg_lo, zg_hi = min(COUCH.values()) - SZ / 2, max(COUCH.values()) + SZ / 2
    ng = int(round((zg_hi - zg_lo) / dz))
    zglob = zg_lo + (np.arange(ng) + 0.5) * dz

    cover = np.zeros((NXY, NXY, ng), np.uint8)
    vol_g = np.zeros((NXY, NXY, ng), np.float32)
    wsum = np.zeros(ng)
    for c, cz in COUCH.items():
        off = int(round((cz - SZ / 2 - zg_lo) / dz))
        cover[:, :, off:off + NZ] += ok
        vol_g[:, :, off:off + NZ] += vols[c] * ok
        wsum[off:off + NZ] += 1
    with np.errstate(invalid='ignore', divide='ignore'):
        vol_g /= np.maximum(cover, 1)

    covered = cover >= 1
    double = cover >= 2
    frac = double.sum() / max(covered.sum(), 1)
    print(f'geometric double-coverage: {100*frac:.1f}% of covered volume')

    # z-distribution
    per_z_cov = covered.sum((0, 1)).astype(float)
    per_z_dbl = double.sum((0, 1)).astype(float)
    with np.errstate(invalid='ignore'):
        zfrac = np.where(per_z_cov > 0, per_z_dbl / np.maximum(per_z_cov, 1), 0)
    bands = []
    inb = False
    for i, f in enumerate(zfrac):
        if f > 0.05 and not inb:
            inb = True; s = i
        elif f <= 0.05 and inb:
            inb = False; bands.append((zglob[s], zglob[i - 1]))
    if inb: bands.append((zglob[s], zglob[-1]))
    print(f'double-covered z bands (mm, couch coords): '
          f'{[(round(a,1),round(b,1)) for a,b in bands]}')

    # tissue composition, from the HU-calibrated stitched volume
    import h5py
    with h5py.File(f'{CAL}/HU_water_table.mat', 'r') as f:
        s = float(f['HU_water_table']['total'][()].ravel()[0])
    hu = vol_g * s - 1000.0
    kernel = (hu > -400) & (hu < 150) & covered
    shell = (hu >= 150) & covered
    res = dict(
        double_cov_fraction=float(frac),
        covered_voxels=int(covered.sum()), double_voxels=int(double.sum()),
        z_bands_mm=[[float(a), float(b)] for a, b in bands],
        n_bands=len(bands),
        axial_fov_iso_mm=float(NV_DET * DV / (DSD / DSO)),
        couch_step_mm=15.0,
        kernel_voxels=int(kernel.sum()), shell_voxels=int(shell.sum()),
        kernel_double=int((kernel & double).sum()), shell_double=int((shell & double).sum()),
        kernel_double_frac=float((kernel & double).sum() / max(kernel.sum(), 1)),
        shell_double_frac=float((shell & double).sum() / max(shell.sum(), 1)),
    )
    print(f"tissue in double-covered regions: kernel {100*res['kernel_double_frac']:.1f}% "
          f"of kernel voxels, shell {100*res['shell_double_frac']:.1f}% of shell voxels")

    # registration check: cross-correlate adjacent slabs in their shared band
    shifts = []
    for c in (1, 2, 3):
        v1, v2 = vols[c], vols[c + 1]
        p1 = v1.mean((0, 1)); p2 = v2.mean((0, 1))
        p1 = p1 - p1.mean(); p2 = p2 - p2.mean()
        best, bs = 0, -np.inf
        for sft in range(-NZ + 10, NZ - 10):
            a0, a1 = max(0, sft), min(NZ, NZ + sft)
            b0, b1 = max(0, -sft), min(NZ, NZ - sft)
            if a1 - a0 < 25: continue
            x, y = p1[a0:a1], p2[b0:b1]
            d = np.linalg.norm(x) * np.linalg.norm(y)
            if d <= 0: continue
            r = float(x @ y / d)
            if r > bs: bs, best = r, sft
        shifts.append(dict(pair=f'{c}->{c+1}', shift_vox=best,
                           shift_mm=float(best * dz), corr=bs))
        print(f'  registration couch_{c}->{c+1}: {best*dz:+.2f} mm (corr {bs:.3f}), '
              f'nominal +15.00 mm')
    res['registration'] = shifts
    np.save('gateB/coverage_count.npy', cover)
    np.save('gateB/vol_global_hu.npy', hu.astype(np.float32))
    json.dump(res, open('gateB/results.json', 'w'), indent=1)
    print('wrote gateB/results.json')


if __name__ == '__main__':
    main()
