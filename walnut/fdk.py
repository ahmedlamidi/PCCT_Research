"""CPU cone-beam FDK matching the TIGRE geometry used in ProjDataRecon.m.

TIGRE geo built by their code:
    DSD = SDD, DSO = SID
    nDetector = [nChannelNum; nSliceNum], dDetector = [fDetU; fDetV]
    offDetector = [(nChannelNum/2 - U0); (nSliceNum/2 - V0)] * 0.1 mm
      -> the piercing point sits at detector index (U0, V0)
    rotDetector = [deg2rad(InpRot); 0; 0]      (InpRot is in DEGREES)
    angles = objViewAngle - pi, unwrapped

No GPU here, so this is numpy + scipy bilinear interpolation.
"""
import numpy as np
from scipy.ndimage import map_coordinates


def ramp_kernel(n, d, window='hann'):
    """Ram-Lak kernel sampled at spacing d, windowed in frequency."""
    k = np.arange(-n, n + 1)
    h = np.zeros(k.size)
    h[k == 0] = 1.0 / (4 * d * d)
    odd = k % 2 != 0
    h[odd] = -1.0 / (np.pi * k[odd] * d) ** 2
    H = np.fft.rfft(np.fft.ifftshift(h))
    f = np.fft.rfftfreq(h.size)
    if window == 'hann':
        H *= 0.5 * (1 + np.cos(2 * np.pi * f))
    elif window == 'hamming':
        H *= 0.54 + 0.46 * np.cos(2 * np.pi * f)
    elif window == 'cosine':
        H *= np.cos(np.pi * f)
    return H, h.size


def fdk(P, angles, geo, nVoxel, sVoxel, offOrigin=(0, 0, 0), window='hann',
        zchunk=16, verbose=True):
    """P: (nviews, nv, nu) already air-corrected/STEPC/ring-corrected line integrals."""
    DSD, DSO = geo['DSD'], geo['DSO']
    du, dv = geo['du'], geo['dv']
    U0, V0 = geo['U0'], geo['V0']
    roll = geo.get('roll', 0.0)
    nviews, nv, nu = P.shape

    # detector coordinates in mm relative to the piercing point
    u = (np.arange(nu) - U0) * du
    v = (np.arange(nv) - V0) * dv
    UU, VV = np.meshgrid(u, v)

    # FDK cosine weight, then ramp filter along the channel axis
    W = DSD / np.sqrt(DSD**2 + UU**2 + VV**2)
    H, npad = ramp_kernel(nu, du, window)
    G = np.empty_like(P, dtype=np.float32)
    for i in range(nviews):
        pw = P[i] * W
        F = np.fft.rfft(pw, n=npad, axis=1) * H
        G[i] = (np.fft.irfft(F, n=npad, axis=1)[:, :nu] * du).astype(np.float32)
    if verbose: print('  filtered', G.shape, flush=True)

    nx, ny, nz = nVoxel
    sx, sy, sz = sVoxel
    ox, oy, oz = offOrigin
    x = (np.arange(nx) - (nx - 1) / 2) * (sx / nx) + ox
    y = (np.arange(ny) - (ny - 1) / 2) * (sy / ny) + oy
    z = (np.arange(nz) - (nz - 1) / 2) * (sz / nz) + oz
    X, Y = np.meshgrid(x, y, indexing='ij')

    vol = np.zeros((nx, ny, nz), np.float32)
    dbeta = 2 * np.pi / nviews
    cr, sr = np.cos(roll), np.sin(roll)
    for i, b in enumerate(angles):
        cb, sb = np.cos(b), np.sin(b)
        a = DSO - (X * cb + Y * sb)          # source->voxel distance along the axis
        np.maximum(a, 1e-6, out=a)
        s = -X * sb + Y * cb
        umm = DSD * s / a
        iu_base = umm
        w2 = (DSO / a) ** 2
        for z0 in range(0, nz, zchunk):
            zz = z[z0:z0 + zchunk]
            vmm = DSD * (zz[None, None, :] / a[:, :, None])
            uu = iu_base[:, :, None] + np.zeros_like(vmm)
            if roll:
                uu, vmm = uu * cr + vmm * sr, -uu * sr + vmm * cr
            coords = np.stack([(vmm / dv + V0).ravel(), (uu / du + U0).ravel()])
            samp = map_coordinates(G[i], coords, order=1, mode='constant', cval=0.0)
            vol[:, :, z0:z0 + zchunk] += (samp.reshape(vmm.shape) *
                                          w2[:, :, None]).astype(np.float32)
        if verbose and i % 100 == 0:
            print(f'  backproject {i}/{nviews}', flush=True)
    # The ramp was applied in detector-plane coordinates (spacing du at DSD) but the
    # backprojection weight (DSO/a)^2 assumes the virtual detector at the isocentre.
    # Since the ramp kernel obeys h(lambda x) = h(x)/lambda^2, that costs a factor
    # DSO/DSD; undo it here.
    vol *= (dbeta / 2.0) * (DSD / DSO)        # 360 deg scan -> 2x redundancy
    return vol
