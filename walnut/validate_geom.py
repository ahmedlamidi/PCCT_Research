"""Forward-project analytic spheres through the real geometry, FDK them back.

If the geometry conventions (offset piercing point, angle sign, cone weights) are
wrong, the spheres come back blurred, doubled, or displaced. This is the check
that has to pass before trusting anything on the walnut.
"""
import numpy as np, sys, time
sys.path.insert(0, 'walnut')
from fdk import fdk

B = 4
NR, NC = 505, 2063
geo = dict(DSD=324.335, DSO=140.24411, du=0.1*B, dv=0.1*B,
           U0=(983.5178-(B-1)/2)/B, V0=(253.99112-(B-1)/2)/B,
           roll=np.deg2rad(0.12198271458335226))
nu, nv = NC//B, NR//B
NVIEWS = 360
angles = np.linspace(0, 2*np.pi, NVIEWS, endpoint=False)

SPHERES = [((0.0, 0.0, 3.0), 6.0, 0.02),      # big central
           ((8.0, 0.0, 3.0), 2.0, 0.04),      # +x
           ((0.0, 8.0, 6.0), 2.0, 0.06)]      # +y, higher z

def forward(angles, sign=+1):
    u = (np.arange(nu) - geo['U0']) * geo['du']
    v = (np.arange(nv) - geo['V0']) * geo['dv']
    UU, VV = np.meshgrid(u, v)
    cr, sr = np.cos(geo['roll']), np.sin(geo['roll'])
    UUr, VVr = UU*cr - VV*sr, UU*sr + VV*cr
    P = np.zeros((len(angles), nv, nu), np.float32)
    for i, b0 in enumerate(angles):
        b = sign*b0
        cb, sb = np.cos(b), np.sin(b)
        S = np.array([geo['DSO']*cb, geo['DSO']*sb, 0.0])
        axis = np.array([cb, sb, 0.0]); eu = np.array([-sb, cb, 0.0]); ez = np.array([0,0,1.0])
        D = (S - geo['DSD']*axis)[None,None,:] + UUr[...,None]*eu + VVr[...,None]*ez
        d = D - S; d /= np.linalg.norm(d, axis=-1, keepdims=True)
        acc = np.zeros((nv, nu))
        for c, R, mu in SPHERES:
            m = S - np.array(c)
            bq = (m*d).sum(-1); cq = m@m - R*R
            disc = bq*bq - cq
            acc += np.where(disc > 0, 2*np.sqrt(np.maximum(disc, 0))*mu, 0)
        P[i] = acc
    return P

if __name__ == '__main__':
    t=time.time(); P = forward(angles); print(f'forward {P.shape} {time.time()-t:.0f}s', flush=True)
    for sign, tag in ((+1,'plus'), (-1,'minus')):
        vol = fdk(P, sign*angles, geo, (192,192,64), (48,48,16), offOrigin=(0,0,3),
                  window='hann', verbose=False)
        np.save(f'walnut/val_{tag}.npy', vol)
        c = vol[:,:,32]
        print(f'sign {tag:5s}: vol range [{vol.min():+.4f},{vol.max():+.4f}]  '
              f'centre-slice peak {c.max():.4f}  mean|v| {np.abs(vol).mean():.5f}')
