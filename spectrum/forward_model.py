"""Serialized two-bin forward model. This is what downstream experiments import.

    fm = ForwardModel.load('spectrum/forward_model.npz')
    lamL, lamH = fm.predict({'Water': 30.0, 'HA': 2.0})     # thicknesses in mm
    J          = fm.jacobian({'Water': 30.0, 'HA': 2.0}, ['Water', 'HA'])

Carries the fitted spectrum, the flux scale, the parameter covariance from the
bootstrap, and the per-acquisition threshold offset. That last one is not
cosmetic: the water check showed the spectrum SHAPE transfers between scans but
the absolute threshold does not, so anything applied to a new acquisition must
refit `delta_keV` for that acquisition rather than assume the slab value.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from physics import E_GRID, mu_lin, eta_cdte, threshold_response
import model as M


class ForwardModel:
    def __init__(self, params, kind='uniform', t=1.0, delta_keV=0.0,
                 cov=None, param_names=None, meta=None):
        self.p = np.asarray(params, float)
        self.kind = kind                 # 'uniform' | 'escape'
        self.t = float(t)
        self.delta_keV = float(delta_keV)
        self.cov = None if cov is None else np.asarray(cov, float)
        self.param_names = list(param_names or [])
        self.meta = dict(meta or {})

    # ---------------- spectra ----------------
    def spectra(self, p=None, delta=None):
        """S_L(E), S_H(E) including flux scale, on E_GRID (1..160 keV)."""
        p = self.p if p is None else np.asarray(p, float)
        dl = self.delta_keV if delta is None else float(delta)
        q = p.copy()
        q[2] += dl; q[3] += dl                     # shift both thresholds
        if self.kind == 'escape':
            import escape_test as ET
            S_L, S_H = ET.esc_shape(q)
        else:
            S_L, S_H = M.spectra_shape(q)
        return self.t * S_L, self.t * S_H

    # ---------------- prediction ----------------
    def _atten(self, a):
        e = np.zeros_like(E_GRID)
        for mat, mm in a.items():
            if mm:
                e = e + mm * mu_lin(mat)
        return np.exp(-e)

    def predict(self, a, p=None, delta=None):
        S_L, S_H = self.spectra(p, delta)
        w = self._atten(a)
        return float((S_L * w).sum()), float((S_H * w).sum())

    def jacobian(self, a, materials, p=None, delta=None):
        """d(lam_L, lam_H)/d(thickness) for the listed materials. Shape (2, n_mat)."""
        S_L, S_H = self.spectra(p, delta)
        w = self._atten(a)
        J = np.empty((2, len(materials)))
        for j, mat in enumerate(materials):
            mu = mu_lin(mat)
            J[0, j] = -(S_L * w * mu).sum()
            J[1, j] = -(S_H * w * mu).sum()
        return J

    def fisher(self, a, materials, **kw):
        """Poisson Fisher information for one measurement of (L, H)."""
        lam = np.array(self.predict(a, **kw))
        J = self.jacobian(a, materials, **kw)
        return J.T @ np.diag(1.0 / np.maximum(lam, 1e-12)) @ J

    def crlb(self, a, materials, **kw):
        """Cramer-Rao sd on each thickness, and the whitened-Jacobian condition number."""
        lam = np.array(self.predict(a, **kw))
        J = self.jacobian(a, materials, **kw)
        Jw = J / np.sqrt(np.maximum(lam, 1e-12))[:, None]
        F = Jw.T @ Jw
        try:
            C = np.linalg.inv(F)
            sd = np.sqrt(np.clip(np.diag(C), 0, None))
        except np.linalg.LinAlgError:
            sd = np.full(len(materials), np.inf)
        s = np.linalg.svd(Jw, compute_uv=False)
        kappa = float(s[0] / max(s[-1], 1e-30))
        return sd, kappa

    # ---------------- io ----------------
    def save(self, path):
        np.savez(path, p=self.p, kind=self.kind, t=self.t,
                 delta_keV=self.delta_keV,
                 cov=np.array([]) if self.cov is None else self.cov,
                 param_names=np.array(self.param_names, dtype=object),
                 meta_keys=np.array(list(self.meta.keys()), dtype=object),
                 meta_vals=np.array([str(v) for v in self.meta.values()], dtype=object))

    @staticmethod
    def load(path):
        d = np.load(path, allow_pickle=True)
        cov = d['cov']
        meta = dict(zip(list(d['meta_keys']), list(d['meta_vals'])))
        return ForwardModel(d['p'], str(d['kind']), float(d['t']),
                            float(d['delta_keV']),
                            None if cov.size == 0 else cov,
                            list(d['param_names']), meta)


def ideal_model(fm, E_L=15.0, E_H=30.0):
    """Same beam and sensor, but NOMINAL sharp thresholds and no energy-loss tail.

    This is the reference the real detector is measured against: what the two bins
    could distinguish if the thresholds were perfect.
    """
    q = fm.p.copy()
    q[2], q[3] = E_L, E_H
    q[4] = q[5] = 0.2                     # effectively sharp
    if fm.kind == 'escape':
        q[8] = 0.0                        # f0: no escape
    else:
        q[8] = 0.0                        # f_tail: no tail
    out = ForwardModel(q, fm.kind, fm.t, 0.0, None, fm.param_names,
                       {**fm.meta, 'variant': 'ideal sharp thresholds, no tail'})
    return out
