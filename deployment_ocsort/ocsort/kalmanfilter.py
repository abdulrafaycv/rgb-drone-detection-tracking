"""
Self-contained Kalman filter for OC-SORT — no filterpy dependency.
Adapted from OC-SORT (https://github.com/noahcao/OC_SORT, MIT License)
and the original filterpy implementation by Roger Labbe (MIT License).
Only includes the methods required by OC-SORT's KalmanBoxTracker.
"""
import numpy as np
from numpy import dot, zeros, eye


def _reshape_z(z, dim_z, x_ndim):
    """Reshape z to (dim_z, 1) column vector."""
    z = np.atleast_2d(z)
    if z.shape[1] == dim_z:
        z = z.T
    if x_ndim == 1:
        z = z[:, 0]
    return z


class KalmanFilterNew:
    """
    Minimal Kalman filter for OC-SORT tracking.
    State: (x, y, s, r, vx, vy, vs) — centre, scale, aspect ratio + velocities.
    """

    def __init__(self, dim_x, dim_z, dim_u=0):
        self.dim_x = dim_x
        self.dim_z = dim_z

        self.x = zeros((dim_x, 1))
        self.P = eye(dim_x)
        self.Q = eye(dim_x)
        self.B = None
        self.F = eye(dim_x)
        self.H = zeros((dim_z, dim_x))
        self.R = eye(dim_z)
        self._alpha_sq = 1.
        self.M = zeros((dim_x, dim_z))
        self.z = np.array([[None] * dim_z]).T

        self.K = zeros((dim_x, dim_z))
        self.y = zeros((dim_z, 1))
        self.S = zeros((dim_z, dim_z))
        self.SI = zeros((dim_z, dim_z))
        self._I = eye(dim_x)

        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        self.inv = np.linalg.inv

        # OC-SORT specific additions
        self.history_obs = []
        self.attr_saved = None
        self.observed = False

    def predict(self, u=None, B=None, F=None, Q=None):
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q

        if B is not None and u is not None:
            self.x = dot(F, self.x) + dot(B, u)
        else:
            self.x = dot(F, self.x)

        self.P = self._alpha_sq * dot(dot(F, self.P), F.T) + Q
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def update(self, z, R=None, H=None):
        self.history_obs.append(z)

        if z is None:
            if self.observed:
                self.freeze()
            self.observed = False
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = zeros((self.dim_z, 1))
            return

        if not self.observed:
            self.unfreeze()
        self.observed = True

        if R is None:
            R = self.R
        if H is None:
            z = _reshape_z(z, self.dim_z, self.x.ndim)
            H = self.H

        self.y = z - dot(H, self.x)
        PHT = dot(self.P, H.T)
        self.S = dot(H, PHT) + R
        self.SI = self.inv(self.S)
        self.K = dot(PHT, self.SI)
        self.x = self.x + dot(self.K, self.y)
        I_KH = self._I - dot(self.K, H)
        self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(self.K, R), self.K.T)
        self.z = z.copy()
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def freeze(self):
        """Save state before entering non-observation period (for OC-SORT re-update)."""
        self.attr_saved = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                           for k, v in self.__dict__.items()
                           if k not in ('attr_saved', 'history_obs', 'inv')}
        self.attr_saved['history_obs'] = list(self.history_obs)

    def unfreeze(self):
        """Restore saved state and apply online smoothing to fill the gap."""
        if self.attr_saved is None:
            return

        new_history = list(self.history_obs)
        for key, val in self.attr_saved.items():
            setattr(self, key, val)
        self.attr_saved = None

        # Drop the last obs entry (it was None that triggered freeze)
        self.history_obs = self.history_obs[:-1]

        # Find the two most recent real observations to interpolate between
        indices = [i for i, d in enumerate(new_history) if d is not None]
        if len(indices) < 2:
            return
        index1, index2 = indices[-2], indices[-1]

        box1 = new_history[index1].flatten()
        x1, y1, s1, r1 = box1
        w1 = np.sqrt(max(s1 * r1, 1e-6))
        h1 = np.sqrt(max(s1 / max(r1, 1e-6), 1e-6))

        box2 = new_history[index2].flatten()
        x2, y2, s2, r2 = box2
        w2 = np.sqrt(max(s2 * r2, 1e-6))
        h2 = np.sqrt(max(s2 / max(r2, 1e-6), 1e-6))

        time_gap = max(index2 - index1, 1)
        dx, dy = (x2 - x1) / time_gap, (y2 - y1) / time_gap
        dw, dh = (w2 - w1) / time_gap, (h2 - h1) / time_gap

        for i in range(time_gap):
            nx = x1 + (i + 1) * dx
            ny = y1 + (i + 1) * dy
            nw = max(w1 + (i + 1) * dw, 1e-3)
            nh = max(h1 + (i + 1) * dh, 1e-3)
            ns = nw * nh
            nr = nw / nh
            new_box = np.array([nx, ny, ns, nr]).reshape((4, 1))
            self.update(new_box)
            if i < time_gap - 1:
                self.predict()
