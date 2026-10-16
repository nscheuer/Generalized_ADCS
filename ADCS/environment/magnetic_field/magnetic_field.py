"""Parent interface for magnetic field models."""

from abc import ABC, abstractmethod

import numpy as np


class MagneticField(ABC):
    """Evaluate Earth's magnetic field from an ECEF position and date."""

    def field(self, position_ecef_km, date):
        """Return ECEF magnetic field vectors in tesla for one or many positions."""
        position = np.asarray(position_ecef_km, dtype=float)
        if position.ndim not in (1, 2) or position.shape[-1] != 3:
            raise ValueError("position_ecef_km must have shape (3,) or (N, 3)")
        radius = np.linalg.norm(position, axis=-1)
        if np.any(radius <= 0.0):
            raise ValueError("position_ecef_km must be nonzero")

        theta = np.degrees(np.arccos(np.clip(position[..., 2] / radius, -1.0, 1.0)))
        phi = np.arctan2(position[..., 1], position[..., 0])
        br, bt, bp = self.field_gc(radius, theta, np.degrees(phi), date)

        sin_theta = np.sin(np.radians(theta))
        cos_theta = np.cos(np.radians(theta))
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        e_r = np.stack((sin_theta * cos_phi, sin_theta * sin_phi, cos_theta), axis=-1)
        e_theta = np.stack((cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta), axis=-1)
        e_phi = np.stack((-sin_phi, cos_phi, np.zeros_like(phi)), axis=-1)
        return (np.asarray(br)[..., None] * e_r
                + np.asarray(bt)[..., None] * e_theta
                + np.asarray(bp)[..., None] * e_phi) * 1e-9

    @abstractmethod
    def field_gc(self, r, theta, phi, date):
        """Return ``(Br, Btheta, Bphi)`` in nT for radius km and angles deg."""
        raise NotImplementedError
