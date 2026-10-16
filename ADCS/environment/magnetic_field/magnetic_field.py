"""Parent interface for magnetic field models."""

from abc import ABC, abstractmethod

import numpy as np


class MagneticField(ABC):
    r"""Interface for an Earth magnetic field model.

    Implementations evaluate geocentric spherical field components in
    nanotesla. The common :meth:`field` method converts those components to
    Earth fixed Cartesian vectors in tesla.
    """

    def field(self, position_ecef_km, date):
        r"""Evaluate an Earth fixed magnetic field vector.

        :param position_ecef_km: One ECEF position ``(3,)`` or positions ``(N, 3)`` [km].
        :type position_ecef_km: numpy.ndarray
        :param date: One UTC date or one date per position.
        :type date: datetime.datetime or sequence
        :return: ECEF field vector or vectors [T], matching the position shape.
        :rtype: numpy.ndarray
        :raises ValueError: If a position has the wrong shape or zero radius.
        """
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
        r"""Evaluate geocentric spherical field components.

        :param r: Geocentric radius [km].
        :type r: float or numpy.ndarray
        :param theta: Geocentric colatitude [deg].
        :type theta: float or numpy.ndarray
        :param phi: East longitude [deg].
        :type phi: float or numpy.ndarray
        :param date: One UTC date or one date per position.
        :type date: datetime.datetime or sequence
        :return: Radial, southward, and eastward components [nT].
        :rtype: tuple
        """
        raise NotImplementedError
