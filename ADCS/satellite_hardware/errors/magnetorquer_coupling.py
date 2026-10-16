__all__ = ["MagnetorquerCoupling"]

import numpy as np


class MagnetorquerCoupling:
    r"""
    The field a spacecraft's own magnetorquers produce at a magnetometer.

    A torquer with dipole moment :math:`\mathbf{m}` (A m\ :sup:`2`) produces a
    field at the magnetometer, directly and through the soft iron in between,
    that on a small spacecraft can be far larger than the sensor's noise.
    The coupling is linear in the moment and depends only on the layout, so
    it is one matrix :math:`\mathbf{K}` (tesla per A m\ :sup:`2`, body frame):

    .. math::

        \mathbf{B}_{\mathrm{torquers}} = \mathbf{K}\,\sum_i m_i\,\hat{\mathbf{a}}_i,

    with :math:`m_i` the moment each torquer last applied and
    :math:`\hat{\mathbf{a}}_i` its axis. Attach one instance to each
    magnetometer that should see it (``torquer_coupling=``); the satellite
    connects it to its torquers when it is built. The magnetometer then reads
    the ambient field plus this one. The moment used is the one the torquers
    applied over the previous step, so no duty cycling is modelled: a
    magnetometer that is read while the torquers are on sees them. The
    estimator's clone of the satellite does not carry the coupling.

    For a bare dipole at distance :math:`r` along the unit vector
    :math:`\hat{\mathbf{r}}` the matrix is
    :math:`\frac{\mu_0}{4\pi r^3}(3\hat{\mathbf{r}}\hat{\mathbf{r}}^T - I)`,
    about :math:`10^{-7}/r^3`; with soft iron in the path the measured matrix
    is the thing to use.

    Parameters
    ----------
    coupling : numpy.ndarray
        The 3x3 matrix :math:`\mathbf{K}`.
    """

    def __init__(self, coupling: np.ndarray) -> None:
        matrix = np.asarray(coupling, dtype=float)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("coupling must be a finite 3x3 matrix (tesla per A m^2)")
        self.coupling = matrix.copy()
        self.torquers: list = []

    def attach(self, torquers) -> None:
        """Connect the torquers whose moments produce the field; the satellite does this."""
        self.torquers = list(torquers)

    def moment(self) -> np.ndarray:
        """The total dipole moment the attached torquers last applied, in the body frame."""
        total = np.zeros(3)
        for torquer in self.torquers:
            total = total + float(getattr(torquer, "last_moment", 0.0)) * np.asarray(torquer.axis, dtype=float)
        return total

    def field(self) -> np.ndarray:
        """The torquers' field at the magnetometer, in the body frame (tesla)."""
        return self.coupling @ self.moment()
