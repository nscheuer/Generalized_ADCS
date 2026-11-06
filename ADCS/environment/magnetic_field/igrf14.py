"""IGRF-14 magnetic field model backed by the existing selectable kernels."""

from .magnetic_field import MagneticField


class IGRF14(MagneticField):
    """IGRF-14 with either the Numba or ppigrf backend.

    Coefficients are published by IAGA Division V Working Group V-MOD.
    Cite Beggan et al. (2025), doi:10.1186/s40623-025-02360-0, and the IGRF-14
    coefficient data DOI listed in the project citation guide.

    ``field(position_ecef_km, date)`` accepts one ECEF position or an ``(N, 3)``
    array and returns ECEF field vectors in tesla. Dates may be a single
    ``datetime`` or one date per position.
    """

    def __init__(self, implementation: str = "igrf_numba", *, name: str | None = None) -> None:
        r"""Select an IGRF-14 backend.

        :param implementation: ``"igrf_numba"`` or ``"igrf_ppigrf"``.
        :type implementation: str
        :param name: Legacy alias for ``implementation``.
        :type name: str or None
        :return: None
        :rtype: None
        """
        from . import _normalize_magnetic_model

        if name is not None:
            implementation = name
        self.name = _normalize_magnetic_model(implementation)

    def field_gc(self, r, theta, phi, date):
        r"""Evaluate IGRF-14 in geocentric spherical coordinates.

        :param r: Geocentric radius [km].
        :type r: float or numpy.ndarray
        :param theta: Geocentric colatitude [deg].
        :type theta: float or numpy.ndarray
        :param phi: East longitude [deg].
        :type phi: float or numpy.ndarray
        :param date: One UTC date or a date per position.
        :type date: datetime.datetime or sequence
        :return: Radial, southward, and eastward field components [nT].
        :rtype: tuple
        """
        from . import field_gc

        return field_gc(self.name, r, theta, phi, date)
