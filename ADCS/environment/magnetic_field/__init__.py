r"""
Selectable geomagnetic field model.

Every field evaluation in the package goes through :func:`field_gc`, which
dispatches on a model name so that the implementation can be chosen per
orbit -- the magnetic counterpart of ``zonal_J`` for gravity:

``"igrf_numba"`` (default)
    The in-tree numba kernel, :func:`ADCS.environment.magnetic_field.igrf14_numba.igrf_gc`.

``"igrf_ppigrf"``
    The ``ppigrf`` package, which was the field model up to 0.1.8. Kept so
    that older results can be reproduced exactly. It is an optional
    dependency (``pip install "Generalized_ADCS[ppigrf]"``) and is imported
    only when selected.

``"wmm2025"`` and ``"chaos8"``
    WMM2025 and the internal core plus lithospheric CHAOS-8.7 field.
"""

from __future__ import annotations

from functools import lru_cache

from .igrf14_numba import coefficients_at, igrf_gc
from .igrf_ppigrf import _PPIGRF_CHUNK, _ppigrf_gc

__all__ = ["MAGNETIC_MODELS", "DEFAULT_MAGNETIC_MODEL", "field_gc", "make_magnetic_field", "coefficients_at", "igrf_gc", "MagneticFieldModel", "IGRF14", "IGRF14Numba", "IGRF14Ppigrf", "WMM2025", "WMM", "CHAOS8", "MagneticField"]

#: Names accepted by ``magnetic_model`` arguments throughout the package.
MAGNETIC_MODELS = ("igrf_numba", "igrf_ppigrf", "wmm2025", "chaos8")

#: Model used when none is requested.
DEFAULT_MAGNETIC_MODEL = "igrf_numba"


def _normalize_magnetic_model(magnetic_model) -> str:
    r"""
    Validate and normalize a magnetic field model name.

    :param magnetic_model:
        One of :data:`MAGNETIC_MODELS` (case-insensitive), or ``None`` for
        :data:`DEFAULT_MAGNETIC_MODEL`.
    :type magnetic_model: str or None
    :return:
        The normalized model name.
    :rtype: str
    :raises ValueError:
        If the name is not a known model.
    """
    if magnetic_model is None:
        return DEFAULT_MAGNETIC_MODEL
    value = str(magnetic_model).strip().lower()
    value = {"wmm": "wmm2025", "chaos-8": "chaos8", "chaos-8.7": "chaos8"}.get(value, value)
    if value in MAGNETIC_MODELS:
        return value
    raise ValueError(
        f"magnetic_model must be one of {MAGNETIC_MODELS}, got {magnetic_model!r}"
    )


def field_gc(magnetic_model, r, theta, phi, date):
    r"""
    Geomagnetic field in geocentric spherical components, from a chosen model.

    :param magnetic_model:
        One of :data:`MAGNETIC_MODELS`.
    :type magnetic_model: str
    :param r:
        Geocentric radius [km].
    :type r: float or numpy.ndarray
    :param theta:
        Geocentric colatitude [deg].
    :type theta: float or numpy.ndarray
    :param phi:
        Geocentric longitude [deg], positive east.
    :type phi: float or numpy.ndarray
    :param date:
        One date, or one date per sample (paired elementwise).
    :type date: datetime.datetime or sequence
    :return:
        ``(Br, Btheta, Bphi)`` in nT, each with the broadcast shape of the
        coordinates.
    :rtype: tuple
    :raises ValueError:
        If ``magnetic_model`` is unknown.
    :raises ImportError:
        If ``"igrf_ppigrf"`` is requested and ppigrf is not installed.

    :example:

    >>> from datetime import datetime
    >>> br, bt, bp = field_gc("igrf_numba", 6771.2, 45.0, 10.0, datetime(2024, 3, 1))
    """
    model = _normalize_magnetic_model(magnetic_model)
    if model == "igrf_numba":
        return igrf_gc(r, theta, phi, date)
    if model == "igrf_ppigrf":
        return _ppigrf_gc(r, theta, phi, date)
    return _named_model(model).field_gc(r, theta, phi, date)


@lru_cache(maxsize=2)
def _named_model(name):
    r"""Reuse the default WMM2025 or CHAOS8 model instance by name.

    :param name: Canonical magnetic model name.
    :type name: str
    :return: Cached concrete magnetic model.
    :rtype: MagneticField
    """
    return WMM2025() if name == "wmm2025" else CHAOS8()


from .magnetic_field import MagneticField
from .igrf14 import IGRF14
from .igrf14_numba import IGRF14Numba
from .igrf_ppigrf import IGRF14Ppigrf
from .wmm import WMM2025, WMM
from .chaos8 import CHAOS8


def make_magnetic_field(name: str | None = None) -> MagneticField:
    r"""Create the concrete model selected by a legacy name.

    :param name: Canonical or accepted alias for a magnetic model.
    :type name: str or None
    :return: New concrete magnetic field model instance.
    :rtype: MagneticField
    :raises ValueError: If the model name is unknown.
    """
    model = _normalize_magnetic_model(name)
    if model == "wmm2025":
        return WMM2025()
    if model == "chaos8":
        return CHAOS8()
    return IGRF14(model)


# Backwards-compatible model name; now a concrete class in this package.
MagneticFieldModel = IGRF14
MagneticField.IGRF14 = IGRF14
MagneticField.IGRF14Numba = IGRF14Numba
MagneticField.IGRF14Ppigrf = IGRF14Ppigrf
MagneticField.WMM2025 = WMM2025
MagneticField.WMM = WMM
MagneticField.CHAOS8 = CHAOS8
