r"""
Selectable geomagnetic field model.

Every field evaluation in the package goes through :func:`field_gc`, which
dispatches on a model name so that the implementation can be chosen per
orbit -- the magnetic counterpart of ``zonal_J`` for gravity:

``"igrf_numba"`` (default)
    The in-tree numba kernel, :func:`ADCS.environment.igrf.igrf_gc`.

``"igrf_ppigrf"``
    The ``ppigrf`` package, which was the field model up to 0.1.8. Kept so
    that older results can be reproduced exactly. It is an optional
    dependency (``pip install "Generalized_ADCS[ppigrf]"``) and is imported
    only when selected.

Both evaluate IGRF-14 from the same coefficients and agree to ~1e-12
relative; they differ only in speed. A further model (e.g. WMM) is a new
entry in :data:`MAGNETIC_MODELS` plus a branch in :func:`field_gc`.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from ADCS.environment.igrf import igrf_gc

__all__ = ["MAGNETIC_MODELS", "DEFAULT_MAGNETIC_MODEL", "field_gc"]

#: Names accepted by ``magnetic_model`` arguments throughout the package.
MAGNETIC_MODELS = ("igrf_numba", "igrf_ppigrf")

#: Model used when none is requested.
DEFAULT_MAGNETIC_MODEL = "igrf_numba"

#: Samples per ppigrf call when dates vary per sample. ppigrf forms the outer
#: product of dates and coordinates, so a chunk of m samples costs m^2 field
#: evaluations; this bounds that cost without changing any value.
_PPIGRF_CHUNK = 256


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
    if value in MAGNETIC_MODELS:
        return value
    raise ValueError(
        f"magnetic_model must be one of {MAGNETIC_MODELS}, got {magnetic_model!r}"
    )


def _import_ppigrf():
    try:
        import ppigrf
    except ImportError as exc:
        raise ImportError(
            "magnetic_model='igrf_ppigrf' needs the optional ppigrf package: "
            "pip install \"Generalized_ADCS[ppigrf]\""
        ) from exc
    return ppigrf


def _ppigrf_gc(r, theta, phi, date):
    r"""
    ``ppigrf.igrf_gc`` with dates paired elementwise with the coordinates.

    Same contract as :func:`ADCS.environment.igrf.igrf_gc`. ppigrf returns an
    ``(n_dates, N)`` grid; for one date that is a single row, and for one date
    per sample only the diagonal is wanted, which is taken chunk by chunk so
    the discarded off-diagonal work stays bounded.
    """
    ppigrf = _import_ppigrf()

    scalar = all(np.ndim(x) == 0 for x in (r, theta, phi))
    r_a, th_a, ph_a = np.broadcast_arrays(
        np.asarray(r, dtype=float), np.asarray(theta, dtype=float), np.asarray(phi, dtype=float)
    )
    shape = r_a.shape
    r_f, th_f, ph_f = (a.reshape(-1) for a in (r_a, th_a, ph_a))
    N = r_f.size

    single_date = isinstance(date, datetime) or np.ndim(date) == 0
    dates = None if single_date else list(date)
    if dates is not None and len(dates) == 1:
        single_date, date = True, dates[0]
    if dates is not None and len(dates) != N:
        raise ValueError(
            f"date has {len(dates)} entries; expected 1 or {N} to match the coordinates"
        )

    if single_date:
        out = ppigrf.igrf_gc(r_f, th_f, ph_f, date)
        comps = [np.asarray(c, dtype=float).reshape(N) for c in out]
    else:
        comps = [np.empty(N) for _ in range(3)]
        for i0 in range(0, N, _PPIGRF_CHUNK):
            i1 = min(i0 + _PPIGRF_CHUNK, N)
            out = ppigrf.igrf_gc(r_f[i0:i1], th_f[i0:i1], ph_f[i0:i1], dates[i0:i1])
            for dst, c in zip(comps, out):
                dst[i0:i1] = np.diagonal(np.asarray(c, dtype=float).reshape(i1 - i0, i1 - i0))

    if scalar:
        return tuple(float(c[0]) for c in comps)
    return tuple(c.reshape(shape) for c in comps)


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
    return _ppigrf_gc(r, theta, phi, date)
