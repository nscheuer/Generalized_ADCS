__all__ = ["shadow_factor"]

import math

import numpy as np

from ADCS.orbits.universal_constants import EarthConstants


def shadow_factor(
    r_sat: np.ndarray,
    r_sun: np.ndarray,
    R_occ: float = EarthConstants.R_e,
    R_sun: float = 695700.0,
) -> np.ndarray | float:
    r"""
    Fraction of the solar disk visible from the spacecraft (conical shadow model).

    Implements the conical umbra/penumbra model of Montenbruck & Gill,
    *Satellite Orbits* (2000), Sec. 3.4.2, treating the Sun and the occulting
    body as spheres. From the spacecraft, the apparent solar radius :math:`a`,
    the apparent radius of the occulting body :math:`b` and their apparent
    centre separation :math:`c` are

    .. math::

        a = \arcsin\frac{R_\odot}{\lVert \mathbf{r}_\odot - \mathbf{r} \rVert},\quad
        b = \arcsin\frac{R_\oplus}{\lVert \mathbf{r} \rVert},\quad
        c = \arccos\frac{-\mathbf{r}\cdot(\mathbf{r}_\odot - \mathbf{r})}
                       {\lVert \mathbf{r} \rVert\,\lVert \mathbf{r}_\odot - \mathbf{r} \rVert}.

    The returned factor :math:`\nu` is

    - ``1`` for :math:`c \ge a + b` (full sunlight),
    - ``0`` for :math:`c \le b - a` (umbra),
    - :math:`1 - b^2/a^2` for :math:`c \le a - b` (annular, occulter inside the disk),
    - :math:`1 - A/(\pi a^2)` otherwise (penumbra), with the overlap area

    .. math::

        x = \frac{c^2 + a^2 - b^2}{2c},\quad y = \sqrt{a^2 - x^2},\quad
        A = a^2 \arccos\frac{x}{a} + b^2 \arccos\frac{c - x}{b} - c\,y.

    This is the value the 3-D viewer applies to sunlight, so that rendered
    eclipses match what the simulation computed.

    :param r_sat:
        Spacecraft position relative to the occulting body's centre, shape ``(3,)``
        or ``(N, 3)``, in km.
    :type r_sat: numpy.ndarray
    :param r_sun:
        Sun position relative to the occulting body's centre, in km: ``(3,)`` or
        ``(N, 3)``. One position is paired with every row of the other input.
    :type r_sun: numpy.ndarray
    :param R_occ:
        Radius of the occulting body in km. Defaults to Earth's equatorial radius.
    :type R_occ: float
    :param R_sun:
        Solar radius in km.
    :type R_sun: float

    :return:
        Visible solar-disk fraction in ``[0, 1]``: a float when both inputs are
        ``(3,)``, otherwise an array of shape ``(N,)``.
    :rtype: float | numpy.ndarray
    """
    r_sat = np.asarray(r_sat, dtype=float)
    r_sun = np.asarray(r_sun, dtype=float)
    if r_sat.shape == (3,) and r_sun.shape == (3,):
        # Per-step callers (live streaming) pay ~10x less without numpy overhead.
        return _shadow_factor_scalar(r_sat.tolist(), r_sun.tolist(), R_occ, R_sun)
    if r_sat.shape[-1:] != (3,) or r_sun.shape[-1:] != (3,) or r_sat.ndim > 2 or r_sun.ndim > 2:
        raise ValueError("r_sat and r_sun must have shape (3,) or (N, 3)")
    try:  # one position against many (either way round) broadcasts
        r_sat, r_sun = np.broadcast_arrays(np.atleast_2d(r_sat), np.atleast_2d(r_sun))
    except ValueError:
        raise ValueError(f"r_sat {r_sat.shape} and r_sun {r_sun.shape} have different numbers of rows") from None

    s = r_sun - r_sat
    r_norm = np.linalg.norm(r_sat, axis=-1)
    s_norm = np.linalg.norm(s, axis=-1)
    if np.any(r_norm <= R_occ):
        raise ValueError("spacecraft position lies inside the occulting body")

    a = np.arcsin(np.clip(R_sun / s_norm, 0.0, 1.0))
    b = np.arcsin(np.clip(R_occ / r_norm, 0.0, 1.0))
    cos_c = -np.einsum("ij,ij->i", r_sat, s) / (r_norm * s_norm)
    c = np.arccos(np.clip(cos_c, -1.0, 1.0))

    nu = np.ones_like(c)

    umbra = c <= b - a
    annular = c <= a - b
    partial = (c < a + b) & ~umbra & ~annular

    nu[umbra] = 0.0
    nu[annular] = 1.0 - (b[annular] / a[annular]) ** 2

    if np.any(partial):
        ap, bp, cp = a[partial], b[partial], c[partial]
        x = (cp**2 + ap**2 - bp**2) / (2.0 * cp)
        y = np.sqrt(np.clip(ap**2 - x**2, 0.0, None))
        area = (
            ap**2 * np.arccos(np.clip(x / ap, -1.0, 1.0))
            + bp**2 * np.arccos(np.clip((cp - x) / bp, -1.0, 1.0))
            - cp * y
        )
        nu[partial] = 1.0 - area / (np.pi * ap**2)

    return np.clip(nu, 0.0, 1.0)


def _shadow_factor_scalar(r: list, sun: list, R_occ: float, R_sun: float) -> float:
    """Same model as :func:`shadow_factor` for one position, with the math module."""
    s = [sun[0] - r[0], sun[1] - r[1], sun[2] - r[2]]
    r_norm = math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
    s_norm = math.sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2])
    if r_norm <= R_occ:
        raise ValueError("spacecraft position lies inside the occulting body")
    a = math.asin(min(R_sun / s_norm, 1.0))
    b = math.asin(min(R_occ / r_norm, 1.0))
    cos_c = -(r[0] * s[0] + r[1] * s[1] + r[2] * s[2]) / (r_norm * s_norm)
    c = math.acos(max(-1.0, min(1.0, cos_c)))
    if c >= a + b:
        return 1.0
    if c <= b - a:
        return 0.0
    if c <= a - b:
        return min(1.0, max(0.0, 1.0 - (b / a) ** 2))
    x = (c * c + a * a - b * b) / (2.0 * c)
    y = math.sqrt(max(a * a - x * x, 0.0))
    area = (a * a * math.acos(max(-1.0, min(1.0, x / a)))
            + b * b * math.acos(max(-1.0, min(1.0, (c - x) / b)))
            - c * y)
    return min(1.0, max(0.0, 1.0 - area / (math.pi * a * a)))
