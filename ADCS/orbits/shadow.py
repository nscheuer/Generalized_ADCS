__all__ = ["shadow_factor"]

import numpy as np

from ADCS.orbits.universal_constants import EarthConstants


def shadow_factor(
    r_sat: np.ndarray,
    r_sun: np.ndarray,
    R_occ: float = EarthConstants.R_e,
    R_sun: float = EarthConstants.R_sun,
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
        Sun position relative to the occulting body's centre, same shape and
        units as ``r_sat``.
    :type r_sun: numpy.ndarray
    :param R_occ:
        Radius of the occulting body in km. Defaults to Earth's equatorial radius.
    :type R_occ: float
    :param R_sun:
        Solar radius in km.
    :type R_sun: float

    :return:
        Visible solar-disk fraction in ``[0, 1]``: a float for ``(3,)`` input,
        otherwise an array of shape ``(N,)``.
    :rtype: float | numpy.ndarray
    """
    r_sat = np.asarray(r_sat, dtype=float)
    r_sun = np.asarray(r_sun, dtype=float)
    scalar = r_sat.ndim == 1
    r_sat = np.atleast_2d(r_sat)
    r_sun = np.atleast_2d(r_sun)
    if r_sat.shape[-1] != 3 or r_sun.shape[-1] != 3:
        raise ValueError("r_sat and r_sun must have a trailing dimension of 3")

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

    nu = np.clip(nu, 0.0, 1.0)
    return float(nu[0]) if scalar else nu
