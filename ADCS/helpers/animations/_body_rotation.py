"""IAU body orientation models for the solar-system viewer.

The constants are the linear pole and prime-meridian terms from NAIF's
``pck00011.tpc``. They are adequate for a fast visualization, and use the
same J2000 inertial frame as the bundled planetary ephemeris.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class RotationModel:
    # IAU pole right ascension/declination at J2000 and degrees per century.
    pole_ra: float
    pole_ra_rate: float
    pole_dec: float
    pole_dec_rate: float
    # Prime-meridian angle at J2000 and degrees per TDB day.
    prime_meridian: float
    spin_rate: float


ROTATION_MODELS = {
    'sun': RotationModel(286.13, 0.0, 63.87, 0.0, 84.176, 14.18440),
    'mercury': RotationModel(281.0103, -0.0328, 61.4155, -0.0049,
                             329.5988, 6.1385108),
    'venus': RotationModel(272.76, 0.0, 67.16, 0.0, 160.20, -1.4813688),
    'earth': RotationModel(0.0, -0.641, 90.0, -0.557,
                           190.147, 360.9856235),
    'moon': RotationModel(269.9949, 0.0031, 66.5392, 0.0130,
                          38.3213, 13.17635815),
    'mars': RotationModel(317.269202, -0.10927547, 54.432516, -0.05827105,
                          176.049863, 350.891982443297),
    'jupiter': RotationModel(268.056595, -0.006499, 64.495303, 0.002413,
                             284.95, 870.536),
    'saturn': RotationModel(40.589, -0.036, 83.537, -0.004,
                            38.90, 810.7939024),
    'uranus': RotationModel(257.311, 0.0, -15.175, 0.0,
                            203.81, -501.1600928),
    'neptune': RotationModel(299.36, 0.0, 43.46, 0.0,
                             249.978, 541.1397757),
}


def body_orientation_quaternion(name: str, tdb_julian_date: float):
    """Return body-local to J2000 orientation at a TDB Julian date.

    The PCK Euler angles define the inertial-to-body frame. Transposing that
    transform gives the local-to-inertial matrix used by Panda3D. The final
    half-turn aligns the viewer's equirectangular texture convention (u=0 at
    the seam, u=0.5 at longitude zero) with the IAU prime meridian.
    """
    from panda3d.core import Quat

    model = ROTATION_MODELS[name]
    days = float(tdb_julian_date) - 2451545.0
    centuries = days / 36525.0
    ra = np.deg2rad(model.pole_ra + model.pole_ra_rate * centuries)
    dec = np.deg2rad(model.pole_dec + model.pole_dec_rate * centuries)
    w = np.deg2rad(model.prime_meridian + model.spin_rate * days)

    # SPICE body-fixed -> inertial: R3(90+RA) R1(90-DEC) R3(W).
    # Matrix convention here is the standard active, right-handed convention.
    cz, sz = np.cos(ra + np.pi / 2), np.sin(ra + np.pi / 2)
    cw, sw = np.cos(w), np.sin(w)
    cd, sd = np.cos(np.pi / 2 - dec), np.sin(np.pi / 2 - dec)
    rz_ra = np.array(((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0)))
    rx_dec = np.array(((1.0, 0.0, 0.0), (0.0, cd, -sd), (0.0, sd, cd)))
    rz_w = np.array(((cw, -sw, 0.0), (sw, cw, 0.0), (0.0, 0.0, 1.0)))
    texture_meridian = np.diag((-1.0, -1.0, 1.0))
    matrix = rz_ra @ rx_dec @ rz_w @ texture_meridian
    x, y, z, scalar = Rotation.from_matrix(matrix).as_quat()
    return Quat(float(scalar), float(x), float(y), float(z))
