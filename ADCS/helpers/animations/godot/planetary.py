"""Linear IAU planet orientation models (PCK00011 terms)."""

import numpy as np
from scipy.spatial.transform import Rotation

MODELS = {
    "mercury": (281.0103, -0.0328, 61.4155, -0.0049, 329.5988, 6.1385108),
    "venus": (272.76, 0.0, 67.16, 0.0, 160.20, -1.4813688),
    "mars": (317.269202, -0.10927547, 54.432516, -0.05827105, 176.049863, 350.891982443297),
    "jupiter": (268.056595, -0.006499, 64.495303, 0.002413, 284.95, 870.536),
    "saturn": (40.589, -0.036, 83.537, -0.004, 38.90, 810.7939024),
    "uranus": (257.311, 0.0, -15.175, 0.0, 203.81, -501.1600928),
    "neptune": (299.36, 0.0, 43.46, 0.0, 249.978, 541.1397757),
}


def orientation(name: str, jd_tt: np.ndarray) -> np.ndarray:
    """Body-to-GCRS scalar-first quaternion at each Julian date."""
    ra0, ra_rate, dec0, dec_rate, w0, spin = MODELS[name]
    days = np.atleast_1d(jd_tt) - 2451545.0
    centuries = days / 36525.0
    ra = np.deg2rad(ra0 + ra_rate * centuries + 90.0)
    dec = np.deg2rad(90.0 - dec0 - dec_rate * centuries)
    w = np.deg2rad(w0 + spin * days)
    mats = Rotation.from_euler("z", ra).as_matrix() @ Rotation.from_euler("x", dec).as_matrix() @ Rotation.from_euler("z", w).as_matrix()
    xyzw = Rotation.from_matrix(mats).as_quat()
    return xyzw[:, [3, 0, 1, 2]]
