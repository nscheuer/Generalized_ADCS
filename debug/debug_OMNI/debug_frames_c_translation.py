"""Compare the ADCS Omni C++ frame helpers with their Python reference."""

import numpy as np

from ADCS_Omni.ADCS_C.helpers.math import frames as c_frames
from ADCS_Omni.ADCS_Python.helpers.math.frames import conversions as py_conversions
from ADCS_Omni.ADCS_Python.helpers.math.frames import frames as py_frames


def main():
    jd_ut1 = 2459999.123
    jd_tt = 2459999.124
    xp, yp = 1.0e-6, -2.0e-6
    position = np.array([7000.0, -1200.0, 500.0])
    velocity = np.array([1.2, 7.1, -0.4])
    quaternion = np.array([0.8, 0.1, -0.2, 0.3])
    quaternion /= np.linalg.norm(quaternion)
    context = {
        "jd_ut1": jd_ut1,
        "jd_tt": jd_tt,
        "xp": xp,
        "yp": yp,
        "position_gcrf": position,
        "velocity_gcrf": velocity,
        "q_body_from_gcrf": quaternion,
    }

    for name, args in (
        ("gcrf_to_lvlh_matrix", (position, velocity)),
        ("gcrf_to_ecef_matrix", (jd_ut1, jd_tt, xp, yp)),
        ("ecef_to_gcrf_matrix", (jd_ut1, jd_tt, xp, yp)),
        ("gcrf_to_teme_matrix", (jd_tt,)),
        ("teme_to_gcrf_matrix", (jd_tt,)),
        ("teme_to_ecef_matrix", (jd_ut1, xp, yp)),
        ("ecef_to_teme_matrix", (jd_ut1, xp, yp)),
        ("ecef_to_geodetic", (position * 1000.0,)),
        ("gcrf_position_to_geodetic", (position * 1000.0, jd_ut1, jd_tt, xp, yp)),
    ):
        python_result = getattr(py_conversions, name)(*args)
        c_result = getattr(c_frames, name)(*args)
        np.testing.assert_allclose(c_result, python_result, rtol=1e-13, atol=1e-12)

    for source in ("GCRF", "ECEF", "BODY", "LVLH", "TEME"):
        for target in ("GCRF", "ECEF", "BODY", "LVLH", "TEME"):
            python_result = py_conversions.frame_rotation(source, target, **context)
            c_result = c_frames.frame_rotation(source, target, **context)
            np.testing.assert_allclose(c_result, python_result, rtol=1e-13, atol=1e-12)

    vector = np.array([1.0, -2.0, 0.5])
    covariance = np.array([[2.0, 0.1, 0.2], [0.1, 3.0, 0.3], [0.2, 0.3, 4.0]])
    np.testing.assert_allclose(
        c_frames.inertial_to_body(vector, quaternion),
        py_frames.inertial_to_body(vector, quaternion), rtol=1e-13, atol=1e-12,
    )
    np.testing.assert_allclose(
        c_frames.body_to_inertial(vector, quaternion),
        py_frames.body_to_inertial(vector, quaternion), rtol=1e-13, atol=1e-12,
    )
    covariance_rotation = c_frames.frame_rotation("GCRF", "BODY", **context)
    np.testing.assert_allclose(
        c_frames.transform_covariance(covariance, covariance_rotation),
        py_frames.transform_covariance(covariance, covariance_rotation), rtol=1e-13, atol=1e-12,
    )
    np.testing.assert_allclose(
        c_frames.convert_vector(vector, "GCRF", "ECEF", jd_ut1=jd_ut1, jd_tt=jd_tt),
        py_conversions.convert_vector(vector, "GCRF", "ECEF", jd_ut1=jd_ut1, jd_tt=jd_tt),
        rtol=1e-13, atol=1e-12,
    )
    print("ADCS Omni C++ frame translation: OK")


if __name__ == "__main__":
    main()
