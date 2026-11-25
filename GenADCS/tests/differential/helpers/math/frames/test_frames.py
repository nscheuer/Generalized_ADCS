import numpy as np


def test_rotation_and_covariance_outputs_agree(frame_backends):
    q = np.array([0.5, 0.5, 0.5, 0.5])
    vector = np.array([1.25, -3.0, 0.75])
    covariance = np.array([[4.0, 1.0, 0.0], [1.0, 2.0, 0.5], [0.0, 0.5, 3.0]])
    outputs = {}
    for name, module in frame_backends.items():
        dcm = module.inertial_to_body_matrix(q)
        outputs[name] = (
            dcm,
            module.inertial_to_body(vector, q),
            module.transform_covariance(covariance, dcm),
        )
    reference = outputs["python"]
    for result in outputs.values():
        for actual, expected in zip(result, reference):
            np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-14)


def test_frame_conversion_named_helpers_agree(frame_backends):
    jd = 2459123.25
    context = dict(jd_ut1=jd, jd_tt=jd + 0.0008, xp=1.2e-6, yp=-2.3e-6)
    names = (
        "gcrf_to_ecef_matrix",
        "ecef_to_gcrf_matrix",
        "gcrf_to_teme_matrix",
        "teme_to_gcrf_matrix",
        "teme_to_ecef_matrix",
        "ecef_to_teme_matrix",
    )
    values = {
        "gcrf_to_ecef_matrix": lambda m: m.gcrf_to_ecef_matrix(
            context["jd_ut1"], context["jd_tt"], context["xp"], context["yp"]
        ),
        "ecef_to_gcrf_matrix": lambda m: m.ecef_to_gcrf_matrix(
            context["jd_ut1"], context["jd_tt"], context["xp"], context["yp"]
        ),
        "gcrf_to_teme_matrix": lambda m: m.gcrf_to_teme_matrix(context["jd_tt"]),
        "teme_to_gcrf_matrix": lambda m: m.teme_to_gcrf_matrix(context["jd_tt"]),
        "teme_to_ecef_matrix": lambda m: m.teme_to_ecef_matrix(
            context["jd_ut1"], context["xp"], context["yp"]
        ),
        "ecef_to_teme_matrix": lambda m: m.ecef_to_teme_matrix(
            context["jd_ut1"], context["xp"], context["yp"]
        ),
    }
    for name in names:
        results = [values[name](module) for module in frame_backends.values()]
        for result in results[1:]:
            np.testing.assert_allclose(result, results[0], rtol=2e-13, atol=2e-14)


def test_lvlh_geodetic_and_vector_conversion_agree(frame_backends):
    p, v = np.array([7e6, -2e6, 3e6]), np.array([1e3, 6.8e3, -2e3])
    vector = np.array([-3.0, 5.0, 8.0])
    results = []
    for module in frame_backends.values():
        lvlh = module.gcrf_to_lvlh_matrix(p, v)
        geo = module.ecef_to_geodetic(np.array([-4510000.0, 3500000.0, 3700000.0]))
        rotated = module.convert_vector(
            vector, "GCRF", "LVLH", position_gcrf=p, velocity_gcrf=v
        )
        results.append((lvlh, geo, rotated))
    for result in results[1:]:
        for actual, expected in zip(result, results[0]):
            np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-12)
