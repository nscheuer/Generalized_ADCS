import numpy as np
from GenADCS.backends.python.hardware.ADCS_subsystem.estimator.attitude_estimator.attitude_ekf import (
    step as ekf,
)
from GenADCS.backends.python.hardware.ADCS_subsystem.estimator.attitude_estimator.attitude_mekf import (
    step as mekf,
)
from GenADCS.backends.python.hardware.ADCS_subsystem.estimator.attitude_estimator.direction_updates import (
    measurement,
)
from GenADCS.backends.python.helpers.math.rotations.quaternion.interpolation import (
    rotation_vector_to_quaternion,
)


def test_direction_corrections_reduce_error_and_preserve_covariance():
    q = np.array([1.0, 0, 0, 0])
    refs = np.array([[0.3, 0.2, 1.0], [0.4, 0.3, 1.0], [-0.3, 0.4, 1.0]])
    for kind in (1, 3, 4, 5):
        width = 3 if kind == 1 else (1 if kind == 3 else 2)
        config = (
            np.array([kind]),
            q[None, :],
            q[None, :],
            np.zeros((1, 6)),
            np.eye(6)[None, :, :] * 0.001,
            np.array([[0.0, 0, 1.0]]),
            np.ones((1, 2)),
            np.array([[100.0, 1.0, 0.001]]),
            np.ones(1, dtype=np.bool_),
            np.zeros(1, dtype=np.int64),
            np.zeros(8, dtype=np.int64),
            np.zeros(8, dtype=np.int64),
        )
        readings = measurement(q, 0, config, refs)
        initial = rotation_vector_to_quaternion(np.array([0.04, -0.03, 0.01]))
        for step, n in ((ekf, 7), (mekf, 6)):
            out, _, p = step(
                initial,
                np.zeros(3),
                np.eye(n) * 0.1,
                np.zeros((3, 3)),
                0.0,
                readings,
                config,
                references=refs,
            )
            assert np.linalg.norm(
                measurement(out, 0, config, refs) - readings
            ) < np.linalg.norm(measurement(initial, 0, config, refs) - readings)
            assert np.linalg.eigvalsh(p).min() > -1e-10
