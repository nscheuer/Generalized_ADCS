"""Shared magnetic and optical attitude corrections for EKF and MEKF."""

import numpy as np
from numba import njit
from ...sensors.kernels import (
    vector_measurement,
    scalar_sun_measurement,
    fine_direction_measurement,
    fine_direction_available,
)
from GenADCS.backends.python.helpers.math.rotations.quaternion.algebra import (
    multiply,
    normalize_quaternion,
)
from GenADCS.backends.python.helpers.math.rotations.quaternion.interpolation import (
    rotation_vector_to_quaternion,
)


@njit(cache=True)
def measurement(q, i, config, references):
    (
        code,
        ori,
        mis,
        bias,
        chol,
        normal,
        fov,
        scalars,
        enabled,
        offsets,
        counts,
        starts,
    ) = config
    kind = code[i]
    if kind == 1:
        return vector_measurement(
            references[0],
            q,
            ori[i],
            mis[i],
            bias[i, :3],
            chol[i, :3, :3],
            scalars[i, 0],
            True,
            False,
        )
    if kind == 3:
        return np.array(
            [
                scalar_sun_measurement(
                    references[1],
                    q,
                    ori[i],
                    mis[i],
                    normal[i],
                    scalars[i, 1],
                    bias[i, 0],
                    scalars[i, 2],
                    scalars[i, 0],
                    True,
                    False,
                )
            ]
        )
    direction = references[1] if kind == 4 else references[2]
    if not fine_direction_available(direction, q, ori[i], mis[i], fov[i], True, True):
        return np.full(2, np.nan)
    return fine_direction_measurement(
        direction, q, ori[i], mis[i], bias[i, :2], chol[i, :2, :2], True, False
    )


@njit(cache=True)
def update(q, w, p, readings, config, references, full, flags=0, diagnostics=None):
    for i in range(config[0].size):
        kind = config[0][i]
        if not config[8][i] or kind not in (1, 3, 4, 5):
            continue
        width = 3 if kind == 1 else (1 if kind == 3 else 2)
        observed = readings[config[9][i] : config[9][i] + width]
        if not np.all(np.isfinite(observed)):
            continue
        if references is None:
            raise ValueError(
                "magnetic and optical corrections require inertial references"
            )
        expected = measurement(q, i, config, references)
        if not np.all(np.isfinite(expected)):
            continue
        h = np.zeros((width, p.shape[0]))
        eps = 1e-7
        for j in range(4 if full else 3):
            if full:
                qp, qm = q.copy(), q.copy()
                qp[j] += eps
                qm[j] -= eps
                qp, qm = normalize_quaternion(qp), normalize_quaternion(qm)
            else:
                d = np.zeros(3)
                d[j] = eps
                qp = multiply(rotation_vector_to_quaternion(d), q)
                qm = multiply(rotation_vector_to_quaternion(-d), q)
            h[:, j] = (
                measurement(qp, i, config, references)
                - measurement(qm, i, config, references)
            ) / (2 * eps)
        if not np.all(np.isfinite(h)):
            continue
        r = config[4][i, :width, :width] @ config[4][i, :width, :width].T
        if kind == 3:
            r[0, 0] = config[7][i, 2] ** 2
        residual = observed - expected
        if kind == 4 or kind == 5:
            residual = (residual + np.pi) % (2 * np.pi) - np.pi
        innovation = h @ p @ h.T + r
        gain = np.linalg.solve(innovation, h @ p).T
        delta = gain @ residual
        if diagnostics is not None:
            reported_gain = gain
            if full:
                transform = np.zeros((6, 7))
                transform[:3, 0] = -2 * q[1:]
                v = q[1:]
                cross = np.array(
                    [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
                )
                transform[:3, 1:4] = 2 * (q[0] * np.eye(3) + cross)
                transform[3:, 4:] = np.eye(3)
                reported_gain = transform @ gain
            if flags & 16:
                diagnostics[:3] = np.nan
                diagnostics[:width] = residual
            if flags & 32:
                block = np.full((3, 3), np.nan)
                block[:width, :width] = innovation
                diagnostics[3:12] = block.ravel()
            if flags & 64:
                diagnostics[12] = residual @ np.linalg.solve(innovation, residual)
            if flags & 256:
                diagnostics[14:17] = np.nan
                diagnostics[14 : 14 + width] = residual - h @ delta
            if flags & 512:
                block = np.full((6, 3), np.nan)
                block[:, :width] = reported_gain
                diagnostics[17:35] = block.ravel()
        a = np.eye(p.shape[0]) - gain @ h
        p = a @ p @ a.T + gain @ r @ gain.T
        if full:
            q = normalize_quaternion(q + delta[:4])
            w = w + delta[4:]
            reset = np.eye(7)
            reset[:4, :4] -= np.outer(q, q)
        else:
            q = normalize_quaternion(
                multiply(rotation_vector_to_quaternion(delta[:3]), q)
            )
            w = w + delta[3:]
            d = delta[:3]
            reset = np.eye(6)
            reset[:3, :3] += 0.5 * np.array(
                [[0.0, -d[2], d[1]], [d[2], 0.0, -d[0]], [-d[1], d[0], 0.0]]
            )
        p = reset @ p @ reset.T
        p = 0.5 * (p + p.T)
    return q, w, p
