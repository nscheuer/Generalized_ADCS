__all__ = ["vector_alignment_error"]

import numpy as np
from ADCS.helpers.math_helpers import normalize, rot_mat, norm

def vector_alignment_error(q: np.ndarray, eci_goal: np.ndarray, body_boresight: np.ndarray) -> np.ndarray:
    v_bore = normalize(body_boresight)
    R_b2i = rot_mat(q)                    # q: body -> ECI (Hamilton)
    v_goal_body = normalize(R_b2i.T @ eci_goal)

    dot = np.dot(v_bore, v_goal_body)

    if dot < -0.9999:
        # 180° case: pick any orthogonal axis
        axis = np.cross(v_bore, [1.0, 0.0, 0.0])
        if norm(axis) < 1e-3:
            axis = np.cross(v_bore, [0.0, 1.0, 0.0])
        q_err_full = np.concatenate([[0.0], normalize(axis)])
    else:
        # NOTE: goal × bore, not bore × goal
        cross = np.cross(v_goal_body, v_bore)
        q_err_full = normalize(np.concatenate([[1.0 + dot], cross]))

    q_err_vec = q_err_full[1:4] * np.sign(q_err_full[0])
    return q_err_vec