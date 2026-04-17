import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import ADCS as ADCS


def build_controller():
    real_sat = ADCS.satellite_factory.create_beavercube2_cubesat(estimated=False)
    return ADCS.controller.MTQ_w_RW_LP(
        est_sat=real_sat,
        p_gain=0.00005,
        d_gain=0.002,
        c_gain=0.001,
        h_target=np.array([0.0, 0.0, 0.0]),
    )


if __name__ == "__main__":
    host = os.getenv("ADCS_REMOTE_BIND_HOST", "0.0.0.0")
    port = int(os.getenv("ADCS_REMOTE_PORT", "5000"))
    controller = build_controller()
    ADCS.remote.serve_remote_controller(controller, host=host, port=port)