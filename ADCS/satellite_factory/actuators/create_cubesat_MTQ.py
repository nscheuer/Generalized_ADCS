__all__ = ['create_isis_magnetorquer_board', 'create_estcube1_magnetorquers']

import numpy as np
from typing import Optional, List, Sequence

from ADCS.satellite_hardware.actuators import MTQ
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.helpers.math_helpers import random_n_unit_vec

def create_isis_magnetorquer_board(axes: np.ndarray = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), bias: Sequence[Bias] | None = None, noise: Sequence[Noise] | None = None, estimate_bias: bool = False) -> List[MTQ]:
    # https://satsearch.co/products/isis-isis-magnetorquer-board-i-mtq
    if bias is None:
        e_bias = random_n_unit_vec(3)*np.random.uniform(0.01, 0.15)
        e_bias = np.zeros(3)
        std_bias = 0.000001*np.ones(3)
        std_bias = np.zeros(3)
        bias = [Bias(bias=e_bias[j], std_bias=std_bias[j]) for j in range(3)]
    if noise is None:
        e_noise = np.zeros(3)
        std_noise = 0.0001*np.ones(3)
        std_noise = np.zeros(3)
        noise = [Noise(noise=e_noise[j], std_noise=std_noise[j]) for j in range(3)]
    mtq_max = 0.2

    return [MTQ(axis=axes[j], max_torque=mtq_max, bias=bias[j], noise=noise[j], estimate_bias=estimate_bias) for j in range(3)]


def create_estcube1_magnetorquers(
    axes: np.ndarray = np.eye(3),
    bias: Sequence[Bias] | None = None,
    noise: Sequence[Noise] | None = None,
    estimate_bias: bool = False,
) -> List[MTQ]:
    r"""
    Create the nominal ESTCube-1 electromagnetic coil set.

    ESTCube-1 used three electromagnetic coils with nominal axes aligned to the
    spacecraft body axes and approximately 0.1 A m^2 nominal dipole authority per
    coil. Flight results reported substantial actuator nonidealities, including
    axis misalignment, positive/negative gain asymmetry, and residual magnetic
    moment; this factory captures the nominal dipole limit only unless explicit
    ``bias`` or ``noise`` models are supplied.

    Sources:

    * `ESTCube-1 magnetic actuator flight results
      <https://www.sciencedirect.com/science/article/pii/S0094576515302216>`__
    * `ESTCube-1 attitude determination flight results
      <https://ascelibrary.org/doi/10.1061/%28ASCE%29AS.1943-5525.0000504>`__
    """
    if bias is None:
        bias = [Bias() for _ in range(3)]
    if noise is None:
        noise = [Noise() for _ in range(3)]

    mtq_max = 0.1
    return [
        MTQ(axis=axes[j], max_torque=mtq_max, bias=bias[j], noise=noise[j], estimate_bias=estimate_bias)
        for j in range(3)
    ]
