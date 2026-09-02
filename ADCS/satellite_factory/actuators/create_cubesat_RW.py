__all__ = [
    'create_cubewheel_smallplus_rw',
    'create_sfl_reaction_wheels',
    'create_sinclair_interplanetary_momentum_wheel',
]

import numpy as np
from typing import Optional, List

from ADCS.satellite_hardware.actuators import RW
from ADCS.satellite_hardware.errors import Bias, Noise

def create_cubewheel_smallplus_rw(axis: np.ndarray = np.array([1, 0, 0]), bias: Bias | None = None, noise: Noise | None = None, h_meas_noise: Noise | None = None, estimate_bias: bool = False) -> RW:
    # As used on BeaverCube 2
    if bias is None:
        e_bias = np.random.normal(0.0, 3e-7)
        std_bias = 1e-9
        bias = Bias(bias=e_bias, std_bias=std_bias)
    if noise is None:
        e_noise = 0.0
        std_noise = 1e-6
        noise = Noise(noise=e_noise, std_noise=std_noise)
    if h_meas_noise is None:
        e_h_meas_noise = 0.0
        std_h_meas_noise = 1e-6
        h_meas_noise = Noise(noise=e_h_meas_noise, std_noise=std_h_meas_noise)
        
    max_torque = 0.0023
    J = 5.7e-6
    h = 0.0
    h_max = 0.0036

    return RW(axis=axis, max_torque=max_torque, J=J, h=h, h_max=h_max, bias=bias, noise=noise, h_meas_noise=h_meas_noise, estimate_bias=estimate_bias)


def create_sfl_reaction_wheels(
    axes: np.ndarray = np.eye(3),
    bias: List[Bias] | None = None,
    noise: List[Noise] | None = None,
    h_meas_noise: List[Noise] | None = None,
    estimate_bias: bool = False,
) -> List[RW]:
    r"""
    Create the nominal three-wheel BRITE/GNB reaction-wheel assembly.

    The BRITE/GNB hardware summary gives an orthogonal three-wheel assembly
    with per-wheel rotor inertia ``5.12e-5 kg m^2``, momentum capacity
    ``0.030 N m s``, and maximum torque ``0.002 N m``. No source-backed torque
    noise or torque bias values were found, so the default error models are
    zero unless supplied by the caller.
    """
    n_axes = len(axes)
    if bias is None:
        bias = [Bias() for _ in range(n_axes)]
    if noise is None:
        noise = [Noise() for _ in range(n_axes)]
    if h_meas_noise is None:
        h_meas_noise = [Noise() for _ in range(n_axes)]

    return [
        RW(
            axis=axes[j],
            max_torque=0.002,
            J=5.12e-5,
            h=0.0,
            h_max=0.030,
            bias=bias[j],
            noise=noise[j],
            h_meas_noise=h_meas_noise[j],
            estimate_bias=estimate_bias,
        )
        for j in range(n_axes)
    ]


def create_sinclair_interplanetary_momentum_wheel(
    axis: np.ndarray = np.array([0, 1, 0]),
    bias: Bias | None = None,
    noise: Noise | None = None,
    h_meas_noise: Noise | None = None,
    estimate_bias: bool = False,
) -> RW:
    r"""
    Create the LightSail 2 single +Y momentum wheel.

    LightSail 2 used one Sinclair Interplanetary momentum wheel about +Y with
    maximum torque ``5.0e-3 N m`` and maximum angular momentum
    ``0.06 N m s`` at ``5920 rpm``. The public source does not provide a rotor
    inertia directly, so ``J`` is inferred from ``H = J omega`` as
    ``0.06 / (5920 * 2 pi / 60) = 9.68e-5 kg m^2``. Torque noise and torque
    bias remain zero unless supplied by the caller.
    """
    if bias is None:
        bias = Bias()
    if noise is None:
        noise = Noise()
    if h_meas_noise is None:
        h_meas_noise = Noise()

    return RW(
        axis=axis,
        max_torque=5.0e-3,
        J=0.06 / (5920.0 * 2.0 * np.pi / 60.0),
        h=0.0,
        h_max=0.06,
        bias=bias,
        noise=noise,
        h_meas_noise=h_meas_noise,
        estimate_bias=estimate_bias,
    )
