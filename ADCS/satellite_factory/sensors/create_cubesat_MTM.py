__all__ = [
    'create_gnb_magnetometer',
    'create_isis_magnetometer',
    'create_adis16405_magnetometers',
    'create_bmx055_magnetometers',
    'create_honeywell_lightsail2_magnetometers',
    'create_hmc5883l_magnetometers',
    'create_micromag3_magnetometers',
]

import numpy as np
from typing import Optional, List

from ADCS.satellite_hardware.sensors import MTM 
from ADCS.satellite_hardware.errors import Noise, Bias
from ADCS.helpers.math_helpers import random_n_unit_vec

def create_isis_magnetometer(axes: np.ndarray = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), bias: Bias | None = None, noise: Noise | None = None, estimate_bias: bool = False) -> List[MTM]:
    r"""
    Create an ISIS magnetometer triad.

    Default error model provenance:
        The hardware identity is linked to the ISIS magnetorquer-board product
        listing below, but the default bias/noise values in this helper are
        package-level representative simulation values rather than traceable
        vendor or flight-calibration values. Pass explicit ``bias`` and
        ``noise`` objects when a source-backed model is required.

    Source for hardware identity:
        https://satsearch.co/products/isis-isis-magnetorquer-board-i-mtq
    """
    if bias is None:
        e_bias = random_n_unit_vec(3)*np.random.uniform(1e-9,1e-7)
        std_bias = 1e-9*np.ones(3)
        bias = [Bias(bias=e_bias[j], std_bias=std_bias[j]) for j in range(3)]
    if noise is None:
        e_noise = np.zeros(3)
        std_noise = 3*1e-7*np.ones(3)
        noise = [Noise(noise=e_noise[j], std_noise=std_noise[j]) for j in range(3)]

    return [MTM(axis=axes[j], bias=bias[j], noise=noise[j], estimate_bias=estimate_bias) for j in range(3)]


def create_gnb_magnetometer(
    axes: np.ndarray = np.eye(3),
    bias: List[Bias] | None = None,
    noise: List[Noise] | None = None,
    estimate_bias: bool = False,
) -> List[MTM]:
    r"""
    Create the nominal BRITE/GNB three-axis magnetometer.

    The BRITE ADCS design gives magnetometer noise ``2.0e-7 T`` and worst-case
    bias ``4.0e-6 T``. The factory represents the latter as a bounded
    per-axis bias uncertainty because no fixed flight bias vector was found.
    """
    n_axes = len(axes)
    if bias is None:
        bias_bound = 4.0e-6
        bias = [Bias(bias=0.0, std_bias=0.0, bounds=(-bias_bound, bias_bound)) for _ in range(n_axes)]
    if noise is None:
        noise = [Noise(noise=0.0, std_noise=2.0e-7) for _ in range(n_axes)]

    return [MTM(axis=axes[j], bias=bias[j], noise=noise[j], estimate_bias=estimate_bias) for j in range(n_axes)]


def create_honeywell_lightsail2_magnetometers(
    axes: np.ndarray | None = None,
    bias: List[Bias] | None = None,
    noise: List[Noise] | None = None,
    estimate_bias: bool = False,
) -> List[MTM]:
    r"""
    Create the two LightSail 2 three-axis Honeywell magnetometers.

    LightSail 2 had one 3-axis unit on the +X solar panel and one on the +Y
    solar panel. Exact sensor-to-body rotation matrices are not encoded here,
    so each unit is represented as an orthogonal triad in body axes. The
    published 1-sigma noise is ``0.2 uT``; no fixed bias is assigned.
    """
    if axes is None:
        axes = np.vstack((np.eye(3), np.eye(3)))
    n_axes = len(axes)
    if bias is None:
        bias = [Bias() for _ in range(n_axes)]
    if noise is None:
        noise = [Noise(noise=0.0, std_noise=0.2e-6) for _ in range(n_axes)]

    return [MTM(axis=axes[j], bias=bias[j], noise=noise[j], estimate_bias=estimate_bias) for j in range(n_axes)]


def create_adis16405_magnetometers(
    axes: np.ndarray = np.eye(3),
    bias: List[Bias] | None = None,
    noise: List[Noise] | None = None,
    estimate_bias: bool = False,
) -> List[MTM]:
    r"""
    Create the ADIS16405 integrated magnetometer channels used on RAX.

    Default error model provenance:
        ``bias`` defaults to a Gaussian initial bias with 400 nT one-sigma
        component error. ``noise`` defaults to 125 nT output-noise RMS. These
        are ADIS16405 component characterization values reported for the RAX
        attitude-determination system, not a published fixed RAX flight bias
        vector.

    Source:
        J. C. Springmann, *Satellite Attitude Determination with Low-Cost
        Sensors*, Ph.D. dissertation, University of Michigan, 2013.
        https://deepblue.lib.umich.edu/bitstream/handle/2027.42/102312/jspringm_1.pdf?sequence=1
    """
    n_axes = len(axes)
    if bias is None:
        bias_std = 400e-9
        e_bias = np.random.normal(0.0, bias_std, size=n_axes)
        bias = [Bias(bias=e_bias[j], std_bias=0.0) for j in range(n_axes)]
    if noise is None:
        std_noise = 125e-9
        noise = [Noise(noise=0.0, std_noise=std_noise) for _ in range(n_axes)]

    return [MTM(axis=axes[j], bias=bias[j], noise=noise[j], estimate_bias=estimate_bias) for j in range(n_axes)]


def create_micromag3_magnetometers(
    axes: np.ndarray = np.eye(3),
    bias: List[Bias] | None = None,
    noise: List[Noise] | None = None,
    estimate_bias: bool = False,
) -> List[MTM]:
    r"""
    Create the PNI Sensor Corporation MicroMag3 magnetometer channels used on RAX.

    Default error model provenance:
        ``noise`` defaults to the 100 nT Gaussian magnetometer noise used in
        the RAX sensor-error simulations. The default constant ``bias`` is zero
        because the RAX flight calibration treated the embedded MicroMag3 error
        as scale-factor, nonorthogonality, constant-bias, and
        spacecraft-current-dependent terms, not as a single published fixed
        mission bias vector.

    Source:
        J. C. Springmann, *Satellite Attitude Determination with Low-Cost
        Sensors*, Ph.D. dissertation, University of Michigan, 2013.
        https://deepblue.lib.umich.edu/bitstream/handle/2027.42/102312/jspringm_1.pdf?sequence=1
    """
    n_axes = len(axes)
    if bias is None:
        bias = [Bias() for _ in range(n_axes)]
    if noise is None:
        std_noise = 100e-9
        noise = [Noise(noise=0.0, std_noise=std_noise) for _ in range(n_axes)]

    return [MTM(axis=axes[j], bias=bias[j], noise=noise[j], estimate_bias=estimate_bias) for j in range(n_axes)]


def create_bmx055_magnetometers(
    axes: np.ndarray = np.eye(3),
    bias: List[Bias] | None = None,
    noise: List[Noise] | None = None,
    estimate_bias: bool = False,
) -> List[MTM]:
    r"""
    Create one Bosch Sensortec BMX055 magnetometer triad used on MOVE-II.

    The BMX055 magnetometer is represented as three one-dimensional
    :class:`~ADCS.satellite_hardware.sensors.MTM` channels.

    Default error model provenance:
        ``bias`` defaults to the MOVE-II HIL simulation vector
        ``[2.0e-6, -3.0e-6, 3.0e-6] T`` for one BMX055 triad. ``noise``
        defaults to additive Gaussian white noise with ``sigma = 5.0e-7 T``.
        The source also models scale factor,
        misalignment, nonorthogonality, quantization, time sampling, and
        low-pass filtering; those terms are not represented directly here.

    Source:
        `Hardware-in-the-Loop Verification of the Distributed,
        Magnetorquer-Based ADCS of the CubeSat MOVE-II
        <https://mediatum.ub.tum.de/doc/1483411/document.pdf>`__
    """
    n_axes = len(axes)
    if bias is None:
        triad_bias = np.array([2.0e-6, -3.0e-6, 3.0e-6])
        bias = [Bias(bias=triad_bias[j], std_bias=0.0) for j in range(n_axes)]
    if noise is None:
        noise = [Noise(noise=0.0, std_noise=5.0e-7) for _ in range(n_axes)]

    return [MTM(axis=axes[j], bias=bias[j], noise=noise[j], estimate_bias=estimate_bias) for j in range(n_axes)]


def create_hmc5883l_magnetometers(
    axes: np.ndarray = np.eye(3),
    bias: List[Bias] | None = None,
    noise: List[Noise] | None = None,
    estimate_bias: bool = False,
) -> List[MTM]:
    r"""
    Create one ESTCube-1 Honeywell HMC5883L magnetometer triad.

    The HMC5883L magnetometer is represented as three one-dimensional
    :class:`~ADCS.satellite_hardware.sensors.MTM` channels.

    Default error model provenance:
        ``bias`` defaults to a uniform initial bias over +/-2400 nT. The
        ESTCube-1 preflight simulator modeled magnetic-field direction noise as
        Gaussian 1.6 deg. Since this package's
        :class:`~ADCS.satellite_hardware.sensors.MTM` model uses scalar magnetic
        field measurements, ``noise`` approximates that 1.6 deg direction error
        as a transverse component error for a representative 50 microtesla
        field. Ground characterization also reported 0.8 deg statistical
        magnetic-direction uncertainty, but the simulator value is the default.

    Source:
        J. Slavinskis et al., "ESTCube-1 attitude determination system flight
        results," *Journal of Aerospace Engineering*, 2016.
        https://ascelibrary.org/doi/10.1061/%28ASCE%29AS.1943-5525.0000504
    """
    n_axes = len(axes)
    if bias is None:
        bias_bound = 2400e-9
        e_bias = np.random.uniform(-bias_bound, bias_bound, size=n_axes)
        bias = [Bias(bias=e_bias[j], std_bias=0.0, bounds=(-bias_bound, bias_bound)) for j in range(n_axes)]
    if noise is None:
        representative_field = 50e-6
        std_noise = representative_field * np.sin(1.6 * np.pi / 180.0)
        noise = [Noise(noise=0.0, std_noise=std_noise) for _ in range(n_axes)]

    return [MTM(axis=axes[j], bias=bias[j], noise=noise[j], estimate_bias=estimate_bias) for j in range(n_axes)]
