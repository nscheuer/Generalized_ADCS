__all__ = [
    'create_Clydespace_3U_array',
    'create_hamamatsu_s3931_sun_sensors',
    'create_osram_sfh2430_sun_sensors',
]

import numpy as np
from typing import Optional, List

from ADCS.satellite_hardware.sensors import SunPair, SunSensor
from ADCS.satellite_hardware.errors import Noise, Bias
from ADCS.helpers.math_helpers import random_n_unit_vec


def _axes_from_az_el_deg(az_el_deg: np.ndarray) -> np.ndarray:
    angles = np.deg2rad(np.asarray(az_el_deg, dtype=float))
    az = angles[:, 0]
    el = angles[:, 1]
    return np.column_stack((np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)))

def create_Clydespace_3U_array(axis: np.ndarray = np.array([1, 0, 0]), bias: Bias | None = None, noise: Noise | None = None, estimate_bias: bool = False) -> List[SunPair]:
    r"""
    Create a Clyde Space 3U solar-array SunPair proxy.

    Default error model provenance:
        The default bias/noise values in this helper are package-level
        representative simulation values for BeaverCube-style examples. They
        are not currently tied to a published Clyde Space data sheet table or
        BeaverCube flight-calibration result. Pass explicit ``bias`` and
        ``noise`` objects when a traceable mission or vendor model is required.
    """
    sun_eff = 0.3
    if bias is None:
        e_bias = np.random.uniform(0,0.2)*sun_eff
        std_bias = 0.00001*sun_eff
        bias = Bias(bias=e_bias, std_bias=std_bias) 
    if noise is None:
        e_noise = 0
        std_noise = 0.001*sun_eff
        noise = Noise(noise=e_noise, std_noise=std_noise)

    return [SunPair(axis=axis, efficiency=(sun_eff, sun_eff), bias=bias, noise=noise, estimate_bias=estimate_bias)]


def create_hamamatsu_s3931_sun_sensors(
    axes: np.ndarray = np.repeat(np.vstack((np.eye(3), -np.eye(3))), 2, axis=0),
    bias: List[Bias] | None = None,
    noise: List[Noise] | None = None,
    estimate_bias: bool = False,
) -> List[SunSensor]:
    r"""
    Create the ESTCube-1 Hamamatsu S3931 one-dimensional Sun sensor channels.

    ESTCube-1 had six custom Sun sensor assemblies, one per spacecraft face.
    Each assembly used two Hamamatsu S3931 one-dimensional PSDs under
    perpendicular slits, so this factory returns twelve one-dimensional
    :class:`~ADCS.satellite_hardware.sensors.SunSensor` channels. The preflight
    simulator used angular noise of approximately 1.25 deg and initial angular
    bias uniformly distributed over +/-1 deg.

    Default error model provenance:
        ``bias`` approximates the published +/-1 deg initial angular bias as a
        scalar cosine-measurement offset of ``sin(1 deg)``. ``noise``
        approximates the published 1.25 deg Gaussian angular simulator noise as
        scalar cosine-measurement noise of ``sin(1.25 deg)``. The useful flight
        region was approximately +/-36.7 deg with incidence-angle-dependent
        flight covariance; the current sensor model does not encode that finite
        field-of-view or covariance schedule.

    Source:
        J. Slavinskis et al., "ESTCube-1 attitude determination system flight
        results," *Journal of Aerospace Engineering*, 2016.
        https://ascelibrary.org/doi/10.1061/%28ASCE%29AS.1943-5525.0000504
    """
    n_axes = len(axes)
    if bias is None:
        bias_bound = np.sin(1.0 * np.pi / 180.0)
        e_bias = np.random.uniform(-bias_bound, bias_bound, size=n_axes)
        bias = [Bias(bias=e_bias[j], std_bias=0.0, bounds=(-bias_bound, bias_bound)) for j in range(n_axes)]
    if noise is None:
        std_noise = np.sin(1.25 * np.pi / 180.0)
        noise = [Noise(noise=0.0, std_noise=std_noise) for _ in range(n_axes)]

    return [
        SunSensor(axis=axes[j], efficiency=1.0, bias=bias[j], noise=noise[j], estimate_bias=estimate_bias)
        for j in range(n_axes)
    ]


def create_osram_sfh2430_sun_sensors(
    az_el_deg: np.ndarray | None = None,
    bias: List[Bias] | None = None,
    noise: List[Noise] | None = None,
    estimate_bias: bool = False,
) -> List[SunSensor]:
    r"""
    Create the OSRAM SFH2430 coarse Sun-sensor photodiodes used on RAX.

    RAX-1 used 9 SFH2430 photodiodes; RAX-2 used 17. Pass the mission-specific
    azimuth/elevation table through ``az_el_deg``. The published RAX estimator
    simulations used 0.05 V additive white noise, with scale factors around
    2.5-4 V. This factory uses an efficiency of 3.0 V so the sensor output and
    noise are in the same approximate voltage units.

    Default error model provenance:
        ``noise`` defaults to the 0.05 V additive white-noise value from the
        RAX photodiode calibration/simulation tables. The default additive
        ``bias`` is zero because the RAX calibration estimated photodiode scale
        factor and mounting azimuth/elevation errors rather than publishing a
        single additive voltage-bias value. The cited calibration work reports
        that 0.05 V corresponds to roughly 1.0-5.7 deg angular uncertainty,
        depending on incidence angle and scale factor.

    Source:
        J. C. Springmann and J. W. Cutler, "On-orbit calibration of photodiodes
        for attitude determination," *Journal of Guidance, Control, and
        Dynamics*, 2014.
        https://deepblue.lib.umich.edu/bitstream/handle/2027.42/140645/1.g000175.pdf?sequence=1
    """
    if az_el_deg is None:
        az_el_deg = np.array([
            [0, 0], [180, 0], [90, 0], [270, 0], [0, 90],
            [0, 90], [0, 90], [0, -90], [0, -90],
        ])

    axes = _axes_from_az_el_deg(az_el_deg)
    n_axes = len(axes)
    if bias is None:
        bias = [Bias() for _ in range(n_axes)]
    if noise is None:
        noise = [Noise(noise=0.0, std_noise=0.05) for _ in range(n_axes)]

    return [
        SunSensor(axis=axes[j], efficiency=3.0, bias=bias[j], noise=noise[j], estimate_bias=estimate_bias)
        for j in range(n_axes)
    ]
