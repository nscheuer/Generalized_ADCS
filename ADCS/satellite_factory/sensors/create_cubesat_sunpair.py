__all__ = [
    'create_Clydespace_3U_array',
    'create_elmos_sun_sensors',
    'create_gnb_sun_sensors',
    'create_hamamatsu_s3931_sun_sensors',
    'create_nano_iss60_sun_sensors',
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


def create_gnb_sun_sensors(
    axes: np.ndarray = np.vstack((np.eye(3), -np.eye(3))),
    bias: List[Bias] | None = None,
    noise: List[Noise] | None = None,
    estimate_bias: bool = False,
) -> List[SunSensor]:
    r"""
    Create the final BRITE/GNB six-face dedicated Sun-sensor set.

    The public GNB summary gives one dedicated sensor per spacecraft face and
    accuracy/resolution specifications, but not statistical noise or bias
    values for the final dedicated sensors. Defaults are therefore ideal unless
    supplied by the caller.
    """
    n_axes = len(axes)
    if bias is None:
        bias = [Bias() for _ in range(n_axes)]
    if noise is None:
        noise = [Noise() for _ in range(n_axes)]

    return [
        SunSensor(axis=axes[j], efficiency=1.0, bias=bias[j], noise=noise[j], estimate_bias=estimate_bias)
        for j in range(n_axes)
    ]


def create_elmos_sun_sensors(
    axes: np.ndarray | None = None,
    bias: List[Bias] | None = None,
    noise: List[Noise] | None = None,
    estimate_bias: bool = False,
) -> List[SunSensor]:
    r"""
    Create the LightSail 2 coarse Sun-sensor set.

    LightSail 2 had four Elmos Sun sensors on deployable solar panels and one
    on the -Z spacecraft face. Exact deployed-panel boresight vectors are not
    encoded here, so panel sensors use a nominal +/-X/+Y layout and the -Z
    sensor uses the documented body vector. No source-backed noise or bias
    values are assigned by default.
    """
    if axes is None:
        axes = np.array([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ])
    n_axes = len(axes)
    if bias is None:
        bias = [Bias() for _ in range(n_axes)]
    if noise is None:
        noise = [Noise() for _ in range(n_axes)]

    return [
        SunSensor(axis=axes[j], efficiency=1.0, bias=bias[j], noise=noise[j], estimate_bias=estimate_bias)
        for j in range(n_axes)
    ]


def create_hamamatsu_s3931_sun_sensors(
    axes: np.ndarray = np.array([[1, 0, 0]]),
    bias: List[Bias] | None = None,
    noise: List[Noise] | None = None,
    estimate_bias: bool = False,
) -> List[SunSensor]:
    r"""
    Create one ESTCube-1 Hamamatsu S3931 one-dimensional Sun sensor channel.

    ESTCube-1 had six custom Sun sensor assemblies, one per spacecraft face.
    Each assembly used two Hamamatsu S3931 one-dimensional PSDs under
    perpendicular slits. This helper represents one such one-dimensional
    channel. The preflight simulator used angular noise of approximately
    1.25 deg and initial angular bias uniformly distributed over +/-1 deg.

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
    Create OSRAM SFH2430 coarse Sun-sensor photodiode channels used on RAX.

    By default this helper creates one photodiode aligned with ``+X``. RAX-1
    used 9 SFH2430 photodiodes and RAX-2 used 17; pass the mission-specific
    azimuth/elevation table through ``az_el_deg`` to create those arrays. The
    published RAX estimator simulations used 0.05 V additive white noise, with
    scale factors around 2.5-4 V. This factory uses an efficiency of 3.0 V so
    the sensor output and noise are in the same approximate voltage units.

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
        az_el_deg = np.array([[0, 0]])

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


def create_nano_iss60_sun_sensors(
    axes: np.ndarray = np.array([[1, 0, 0]]),
    bias: List[Bias] | None = None,
    noise: List[Noise] | None = None,
    estimate_bias: bool = False,
) -> List[SunSensor]:
    r"""
    Create one Solar MEMS NANO-ISS60 Sun sensor proxy used on MOVE-II.

    A NANO-ISS60 is a two-axis analog Sun position sensor. It is represented
    here by one body-normal
    :class:`~ADCS.satellite_hardware.sensors.SunSensor` proxy because the
    current package has scalar Sun-sensor channels rather than a native
    two-axis NANO-ISS60 model.

    Default error model provenance:
        ``noise`` approximates the MOVE-II HIL angular Sun-vector noise
        ``sigma = 0.06 deg = 1.047e-3 rad`` as scalar cosine-measurement noise
        ``sin(0.06 deg)``. No nominal fixed Sun-sensor bias was given in the
        accessible tables, so ``bias`` defaults to zero. The source documents a
        10 deg Sun-sensor bias sensitivity case, but that is not the nominal
        default.

    Source:
        `Hardware-in-the-Loop Verification of the Distributed,
        Magnetorquer-Based ADCS of the CubeSat MOVE-II
        <https://mediatum.ub.tum.de/doc/1483411/document.pdf>`__
    """
    n_axes = len(axes)
    if bias is None:
        bias = [Bias() for _ in range(n_axes)]
    if noise is None:
        std_noise = np.sin(0.06 * np.pi / 180.0)
        noise = [Noise(noise=0.0, std_noise=std_noise) for _ in range(n_axes)]

    return [
        SunSensor(axis=axes[j], efficiency=1.0, bias=bias[j], noise=noise[j], estimate_bias=estimate_bias)
        for j in range(n_axes)
    ]
