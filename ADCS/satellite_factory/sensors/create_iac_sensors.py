"""Sensor suite for the IAC-26 "One Wheel Is Enough" reference bus.

The campaign runs at a **higher sensor grade** than the companion papers on purpose: the
frontier the paper draws should be limited by *actuation*, not by a 3U-grade attitude
solution, and the Section-IV dipole-cancellation result is meaningless without a real
estimate to cancel against.

Spec (campaign §3) and how each line maps onto the repo's error models:

============  ========================================  ==========================================
Element       Spec                                      Model
============  ========================================  ==========================================
Star tracker  10" cross-boresight, 60" about boresight   anisotropic small-rotation perturbation
              4 Hz, rate limit ~2 deg/s                  ``max_rate``
              30 deg sun, 25 deg Earth-limb exclusion    ``sun_exclusion`` / ``earth_limb_exclusion``
Gyro          ARW 1 deg/sqrt(hr), bias instab. 5 deg/hr  white ``Noise`` + random-walk ``Bias``
Magnetometer  100 nT (1 sigma) post-calibration          white ``Noise``
============  ========================================  ==========================================

**The two noise conversions are stated here because the paper has to state them once.**

*Gyro angle random walk.* ARW is an integrated-angle spec; the repo adds white noise to each
rate sample. For a sample interval ``dt`` the equivalent per-sample rate sigma is
``sigma = ARW / sqrt(dt)``. At ARW = 1 deg/sqrt(hr) = (1/60) deg/sqrt(s) and ``dt = 1`` s
this is 0.0167 deg/s = 2.91e-4 rad/s. Pass the campaign's control interval as ``dt``.

*Gyro bias instability.* Allan bias instability is a floor on a log-log Allan plot; the repo
models bias as a per-step random walk, and the two are not the same statistic. The mapping
used here is deliberately simple and stated rather than hidden: the **initial** bias is drawn
with 1-sigma equal to the quoted instability (5 deg/hr = 2.42e-5 rad/s), and the random-walk
step is sized so the bias wanders by about that much over one orbit,
``std_bias = BI / sqrt(T_orbit / dt)``. That reproduces the right order of magnitude of
in-run drift over the horizons this campaign uses without over-claiming an Allan-variance
fit the model does not support.
"""

from __future__ import annotations

__all__ = [
    "create_iac_gyro",
    "create_iac_magnetometer",
    "create_iac_star_tracker",
    "IAC_SENSOR_SPEC",
]

from typing import List, Optional

import numpy as np

from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.satellite_hardware.sensors import MTM, Gyro
from ADCS.satellite_hardware.sensors.star_tracker_quaternion import StarTrackerQuaternion
from ADCS.helpers.math_helpers import random_n_unit_vec

_DEG = np.pi / 180.0
_ARCSEC = _DEG / 3600.0
_HR = 3600.0

#: Campaign §3 sensing table, in SI, in one place.
IAC_SENSOR_SPEC = {
    "mtm_sigma_T": 100e-9,                     # 100 nT, 1 sigma
    "gyro_arw_rad_per_sqrt_s": (1.0 * _DEG) / np.sqrt(_HR),   # 1 deg/sqrt(hr)
    "gyro_bias_instab_rad_per_s": 5.0 * _DEG / _HR,           # 5 deg/hr
    "st_sigma_cross_rad": 10.0 * _ARCSEC,
    "st_sigma_roll_rad": 60.0 * _ARCSEC,
    "st_sample_time_s": 0.25,                  # 4 Hz
    "st_max_rate_rad_per_s": 2.0 * _DEG,       # ~2 deg/s
    "st_sun_exclusion_rad": 30.0 * _DEG,
    "st_earth_limb_exclusion_rad": 25.0 * _DEG,
    "st_fov_rad": 20.0 * _DEG,
}

#: Reference orbital period [s] used for the bias random-walk sizing (400 km circular).
_T_ORBIT_REF = 5553.6


def create_iac_magnetometer(
    axes: np.ndarray = None,
    estimate_bias: bool = False,
    sigma_T: Optional[float] = None,
) -> List[MTM]:
    """Three-axis magnetometer at the campaign's 100 nT (1 sigma) post-calibration grade.

    The default repo part (``create_isis_magnetometer``) is 300 nT, which would make the
    magnetometer -- not the field model -- the limit on the residual-dipole estimate.
    """
    if axes is None:
        axes = np.eye(3)
    sigma = IAC_SENSOR_SPEC["mtm_sigma_T"] if sigma_T is None else float(sigma_T)
    # Post-calibration residual bias: small compared with the noise, random direction.
    e_bias = random_n_unit_vec(3) * np.random.uniform(1e-9, 1e-8)
    bias = [Bias(bias=e_bias[j], std_bias=1e-10) for j in range(3)]
    noise = [Noise(noise=0.0, std_noise=sigma) for _ in range(3)]
    return [
        MTM(axis=axes[j], bias=bias[j], noise=noise[j], estimate_bias=estimate_bias)
        for j in range(3)
    ]


def create_iac_gyro(
    axes: np.ndarray = None,
    estimate_bias: bool = True,
    dt: float = 1.0,
    T_orbit: float = _T_ORBIT_REF,
) -> List[Gyro]:
    """Three-axis MEMS gyro: ARW 1 deg/sqrt(hr), bias instability 5 deg/hr.

    :param dt: Sample interval [s]. Sets the white-noise sigma via ``ARW / sqrt(dt)``.
    :param T_orbit: Horizon [s] over which the bias random walk is sized to reach the
        quoted bias instability. See the module docstring.

    ``estimate_bias`` defaults to ``True`` here (unlike the repo's stock factories): the
    campaign's filter carries gyro bias in the augmented state, and a MEMS-class bias left
    unestimated would dominate the attitude solution.
    """
    if axes is None:
        axes = np.eye(3)
    arw = IAC_SENSOR_SPEC["gyro_arw_rad_per_sqrt_s"]
    bi = IAC_SENSOR_SPEC["gyro_bias_instab_rad_per_s"]

    std_noise = arw / np.sqrt(float(dt))
    n_steps = max(float(T_orbit) / float(dt), 1.0)
    std_bias = bi / np.sqrt(n_steps)

    e_bias = random_n_unit_vec(3) * abs(np.random.normal(0.0, bi))
    bias = [Bias(bias=e_bias[j], std_bias=std_bias) for j in range(3)]
    noise = [Noise(noise=0.0, std_noise=std_noise) for _ in range(3)]
    return [
        Gyro(axis=axes[j], bias=bias[j], noise=noise[j], estimate_bias=estimate_bias)
        for j in range(3)
    ]


def create_iac_star_tracker(
    boresight: np.ndarray = None,
    permissive: bool = False,
) -> StarTrackerQuaternion:
    """Campaign-grade quaternion star tracker.

    10" cross-boresight / 60" about boresight, 4 Hz, ~2 deg/s rate limit, 30 deg sun and
    25 deg Earth-limb keep-outs. Availability is not continuous by design -- that is the
    point. Log ``tracker.available`` per step and report the per-trial mean; the exclusion
    angles and the rate limit interact directly with the agility boundary.

    Note the roll term lands on the boresight, which on this bus is **also the wheel axis**,
    so the 6x anisotropy is not a cosmetic detail here.

    :param permissive: Disable all dropout mechanisms (no exclusions, no rate limit,
        ``min_stars=0``). Only for debugging filters -- a NaN measurement poisons UKF sigma
        points, so an unstable filter is easier to diagnose with dropouts off. Never use
        for a production run.
    """
    if boresight is None:
        boresight = np.array([0.0, 0.0, 1.0])
    s = IAC_SENSOR_SPEC

    # The estimator needs a measurement-noise covariance shaped like the *output* (4, a
    # quaternion), even though the anisotropic model does not draw its perturbation from it.
    # A quaternion component is a half-angle, so sigma_q ~ sigma_angle / 2; the isotropic
    # value the filter is given is the RMS over the three body axes, which neither
    # understates the roll term (6x worse than cross) nor pretends the filter knows the
    # anisotropy. Slightly conservative on the two cross axes, slightly optimistic on roll.
    sigma_rms = np.sqrt((2.0 * s["st_sigma_cross_rad"] ** 2
                         + s["st_sigma_roll_rad"] ** 2) / 3.0)
    st_noise = Noise(noise=np.zeros(4), std_noise=np.full(4, sigma_rms / 2.0))
    if permissive:
        return StarTrackerQuaternion(
            sample_time=s["st_sample_time_s"],
            boresight=boresight,
            noise=st_noise,
            fov=np.deg2rad(179.0),
            sun_exclusion=0.0,
            min_stars=0,
            earth_limb_exclusion=0.0,
            max_rate=None,
            sigma_cross=s["st_sigma_cross_rad"],
            sigma_roll=s["st_sigma_roll_rad"],
        )
    # min_stars=0 is deliberate, and it is the honest choice here.
    #
    # ``StarCatalog`` ships **30** stars. A 20-degree FOV subtends 0.76% of the sky, so the
    # expected count inside it is 0.23 -- requiring 2 makes the tracker unavailable ~98% of
    # the time. That is an artifact of a toy catalog, not physics: a real tracker works from
    # a catalog of 10^4-10^5 stars and is star-starved essentially never. Gating on this
    # catalog would put a fictitious 2% availability into the paper.
    #
    # So star density is not modelled as a constraint; the availability drivers that *are*
    # physical are kept and are the ones the campaign reports: the sun keep-out, the
    # Earth-limb keep-out, and the slew-rate limit. The FOV is opened up for the same
    # reason -- it only feeds star selection, not the keep-outs.
    return StarTrackerQuaternion(
        sample_time=s["st_sample_time_s"],
        boresight=boresight,
        noise=st_noise,
        fov=np.deg2rad(179.0),
        sun_exclusion=s["st_sun_exclusion_rad"],
        min_stars=0,
        earth_limb_exclusion=s["st_earth_limb_exclusion_rad"],
        max_rate=s["st_max_rate_rad_per_s"],
        sigma_cross=s["st_sigma_cross_rad"],
        sigma_roll=s["st_sigma_roll_rad"],
    )
