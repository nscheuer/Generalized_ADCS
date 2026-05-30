import warnings

import numpy as np
from scipy.optimize import lsq_linear

from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.satellite import Satellite
from ADCS.satellite_hardware.sensors import MTM
from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import skewsym


def make_satellite(
    *,
    include_rw: bool = True,
    rw_axes: list[np.ndarray] | None = None,
    mtq_max_torque: float = 0.4,
    rw_max_torque: float = 7.0e-3,
    rw_h: float | list[float] | np.ndarray = 5.0e-3,
) -> Satellite:
    mtqs = [MTQ(axis=axis, max_torque=mtq_max_torque) for axis in MathConstants.unitvecs]
    actuators = list(mtqs)

    if include_rw:
        axes = [MathConstants.unitvecs[0]] if rw_axes is None else rw_axes
        rw_h_array = (
            np.repeat(float(rw_h), len(axes))
            if np.isscalar(rw_h)
            else np.asarray(rw_h, dtype=float)
        )
        for index, axis in enumerate(axes):
            actuators.append(
                RW(
                    axis=axis,
                    max_torque=rw_max_torque,
                    J=1.0e-3,
                    h=rw_h_array[index],
                    h_max=16.2e-3,
                )
            )

    sensors = [MTM(axis=axis) for axis in MathConstants.unitvecs]
    return Satellite(
        mass=1.2,
        J_0=np.diagflat([0.022, 0.022, 0.004]),
        actuators=actuators,
        sensors=sensors,
        boresight=np.array([0.0, 0.0, 1.0]),
    )


def make_orbital_state(*, b_body: np.ndarray | None = None) -> Orbital_State:
    return Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 8.0, 0.0]),
        B=np.array([2.0e-5, -3.0e-5, 4.0e-5]) if b_body is None else np.asarray(b_body, dtype=float),
        S=np.array([1.0e5, 0.0, 0.0]),
        rho=5.0e-12,
    )


def make_controller(controller_cls, satellite: Satellite, /, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return controller_cls(satellite, *args, **kwargs)


def achieved_torque(
    satellite: Satellite,
    u_rw: np.ndarray,
    u_mtq: np.ndarray,
    b_body: np.ndarray,
) -> np.ndarray:
    rws = [actuator for actuator in satellite.actuators if isinstance(actuator, RW)]
    mtqs = [actuator for actuator in satellite.actuators if isinstance(actuator, MTQ)]

    tau_rw = (
        sum(np.asarray(rw.axis, dtype=float) * u_rw[index] for index, rw in enumerate(rws))
        if rws
        else np.zeros(3)
    )
    tau_mtq = (
        sum(
            np.cross(np.asarray(mtq.axis, dtype=float) * u_mtq[index], b_body)
            for index, mtq in enumerate(mtqs)
        )
        if mtqs
        else np.zeros(3)
    )
    return tau_rw + tau_mtq


def combined_actuation_matrix(satellite: Satellite, b_body: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rws = [actuator for actuator in satellite.actuators if isinstance(actuator, RW)]
    mtqs = [actuator for actuator in satellite.actuators if isinstance(actuator, MTQ)]

    if rws:
        a_rw = np.column_stack([np.asarray(rw.axis, dtype=float).reshape(3,) for rw in rws])
        rw_limits = np.array([rw.u_max for rw in rws], dtype=float)
    else:
        a_rw = np.zeros((3, 0))
        rw_limits = np.zeros(0, dtype=float)

    if mtqs:
        a_mtq_axes = np.column_stack([np.asarray(mtq.axis, dtype=float).reshape(3,) for mtq in mtqs])
        a_mtq = -skewsym(np.asarray(b_body, dtype=float)) @ a_mtq_axes
        mtq_limits = np.array([mtq.u_max for mtq in mtqs], dtype=float)
    else:
        a_mtq = np.zeros((3, 0))
        mtq_limits = np.zeros(0, dtype=float)

    return np.hstack([a_rw, a_mtq]), rw_limits, mtq_limits


def plain_bounded_lsq(satellite: Satellite, tau_des: np.ndarray, b_body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a_total, rw_limits, mtq_limits = combined_actuation_matrix(satellite, b_body)
    lower = np.concatenate([-rw_limits, -mtq_limits])
    upper = np.concatenate([rw_limits, mtq_limits])
    result = lsq_linear(a_total, np.asarray(tau_des, dtype=float), bounds=(lower, upper), method="trf")
    return result.x, a_total @ result.x


def assert_command_bounds(u_rw: np.ndarray, u_mtq: np.ndarray, rw_limits: np.ndarray, mtq_limits: np.ndarray) -> None:
    np.testing.assert_array_less(np.abs(u_rw), rw_limits + 1.0e-12)
    np.testing.assert_array_less(np.abs(u_mtq), mtq_limits + 1.0e-12)
