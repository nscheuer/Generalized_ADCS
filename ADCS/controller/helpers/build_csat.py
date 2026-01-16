__all__ = ["build_cpp_satellite", "get_cpp_to_python_control_permutation", "reorder_controls_cpp_to_python"]

import numpy as np
from typing import List, Tuple

from ADCS.controller.helpers import PlannerSettings
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.actuators import Actuator, MTQ, RW

import trajectory_planner.build.tplaunch as tplaunch
import trajectory_planner.build.pysat as pysat


def get_cpp_to_python_control_permutation(actuators: List[Actuator]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the permutation indices to reorder controls from C++ ordering to Python ordering.

    C++ planner outputs controls in fixed order: [MTQs, RWs, magic]
    Python actuator list can have any order (e.g., [RW, RW, RW, MTQ, MTQ, MTQ])

    Parameters
    ----------
    actuators : List[Actuator]
        The Python actuator list in the order they appear in est_sat.actuators

    Returns
    -------
    cpp_to_py : np.ndarray
        Permutation array where cpp_to_py[i] gives the Python index for C++ control index i.
        Use: python_controls = cpp_controls[cpp_to_py] (for 1D) or
             python_controls = cpp_controls[cpp_to_py, :] (for 2D row-major)
    py_to_cpp : np.ndarray
        Inverse permutation. py_to_cpp[i] gives the C++ index for Python actuator index i.
    """
    # Find indices of each actuator type in Python ordering
    mtq_py_indices = [i for i, act in enumerate(actuators) if isinstance(act, MTQ)]
    rw_py_indices = [i for i, act in enumerate(actuators) if isinstance(act, RW)]
    # magic_py_indices would go here if needed

    # C++ ordering is: MTQs first, then RWs
    # cpp_index 0..n_mtq-1 -> MTQs
    # cpp_index n_mtq..n_mtq+n_rw-1 -> RWs

    n_mtq = len(mtq_py_indices)
    n_rw = len(rw_py_indices)
    n_total = n_mtq + n_rw

    # Build cpp_to_py: for each C++ index, what's the corresponding Python index?
    cpp_to_py = np.zeros(n_total, dtype=int)

    # First n_mtq C++ indices map to MTQ positions in Python
    for cpp_idx, py_idx in enumerate(mtq_py_indices):
        cpp_to_py[cpp_idx] = py_idx

    # Next n_rw C++ indices map to RW positions in Python
    for i, py_idx in enumerate(rw_py_indices):
        cpp_idx = n_mtq + i
        cpp_to_py[cpp_idx] = py_idx

    # Build inverse permutation
    py_to_cpp = np.zeros(n_total, dtype=int)
    for cpp_idx, py_idx in enumerate(cpp_to_py):
        py_to_cpp[py_idx] = cpp_idx

    return cpp_to_py, py_to_cpp


def reorder_controls_cpp_to_python(Uset: np.ndarray, actuators: List[Actuator]) -> np.ndarray:
    """
    Reorder control matrix from C++ ordering (MTQ, RW) to Python actuator ordering.

    Parameters
    ----------
    Uset : np.ndarray
        Control matrix from C++ planner. Shape is either (n_controls, n_timesteps) [col-major]
        or (n_timesteps, n_controls) [row-major].
    actuators : List[Actuator]
        Python actuator list defining the target ordering.

    Returns
    -------
    np.ndarray
        Reordered control matrix with same shape as input.
    """
    cpp_to_py, _ = get_cpp_to_python_control_permutation(actuators)
    n_ctrl = len(cpp_to_py)

    # Detect layout: if first dimension equals n_ctrl, it's col-major (n_ctrl x N)
    if Uset.shape[0] == n_ctrl:
        # Column-major: controls are rows
        return Uset[cpp_to_py, :]
    elif Uset.shape[1] == n_ctrl:
        # Row-major: controls are columns
        return Uset[:, cpp_to_py]
    else:
        raise ValueError(f"Uset shape {Uset.shape} doesn't match n_controls={n_ctrl}")


def reorder_gains_cpp_to_python(Kset: np.ndarray, actuators: List[Actuator]) -> np.ndarray:
    """
    Reorder gain matrix from C++ ordering to Python actuator ordering.

    The gain matrix K maps state errors to control adjustments: u = -K @ dx
    We need to reorder the rows of K to match Python control ordering.

    Parameters
    ----------
    Kset : np.ndarray
        Gain tensor from C++ planner. Expected shapes:
        - (n_timesteps, n_controls, n_states) for row-major time
        - (n_controls, n_states, n_timesteps) for col-major time
        - (n_controls * n_states, n_timesteps) for flattened
    actuators : List[Actuator]
        Python actuator list defining the target ordering.

    Returns
    -------
    np.ndarray
        Reordered gain tensor with same shape as input.
    """
    cpp_to_py, _ = get_cpp_to_python_control_permutation(actuators)
    n_ctrl = len(cpp_to_py)

    if Kset.ndim == 3:
        # 3D tensor
        if Kset.shape[0] >= Kset.shape[2]:
            # Time is first axis: (N, n_ctrl, n_states)
            return Kset[:, cpp_to_py, :]
        else:
            # Time is last axis: (n_ctrl, n_states, N)
            return Kset[cpp_to_py, :, :]
    elif Kset.ndim == 2:
        # Flattened: (n_ctrl * n_states, N) - need to unflatten, reorder, reflatten
        # This is complex; for now just return as-is and handle in Trajectory
        return Kset
    else:
        return Kset


def build_cpp_satellite(est_sat: EstimatedSatellite, planner_settings: PlannerSettings) -> pysat.Satellite:
    csat = pysat.Satellite()
    csat.change_Jcom(est_sat.J_0)

    for act in est_sat.actuators:
        add_actuator(act=act, csat=csat, planner_settings=planner_settings)

    if planner_settings.wmax > 0:
        csat.set_AV_constraint(planner_settings.wmax)
    if planner_settings.sun_limit_angle > 0:
        csat.add_sunpoint_constraint(planner_settings.camera_axis, planner_settings.sun_limit_angle, 0)
    if planner_settings.plan_for_gg:
        csat.add_gg_torq()
    if planner_settings.plan_for_aero:
        csat.add_aero_torq(planner_settings.drag_coeff,planner_settings.coeff_N)
    if planner_settings.plan_for_srp:
        csat.add_srp_torq(planner_settings.srp_coeff,planner_settings.coeff_N)
    if planner_settings.plan_for_resdipole:
        csat.add_resdipole_torq(planner_settings.res_dipole.reshape((3,1)))
    if planner_settings.plan_for_prop:
        csat.add_prop_torq(planner_settings.prop_torque.reshape((3,1)))
    if planner_settings.plan_for_gendist:
        csat.add_gendist_torq(planner_settings.gendist_torq.reshape((3,1)))

    return csat


def add_actuator(act: Actuator, csat: pysat.Satellite, planner_settings: PlannerSettings) -> None:
    if isinstance(act, MTQ):
        csat.add_MTQ(act.axis, act.u_max, planner_settings.mtq_control_weight)
    elif isinstance(act, RW):
        mult = getattr(planner_settings, 'RWh_max_mult', 0.8)
        cost_threshold = act.h_max * mult

        csat.add_RW(
            act.axis, 
            act.J, 
            act.u_max * planner_settings.control_limit_scale, 
            act.h_max * planner_settings.RWh_max_mult,
            planner_settings.rw_control_weight, 
            planner_settings.rw_AM_weight, 
            act.h_max * planner_settings.RWh_ok_mult,
            planner_settings.rw_stic_weight,
            act.h_max * planner_settings.RWh_stiction_mult
        )
    else:
        raise ValueError(f"Unknown actuator received: {act.__name__}")