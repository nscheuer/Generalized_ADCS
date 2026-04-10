from __future__ import annotations

__all__ = ["build_cpp_satellite", "get_cpp_to_python_control_permutation", "reorder_controls_cpp_to_python"]

import numpy as np
from typing import List, Tuple, TYPE_CHECKING
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ADCS.controller.helpers import PlannerSettings
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.satellite_hardware.actuators import Actuator, MTQ, RW
from ADCS.controller.helpers.optional_dependencies import get_trajectory_planner_modules


def get_cpp_to_python_control_permutation(actuators: List[Actuator]) -> Tuple[NDArray[np.intp], NDArray[np.intp]]:
    r"""
    Compute the permutation between C++ planner control ordering and Python actuator ordering.

    The C++ trajectory planner outputs control vectors in a fixed, type-based order:

    .. math::

        \mathbf{u}_{\text{C++}} =
        \begin{bmatrix}
        \mathbf{u}_{\text{MTQ}} \\
        \mathbf{u}_{\text{RW}}
        \end{bmatrix}

    where all magnetorquers (MTQs) are listed first, followed by all reaction wheels (RWs).
    In contrast, the Python-side actuator list stored in
    :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
    may have an arbitrary ordering.

    This function computes permutation indices that allow reordering control vectors
    and matrices between these two conventions.

    Let :math:`\pi` be a permutation such that

    .. math::

        \mathbf{u}_{\text{Python}} = \mathbf{u}_{\text{C++}}[\pi]

    and :math:`\pi^{-1}` its inverse.

    :param actuators:
        List of Python actuator objects in the order they appear in
        :attr:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite.actuators`.

    :type actuators:
        List[:class:`~ADCS.satellite_hardware.actuators.Actuator`]

    :return:
        A tuple ``(cpp_to_py, py_to_cpp)`` where ``cpp_to_py`` maps C++ control indices
        to Python indices, and ``py_to_cpp`` is the inverse permutation.

    :rtype:
        Tuple[numpy.ndarray, numpy.ndarray]

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


def reorder_controls_cpp_to_python(Uset: NDArray[np.float64], actuators: List[Actuator]) -> NDArray[np.float64]:
    r"""
    Reorder a control matrix from C++ planner ordering to Python actuator ordering.

    The C++ planner outputs control trajectories using the ordering described in
    :func:`~ADCS.controller.helpers.get_cpp_to_python_control_permutation`.
    This function applies the appropriate permutation to match the actuator order
    expected by the Python simulation and control stack.

    Two matrix layouts are supported:

    - Column-major controls:

      .. math::

          \mathbf{U} \in \mathbb{R}^{n_u \times N}

      where each row corresponds to one control input over time.

    - Row-major controls:

      .. math::

          \mathbf{U} \in \mathbb{R}^{N \times n_u}

      where each column corresponds to one control input.

    The output matrix preserves the original layout and dimensions.

    :param Uset:
        Control matrix produced by the C++ planner.

    :type Uset:
        numpy.ndarray

    :param actuators:
        Python actuator list defining the desired control ordering.

    :type actuators:
        List[:class:`~ADCS.satellite_hardware.actuators.Actuator`]

    :return:
        Control matrix reordered to match Python actuator ordering.

    :rtype:
        numpy.ndarray

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


def reorder_gains_cpp_to_python(Kset: NDArray[np.float64], actuators: List[Actuator]) -> NDArray[np.float64]:
    r"""
    Reorder a feedback gain tensor from C++ planner ordering to Python actuator ordering.

    The feedback law implemented by the planner is

    .. math::

        \mathbf{u} = -\mathbf{K} \, \delta\mathbf{x}

    where :math:`\mathbf{K}` maps state errors to control inputs.
    Since the C++ planner uses a fixed actuator ordering, the rows of
    :math:`\mathbf{K}` must be permuted to match the Python actuator list.

    Supported gain tensor layouts include:

    - Time-major:

      .. math::

          \mathbf{K} \in \mathbb{R}^{N \times n_u \times n_x}

    - Control-major:

      .. math::

          \mathbf{K} \in \mathbb{R}^{n_u \times n_x \times N}

    Flattened 2D representations are passed through unchanged and are expected
    to be handled at a higher level.

    :param Kset:
        Gain tensor produced by the C++ planner.

    :type Kset:
        numpy.ndarray

    :param actuators:
        Python actuator list defining the desired control ordering.

    :type actuators:
        List[:class:`~ADCS.satellite_hardware.actuators.Actuator`]

    :return:
        Gain tensor reordered to match Python actuator ordering.

    :rtype:
        numpy.ndarray

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
    r"""
    Construct a C++ planner-compatible satellite model from a Python estimated satellite.

    This function converts an instance of
    :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
    into a :class:`~trajectory_planner.build.pysat.Satellite` object used by the
    C++ trajectory planner. The conversion includes:

    - Inertia tensor assignment
    - Actuator definitions
    - Environmental disturbance torques
    - Operational constraints

    The resulting satellite model is dynamically equivalent to the Python-side
    representation and suitable for optimization-based planning.

    :param est_sat:
        Estimated satellite containing inertia and actuator definitions.

    :type est_sat:
        :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`

    :param planner_settings:
        Planner configuration including weights, limits, and disturbance flags.

    :type planner_settings:
        :class:`~ADCS.controller.helpers.PlannerSettings`

    :return:
        Satellite object compatible with the C++ trajectory planner.

    :rtype:
        :class:`~trajectory_planner.build.pysat.Satellite`

    """
    _, pysat = get_trajectory_planner_modules()
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
    r"""
    Add a single actuator to a C++ planner satellite model.

    This function translates a Python actuator object into its corresponding
    C++ planner representation and registers it with the satellite model.
    Supported actuator types include:

    - Magnetorquers, modeled as magnetic dipole actuators
    - Reaction wheels, modeled with inertia, torque limits, and angular momentum bounds

    Actuator cost weights and saturation limits are taken from
    :class:`~ADCS.controller.helpers.PlannerSettings`.

    :param act:
        Actuator instance to be added.

    :type act:
        :class:`~ADCS.satellite_hardware.actuators.Actuator`

    :param csat:
        C++ planner satellite object to which the actuator is added.

    :type csat:
        :class:`~trajectory_planner.build.pysat.Satellite`

    :param planner_settings:
        Planner configuration providing weights and scaling factors.

    :type planner_settings:
        :class:`~ADCS.controller.helpers.PlannerSettings`

    :return:
        None.

    :rtype:
        None

    """
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