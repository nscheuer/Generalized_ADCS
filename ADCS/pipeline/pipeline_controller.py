"""
PipelineController: top-level orchestrator for the generalized ADCS pipeline.

Extends the existing Controller base class so it can be dropped into the
simulation loop. Wires Goal Formulation -> Control Law -> Compensation ->
Allocation in sequence.
"""

__all__ = ["PipelineController"]

import numpy as np
from typing import List, Optional

from ADCS.controller import Controller
from ADCS.CONOPS.goals import Goal, No_Goal
from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.actuators import MTQ, RW
from ADCS.satellite_hardware.sensors import MTM
from ADCS.helpers.math_helpers import rot_mat

from ADCS.pipeline.data import (
    CompensationConfig,
    CompensationInputs,
    AllocationConfig,
    ActuatorGroup,
)
from ADCS.pipeline.control_law.law_interface import ControlLaw
from ADCS.pipeline.goal_formulation import goal_formulation_step_legacy
from ADCS.pipeline.compensation import compensation_step
from ADCS.pipeline.allocation import allocation_step


class PipelineController(Controller):
    """Generalized ADCS pipeline controller.

    Orchestrates the 4-stage pipeline:
        1. Goal Formulation  (error signals)
        2. Control Law       (desired torque)
        3. Compensation      (feedforward terms)
        4. Allocation        (actuator commands)

    Parameters
    ----------
    est_sat : EstimatedSatellite
        Estimated satellite model.
    law : ControlLaw
        Control law instance (e.g., PD_Law).
    comp_config : CompensationConfig or None
        Compensation toggles. If None, auto-configured from law.interface.
    alloc_config : AllocationConfig or None
        Allocation method config. Defaults to magnetic_cross.
    """

    def __init__(
        self,
        est_sat: EstimatedSatellite,
        law: ControlLaw,
        comp_config: Optional[CompensationConfig] = None,
        alloc_config: Optional[AllocationConfig] = None,
    ) -> None:
        super().__init__(est_sat)
        self.law = law

        # Auto-configure compensation from law interface if not provided
        if comp_config is not None:
            self.comp_config = comp_config
        else:
            self.comp_config = CompensationConfig.from_law_interface(law.interface)

        # Default allocation: magnetic cross
        self.alloc_config = alloc_config or AllocationConfig(method='magnetic_cross')

        # Build B-field reconstruction matrix (same as Lovera)
        self.M_read, self.mtm_indices = self.build_sensor_matrix_pinv(
            sensors=est_sat.attitude_sensors + est_sat.rw_actuators,
            sensor_type=MTM,
        )

        # Build actuator groups from est_sat
        self.actuator_groups = self._build_actuator_groups(est_sat)
        self.n_actuators = len(est_sat.actuators)

    def find_u(
        self,
        x_hat: np.ndarray,
        sens: np.ndarray,
        est_sat: EstimatedSatellite,
        os_hat: Orbital_State,
        goal: Optional[Goal] = None,
        **kwargs,
    ) -> np.ndarray:
        """Compute actuator commands via the 4-stage pipeline.

        Parameters
        ----------
        x_hat : ndarray
            Estimated state vector [omega(3), q(4), h_rw(n_rw)].
        sens : ndarray
            Raw sensor measurements.
        est_sat : EstimatedSatellite
            Current estimated satellite model.
        os_hat : Orbital_State
            Estimated orbital state.
        goal : Goal or None
            Mission goal.

        Returns
        -------
        ndarray
            Actuator command vector.
        """
        if goal is None:
            goal = No_Goal()

        # --- Parse state ---
        omega = x_hat[0:3]
        q = x_hat[3:7]

        n_rw = len([a for a in est_sat.actuators if isinstance(a, RW)])
        if len(x_hat) >= 7 + n_rw:
            h_rw_states = x_hat[7:7 + n_rw]
        else:
            h_rw_states = np.array([rw.h for rw in est_sat.actuators if isinstance(rw, RW)])

        # Compute total RW angular momentum in body frame
        h_rw_body = np.zeros(3)
        rw_counter = 0
        for actuator in est_sat.actuators:
            if isinstance(actuator, RW):
                h_rw_body += np.asarray(actuator.axis).flatten() * h_rw_states[rw_counter]
                rw_counter += 1

        J = est_sat.J_0

        # Reconstruct B field from magnetometer sensors
        sens_arr = np.asarray(sens).reshape(-1)
        sens_clean = sens_arr.copy()
        sens_clean[np.isnan(sens_clean)] = 0.0
        B_body = self.M_read @ sens_clean

        # --- Stage 1: Goal Formulation ---
        # Use existing Goal interface for error computation
        q_err = goal.error(q=q, body_boresight=est_sat.boresight, os0=os_hat)
        _, omega_ref_eci = goal.to_ref(os0=os_hat)

        gf_out = goal_formulation_step_legacy(
            q_err=q_err,
            omega=omega,
            omega_ref_eci=omega_ref_eci,
            q=q,
            law_interface=self.law.interface,
        )

        # --- Stage 2: Control Law ---
        tau_law = self.law.compute(
            attitude_input=gf_out.attitude_output,
            omega_input=gf_out.omega_output,
        )

        # --- Stage 3: Compensation ---
        comp_inputs = CompensationInputs(
            P=gf_out.P,
            omega_ref_body=gf_out.omega_ref_body,
            goal_type=gf_out.goal_type,
            inject_damping=gf_out.inject_damping,
        )
        tau_desired = compensation_step(
            tau_law=tau_law,
            omega=omega,
            J=J,
            h_rw_body=h_rw_body,
            comp_config=self.comp_config,
            comp_inputs=comp_inputs,
        )

        # --- Stage 4: Allocation ---
        result = allocation_step(
            tau_desired=tau_desired,
            actuator_groups=self.actuator_groups,
            alloc_config=self.alloc_config,
            B_body=B_body,
            n_actuators=self.n_actuators,
        )

        return result.u

    @staticmethod
    def _build_actuator_groups(est_sat: EstimatedSatellite) -> List[ActuatorGroup]:
        """Build ActuatorGroup list from the satellite's actuators.

        Groups actuators by type (MTQ, RW) and records their axes,
        limits, and indices into the full command vector.
        """
        groups = []

        # Collect MTQs
        mtq_axes = []
        mtq_umax = []
        mtq_indices = []
        # Collect RWs
        rw_axes = []
        rw_umax = []
        rw_indices = []

        for i, act in enumerate(est_sat.actuators):
            if isinstance(act, MTQ):
                mtq_axes.append(np.asarray(act.axis).flatten())
                mtq_umax.append(act.u_max)
                mtq_indices.append(i)
            elif isinstance(act, RW):
                rw_axes.append(np.asarray(act.axis).flatten())
                rw_umax.append(act.u_max)
                rw_indices.append(i)

        if mtq_axes:
            groups.append(ActuatorGroup(
                group_type='mtq',
                axes=np.column_stack(mtq_axes),
                u_max=np.array(mtq_umax),
                indices=np.array(mtq_indices, dtype=int),
            ))

        if rw_axes:
            groups.append(ActuatorGroup(
                group_type='rw',
                axes=np.column_stack(rw_axes),
                u_max=np.array(rw_umax),
                indices=np.array(rw_indices, dtype=int),
            ))

        return groups
