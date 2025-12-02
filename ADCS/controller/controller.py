__all__ = ["Controller"]

import numpy as np
from typing import List, Tuple, Type

from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.satellite_hardware.sensors import Sensor
from ADCS.satellite_hardware.actuators import Actuator

class Controller():
    r"""
    Base abstract controller for all ADCS control law implementations.

    This class defines the core interface required in every ADCS controller.
    Controllers are expected to compute actuator input commands that achieve
    a desired torque or attitude regulation objective.

    Parameters
    ----------
    est_sat : :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
        Estimated satellite model containing inertia, actuator layout,
        and sensor configuration.
    **kwargs : dict
        Optional keyword parameters forwarded to derived controllers.

    Notes
    -----
    Controllers that inherit from this base must override
    :meth:`~ADCS.controller.Controller.find_u`.

    """
    def __init__(self, est_sat: EstimatedSatellite, **kwargs) -> None:
        r"""
        Initializes the base controller class.

        Parameters
        ----------
        est_sat : :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
            Estimated satellite model.
        **kwargs : dict
            Additional optional arguments needed by subclasses.

        """
        pass


    def find_u(self, x_hat: np.ndarray, sens: np.ndarray, est_sat: EstimatedSatellite, os_hat: Orbital_State, goal_vector_eci: np.ndarray | None = None, w_ref: np.ndarray | None = None, **kwargs) -> np.ndarray:
        r"""
        Computes actuator inputs required to satisfy the control objective.

        This function must be overridden by subclasses. Implementations
        typically compute a torque objective in the body frame and then
        allocate actuator commands accordingly.

        Parameters
        ----------
        x_hat : :class:`numpy.ndarray`
            Estimated spacecraft state vector. Conventionally includes angular
            velocity and quaternion attitude estimates.
        sens : :class:`numpy.ndarray`
            Flattened sensor measurement vector from onboard hardware such as
            magnetometers, sun sensors, star trackers, etc.
        est_sat : :class:`~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite`
            Estimated satellite object providing inertia, actuator geometries,
            and possible reaction wheel momentum estimates.
        os_hat : :class:`~ADCS.orbits.orbital_state.Orbital_State`
            Estimated orbital state (e.g. ECI position, velocity).
        goal_vector_eci : :class:`numpy.ndarray` or None
            Desired pointing direction expressed in the ECI frame.
        w_ref : :class:`numpy.ndarray` or None
            Desired angular rate reference in the body frame.

        Returns
        -------
        :class:`numpy.ndarray`
            Actuator command vector. The internal indexing convention must
            match that of the satellite model.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the derived controller.

        """
        raise NotImplementedError(
            "find_u() must be implemented in child controller classes."
        )


    def build_sensor_matrix_pinv(self, sensors: List[Sensor], sensor_type: Type[Sensor]) -> Tuple[np.ndarray, List[int]]:
        r"""
        Constructs a measurement reconstruction matrix for a specific
        class of sensors by computing a Moore–Penrose pseudoinverse
        based on their sensing axes.

        Given a stacked sensor measurement vector :math:`y`, this matrix
        maps sensor outputs to a physical 3-vector (e.g. the estimated
        magnetic field in body coordinates).

        .. math::

            \mathbf{v}_{body} \approx M_{\mathrm{sens}} \, y

        Parameters
        ----------
        sensors : list[:class:`~ADCS.satellite_hardware.sensors.Sensor`]
            List of all satellite sensors.
        sensor_type : type
            Sensor type to extract. Must inherit from
            :class:`~ADCS.satellite_hardware.sensors.Sensor`.

        Returns
        -------
        M_sens : :class:`numpy.ndarray`
            Pseudoinverse reconstruction matrix of shape (3, N), mapping
            measurements to a 3-vector physical estimate.
        indices : list[int]
            Flattened measurement indices belonging to the selected sensor type.

        Raises
        ------
        TypeError
            If `sensor_type` does not subclass Sensor.
        ValueError
            If no sensors of the given type are found.

        """
        if not issubclass(sensor_type, Sensor):
            raise TypeError(f"sensor_type must be a subclass of Sensor, got {sensor_type}")
        
        cols = []
        indices = []
        curr_global_idx = 0

        for sens in sensors:
            axis = np.asarray(sens.axis, dtype=float).reshape(3, -1)
            num_inputs = axis.shape[1]

            if isinstance(sens, sensor_type):
                for i in range(num_inputs):
                    cols.append(axis[:, i])
                    indices.append(curr_global_idx + i)

            curr_global_idx += num_inputs

        if not cols:
            raise ValueError(f"No sensos of type {sensor_type.__name__} found in the sensor list.")
        
        A = np.column_stack(cols)
        M_sens = np.linalg.pinv(A)

        return M_sens, indices
    

    def build_torque_to_u_matrix_pinv(self, actuators: List[Actuator], actuator_type: Type[Actuator]) -> Tuple[np.ndarray, List[int]]:
        r"""
        Builds the actuator allocation matrix that maps desired body torques
        into actuator command space via the pseudoinverse of the actuator
        direction matrix.

        The columns of the internal matrix correspond to the unit axis
        directions along which the selected actuators can impart torque.
        Its Moore–Penrose pseudoinverse yields the minimum-norm command
        vector that best matches a desired torque.

        .. math::

            \mathbf{u} \approx A^{\dagger} \, \boldsymbol{\tau}

        Parameters
        ----------
        actuators : list[:class:`~ADCS.satellite_hardware.actuators.Actuator`]
            List of all satellite actuators.
        actuator_type : type
            Target actuator type (e.g. :class:`~ADCS.satellite_hardware.actuators.RW`
            or :class:`~ADCS.satellite_hardware.actuators.MTQ`).

        Returns
        -------
        M_act : :class:`numpy.ndarray`
            Allocation pseudoinverse of size (N, 3), mapping desired torque to commands.
        indices : list[int]
            Indices of the actuator channels belonging to the selected type.

        Raises
        ------
        TypeError
            If `actuator_type` does not subclass Actuator.
        ValueError
            If no actuators of the given type exist.

        """
        if not issubclass(actuator_type, Actuator):
            raise TypeError(f"actuator_type must be a subclass of Actuator, got {actuator_type}")
        
        cols = []
        indices = []
        curr_global_idx = 0

        for act in actuators:
            axis = np.asarray(act.axis, dtype=float).reshape(3, -1)
            num_inputs = axis.shape[1]

            if isinstance(act, actuator_type):
                for i in range(num_inputs):
                    cols.append(axis[:, i])
                    indices.append(curr_global_idx + i)

            curr_global_idx += num_inputs

        if not cols:
            raise ValueError(f"No actuators of type {actuator_type.__name__} found in the actuator list.")
        
        A = np.column_stack(cols)
        M_act = np.linalg.pinv(A)

        return M_act, indices
    

    def build_u_to_torque_matrix_pinv(self, actuators: List[Actuator], actuator_type: Type[Actuator]) -> np.ndarray:
        r"""
        Builds the forward mapping matrix that converts actuator inputs directly
        into the physical torque contribution direction in the body frame.

        Unlike :meth:`~ADCS.controller.Controller.build_torque_to_u_matrix_pinv`,
        which gives the inverse allocation mapping from torque to commands,
        this matrix is used in the forward physical model:

        .. math::

            \boldsymbol{\tau} = A \, \mathbf{u}

        Examples
        --------
        - For reaction wheels:
          each column of :math:`A` is the torque axis direction.
        - For magnetorquers:
          each column represents the direction of the magnetic dipole.
          Body torque is then computed by

        .. math::

            \boldsymbol{\tau} = \left[\mathbf{B}\right]_{\times} A \mathbf{u}

        Parameters
        ----------
        actuators : list[:class:`~ADCS.satellite_hardware.actuators.Actuator`]
            List of all satellite actuators.
        actuator_type : type
            Target actuator type.

        Returns
        -------
        :class:`numpy.ndarray`
            Matrix of stacked axis vectors, shape (3, N). If no actuators
            of the requested type are found, returns an empty (3, 0) matrix.

        """
        cols = []
        for act in actuators:
            if isinstance(act, actuator_type):
                axis = np.asarray(act.axis, dtype=float).reshape(3, -1)
                for i in range(axis.shape[1]):
                    cols.append(axis[:, i])
        
        if not cols:
            return np.zeros((3, 0))
        
        return np.column_stack(cols)
    

    def find_max_torque(self, actuators: List[Actuator], actuator_type: Type[Actuator]) -> np.ndarray:
        r"""
        Extracts the maximum actuator command magnitude for the given actuator type.

        The returned vector matches the number of command channels for that type.
        This value is typically used to clip, saturate, or normalize actuator
        control inputs.

        Parameters
        ----------
        actuators : list[:class:`~ADCS.satellite_hardware.actuators.Actuator`]
            List of all actuators.
        actuator_type : type
            Target actuator type.

        Returns
        -------
        :class:`numpy.ndarray`
            Vector of maximum allowable actuator input values.

        Raises
        ------
        ValueError
            If no actuators of the requested type are present.

        """
        max_torque = np.array([act.u_max for act in actuators if isinstance(act, actuator_type)])
        if len(max_torque) == 0:
            raise ValueError(f"No actuators of type {actuator_type.__name__} found to determine max input limit (u_max).")
        return max_torque