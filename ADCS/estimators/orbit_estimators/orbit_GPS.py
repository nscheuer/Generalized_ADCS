__all__ = ["Orbit_GPS"]

import numpy as np
from typing import List, Optional
from scipy.linalg import block_diag

from ADCS.satellite_hardware.satellite.estimated_satellite import EstimatedSatellite
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.estimators.estimator_helpers.estimator_helpers import EstimatedOrbital_State
from ADCS.estimators.orbit_estimators import Orbit_Estimator

class Orbit_GPS(Orbit_Estimator):
    r"""
    A 'Pass-Through' Estimator that directly converts GPS measurements into an Inertial State.

    This estimator does not perform filtering (such as an Extended Kalman Filter) or propagate state dynamics
    over time. Instead, it accepts the current GPS measurement vector (typically in the Earth-Centered,
    Earth-Fixed frame) and transforms it into the Earth-Centered Inertial (ECI) frame to form the estimate.

    .. math::
        \hat{\mathbf{x}}_{ECI} = T_{ECEF \to ECI}(t) \cdot \mathbf{m}_{meas}

    This class is primarily utilized for:
        * System initialization (providing an initial seed for other propagators).
        * Scenarios where GPS data is trusted implicitly without dynamic modeling.
        * Debugging coordinate frame transformations.
    """
    def __init__(
        self, 
        est_sat: EstimatedSatellite, 
        J2000: float,
        os_template: Orbital_State
    ) -> None:
        r"""
        Initializes the Orbit GPS estimator.

        :param est_sat: The satellite model containing hardware specifications and sensor noise parameters.
        :type est_sat: ~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite
        :param J2000: The current J2000 time epoch [s]. (Unused in initialization logic, but kept for interface consistency).
        :type J2000: float
        :param os_template: A template orbital state used to access environment models (Ephemeris, Density)
                            required for coordinate frame conversions.
        :type os_template: ~ADCS.orbits.orbital_state.Orbital_State
        """
        super().__init__(est_sat=est_sat, dt=0.0)
        self.reset(est_sat=est_sat, os_template=os_template)

    def reset(
        self, 
        est_sat: EstimatedSatellite, 
        os_template: Orbital_State
    ) -> None:
        r"""
        Resets the estimator configuration and noise parameters.

        This method extracts the standard deviation of the noise from the first available GPS sensor
        in the provided satellite model to populate the internal covariance settings.

        :param est_sat: The satellite model containing hardware specifications.
        :type est_sat: ~ADCS.satellite_hardware.satellite.estimated_satellite.EstimatedSatellite
        :param os_template: A template orbital state for environment models.
        :type os_template: ~ADCS.orbits.orbital_state.Orbital_State
        :raises ValueError: If ``est_sat`` does not contain any GPS sensors.
        """
        self.est_sat = est_sat
        self.os_template = os_template
        
        gps_sensors = self.est_sat.GPS_sensors
        if not gps_sensors:
            raise ValueError("Orbit_GPS requires at least one GPS sensor.")
            
        blocks = []
        for gps in gps_sensors:
            std = gps.noise.std_noise

            self.gps_std = std

    def update(
        self,
        GPS_measurements: List[np.ndarray],
        J2000: float
    ) -> EstimatedOrbital_State:
        r"""
        Updates the state estimate using the latest GPS measurements.

        The update process involves:
        
        1. Creating a temporary :class:`~ADCS.orbits.orbital_state.Orbital_State` at the current time ``J2000``.
        2. Converting the input ECEF measurement (Position :math:`\mathbf{r}` or Position+Velocity :math:`\mathbf{r}, \mathbf{v}`) 
           into the ECI frame.
        3. Constructing the covariance matrix :math:`P` based on sensor noise specifications.

        .. math::
            P = \text{diag}(\sigma_{pos}^2, \dots, \sigma_{vel}^2)

        :param GPS_measurements: A list of raw GPS measurements.
                                 Shape is typically ``(1, 3)`` for position only or ``(1, 6)`` for pos+vel.
        :type GPS_measurements: list[np.ndarray]
        :param J2000: The current J2000 time epoch [s].
        :type J2000: float
        :return: The estimated orbital state containing the state vector :math:`\hat{\mathbf{x}}` and covariance :math:`P`.
        :rtype: ~ADCS.estimators.estimator_helpers.estimator_helpers.EstimatedOrbital_State
        :raises ValueError: If the measurement vector size is neither 3 nor 6.
        """
        
        if not GPS_measurements:
            return self.os_hat


        m = np.asarray(GPS_measurements[0]).reshape(-1)

        # 1. Create a temporary state at current J2000 to handle frame conversion.
        #    We inherit the physics models (Ephemeris) from the template.
        temp_os = Orbital_State(
            ephem=self.os_template.ephem,
            J2000=J2000,
            R=np.zeros(3), # Dummy values
            V=np.zeros(3),
            density_model=self.os_template.density_model
        )

        # 2. Convert ECEF Measurement -> ECI State
        if m.size == 3:
            # Position only
            r_ecef = m
            r_eci = temp_os.ecef_to_eci(r_ecef)

            if self.os_hat:
                v_eci = self.os_hat.os.V
            else:
                v_eci = np.zeros(3)
                
            std_pos = self.gps_std
            std_vel = 1000.0 # High uncertainty for unmeasured velocity
            P = np.diag([std_pos]*3 + [std_vel]*3)**2

        elif m.size == 6:
            # Position + Velocity
            r_ecef = m[0:3]
            v_ecef = m[3:6]

            r_eci = temp_os.ecef_to_eci(r_ecef)
            v_eci = temp_os.ecef_to_eci(v_ecef)
            
            std = self.gps_std
            P = np.diag([std]*6)**2
            
        else:
            raise ValueError(f"Unknown GPS measurement size: {m.size}")

        # 3. Create the new State
        new_os = Orbital_State(
            ephem=self.os_template.ephem,
            J2000=J2000,
            R=r_eci,
            V=v_eci,
            density_model=self.os_template.density_model
        )

        # 4. Return Estimated State
        self.os_hat = EstimatedOrbital_State(
            os=new_os,
            P=P,
            Q=np.zeros((6,6)) # No process noise in a static pass-through
        )

        return self.os_hat