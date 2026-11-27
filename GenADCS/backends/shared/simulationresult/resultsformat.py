"""Select recorded fields and full-dt sampling interval for a simulation."""

from dataclasses import dataclass, fields


@dataclass(frozen=True, slots=True)
class ResultsFormat:
    satellite_attitude: bool = True
    satellite_rate: bool = True
    satellite_wheel_momenta: bool = True
    satellite_position: bool = False
    satellite_velocity: bool = False
    relative_sun_vector: bool = False
    relative_earth_vector: bool = False
    atmospheric_density: bool = False
    relative_atmospheric_velocity: bool = False
    magnetic_field: bool = False
    eclipse_state: bool = False
    gravity_force: bool = False
    aerodynamic_force: bool = False
    solar_radiation_force: bool = False
    J2_force: bool = False
    J3_force: bool = False
    J4_force: bool = False
    J5_force: bool = False
    J6_force: bool = False
    lunisolar_force: bool = False
    gravity_gradient_torque: bool = False
    aerodynamic_torque: bool = False
    solar_radiation_torque: bool = False
    magnetic_torque: bool = False
    total_force: bool = False
    total_torque: bool = False
    gyroscope_clean: bool = False
    gyroscope_reading: bool = False
    magnetometer_clean: bool = False
    magnetometer_reading: bool = False
    autonomous_star_tracker_clean: bool = False
    autonomous_star_tracker_reading: bool = False
    coarse_sun_sensor_clean: bool = False
    coarse_sun_sensor_reading: bool = False
    fine_sun_sensor_clean: bool = False
    fine_sun_sensor_reading: bool = False
    fine_earth_sensor_clean: bool = False
    fine_earth_sensor_reading: bool = False
    gnss_clean: bool = False
    gnss_reading: bool = False
    accelerometer_clean: bool = False
    accelerometer_reading: bool = False
    N: int = 1
    estimated_attitude: bool = False
    estimated_rate: bool = False
    estimated_covariance: bool = False
    estimated_attitude_covariance: bool = False
    estimated_rate_covariance: bool = False
    measurement_innovation: bool = False
    measurement_innovation_covariance: bool = False
    normalized_innovation_squared: bool = False
    normalized_estimation_error_squared: bool = False
    measurement_residual: bool = False
    kalman_gain: bool = False
    obc_mode: bool = False
    adcs_mode: bool = False
    adcs_goal: bool = False
    goal_attitude: bool = False
    goal_rate: bool = False
    feedforward_torque: bool = False
    feedforward_attitude: bool = False
    feedforward_rate: bool = False
    reference_torque: bool = False
    commanded_wheel_torque: bool = False
    commanded_magnetic_dipole: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.N, bool) or not isinstance(self.N, int) or self.N < 1:
            raise ValueError("N must be a positive integer")
        for name in (field.name for field in fields(self) if field.name != "N"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
