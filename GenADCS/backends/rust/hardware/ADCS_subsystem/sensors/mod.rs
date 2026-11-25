mod accelerometer;
mod accelerometer_py;
mod autonomous_star_tracker;
mod autonomous_star_tracker_py;
mod coarse_sun_sensor;
mod coarse_sun_sensor_py;
mod fine_earth_sensor;
mod fine_earth_sensor_py;
mod fine_sun_sensor;
mod fine_sun_sensor_py;
mod gnss;
mod gnss_py;
mod gyroscope;
mod gyroscope_py;
pub mod kernels;
mod kernels_py;
mod magnetometer;
mod magnetometer_py;
mod sensor;
mod sensor_py;

pub use accelerometer::Accelerometer;
pub use accelerometer::Accelerometer as NativeAccelerometer;
pub use accelerometer_py::PyAccelerometer;
pub use autonomous_star_tracker::AutonomousStarTracker;
pub use autonomous_star_tracker::AutonomousStarTracker as NativeAutonomousStarTracker;
pub use autonomous_star_tracker_py::PyAutonomousStarTracker;
pub use coarse_sun_sensor::CoarseSunSensor;
pub use coarse_sun_sensor::CoarseSunSensor as NativeCoarseSunSensor;
pub use coarse_sun_sensor_py::PyCoarseSunSensor;
pub use fine_earth_sensor::FineEarthSensor;
pub use fine_earth_sensor::FineEarthSensor as NativeFineEarthSensor;
pub use fine_earth_sensor_py::PyFineEarthSensor;
pub use fine_sun_sensor::FineSunSensor as NativeFineSunSensor;
pub use fine_sun_sensor_py::PyFineSunSensor;
pub use gnss::GNSS;
pub use gnss::GNSS as NativeGNSS;
pub use gnss_py::PyGNSS;
pub use gyroscope::Gyroscope;
pub use gyroscope::Gyroscope as NativeGyroscope;
pub use gyroscope_py::PyGyroscope;
pub use magnetometer::Magnetometer;
pub use magnetometer::Magnetometer as NativeMagnetometer;
pub use magnetometer_py::PyMagnetometer;
pub use sensor::Sensor;
pub use sensor_py::PySensor;

#[cfg(test)]
#[path = "../../../../../tests/backends/rust/hardware/ADCS_subsystem/sensors/test_accelerometer.rs"]
mod test_accelerometer;
#[cfg(test)]
#[path = "../../../../../tests/backends/rust/hardware/ADCS_subsystem/sensors/test_autonomous_star_tracker.rs"]
mod test_autonomous_star_tracker;
#[cfg(test)]
#[path = "../../../../../tests/backends/rust/hardware/ADCS_subsystem/sensors/test_coarse_sun_sensor.rs"]
mod test_coarse_sun_sensor;
#[cfg(test)]
#[path = "../../../../../tests/backends/rust/hardware/ADCS_subsystem/sensors/test_fine_earth_sensor.rs"]
mod test_fine_earth_sensor;
#[cfg(test)]
#[path = "../../../../../tests/backends/rust/hardware/ADCS_subsystem/sensors/test_fine_sun_sensor.rs"]
mod test_fine_sun_sensor;
#[cfg(test)]
#[path = "../../../../../tests/backends/rust/hardware/ADCS_subsystem/sensors/test_gnss.rs"]
mod test_gnss;
#[cfg(test)]
#[path = "../../../../../tests/backends/rust/hardware/ADCS_subsystem/sensors/test_gyroscope.rs"]
mod test_gyroscope;
#[cfg(test)]
#[path = "../../../../../tests/backends/rust/hardware/ADCS_subsystem/sensors/test_magnetometer.rs"]
mod test_magnetometer;

pub fn register(module: &pyo3::types::PyModule) -> pyo3::PyResult<()> {
    kernels_py::register(module)?;
    module.add_class::<PySensor>()?;
    module.add_class::<PyAccelerometer>()?;
    module.add_class::<PyAutonomousStarTracker>()?;
    module.add_class::<PyCoarseSunSensor>()?;
    module.add_class::<PyFineEarthSensor>()?;
    module.add_class::<PyFineSunSensor>()?;
    module.add_class::<PyGNSS>()?;
    module.add_class::<PyGyroscope>()?;
    module.add_class::<PyMagnetometer>()?;
    Ok(())
}
