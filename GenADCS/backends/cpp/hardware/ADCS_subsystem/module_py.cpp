#include <pybind11/pybind11.h>
namespace py = pybind11;
void bind_effects(py::module_ &module);
void bind_reaction_wheel(py::module_ &module);
void bind_magnetorquer(py::module_ &module);
void bind_adcs_subsystem(py::module_ &module);
void bind_sensors_gyroscope(py::module_ &module);
void bind_sensors_magnetometer(py::module_ &module);
void bind_sensors_accelerometer(py::module_ &module);
void bind_sensors_coarse_sun_sensor(py::module_ &module);
void bind_sensors_autonomous_star_tracker(py::module_ &module);
void bind_sensors_fine_sun_sensor(py::module_ &module);
void bind_sensors_fine_earth_sensor(py::module_ &module);
void bind_sensors_gnss(py::module_ &module);
void bind_philox(py::module_ &module);

PYBIND11_MODULE(_adcs_subsystem, module)
{
    bind_effects(module);
    bind_reaction_wheel(module);
    bind_magnetorquer(module);
    bind_adcs_subsystem(module);
    bind_sensors_gyroscope(module);
    bind_sensors_magnetometer(module);
    bind_sensors_accelerometer(module);
    bind_sensors_coarse_sun_sensor(module);
    bind_sensors_autonomous_star_tracker(module);
    bind_sensors_fine_sun_sensor(module);
    bind_sensors_fine_earth_sensor(module);
    bind_sensors_gnss(module);
    bind_philox(module);
}
