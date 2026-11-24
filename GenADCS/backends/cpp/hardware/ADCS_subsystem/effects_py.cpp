#include "adcs_subsystem.hpp"
#include "effects.hpp"
#include "magnetorquer.hpp"
#include "reaction_wheel.hpp"
#include "sensors/gyroscope.hpp"
#include "sensors/magnetometer.hpp"
#include "sensors/accelerometer.hpp"
#include "sensors/coarse_sun_sensor.hpp"
#include "sensors/fine_sun_sensor.hpp"
#include "sensors/fine_earth_sensor.hpp"
#include "../../helpers/math/rotations/sensor_kernels_py.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace hardware = omni::hardware;
namespace native_sensors = omni::hardware::sensors;

void bind_effects(py::module_& module) {
    module.def("apply_effects", &hardware::apply_effects,
               py::arg("value"), py::arg("effects"), py::arg("dt"),
               py::arg("rng") = py::none());

}
