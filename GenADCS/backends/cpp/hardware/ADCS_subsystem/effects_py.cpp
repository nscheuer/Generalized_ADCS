#include "effects_py.hpp"
#include "../../helpers/math/rotations/sensor_kernels_py.hpp"
#include "adcs_subsystem.hpp"
#include "effects.hpp"
#include "magnetorquer.hpp"
#include "reaction_wheel.hpp"
#include "sensors/accelerometer.hpp"
#include "sensors/coarse_sun_sensor.hpp"
#include "sensors/fine_earth_sensor.hpp"
#include "sensors/fine_sun_sensor.hpp"
#include "sensors/gyroscope.hpp"
#include "sensors/magnetometer.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace hardware = omni::hardware;
namespace native_sensors = omni::hardware::sensors;

namespace omni::hardware
{
py::object apply_effects(py::object value, const py::sequence &effects, double dt, py::object rng)
{
    std::vector<Effect<py::object>> callbacks;
    for (py::handle effect : effects)
    {
        callbacks.emplace_back(
            [effect = py::reinterpret_borrow<py::object>(effect), rng](py::object input, double elapsed) {
                return effect.attr("apply")(std::move(input), elapsed, rng);
            });
    }
    return apply_effects(std::move(value), callbacks, dt);
}
py::list filter_effects(const py::sequence &effects, const char *module_name, const char *class_name)
{
    const py::object effect_type = py::module_::import(module_name).attr(class_name);
    py::list selected;
    for (const py::handle effect : effects)
    {
        if (py::isinstance(effect, effect_type))
        {
            selected.append(effect);
        }
    }
    return selected;
}

} // namespace omni::hardware

void bind_effects(py::module_ &module)
{
    module.def(
        "apply_effects",
        static_cast<py::object (*)(py::object, const py::sequence &, double, py::object)>(&hardware::apply_effects),
        py::arg("value"), py::arg("effects"), py::arg("dt"), py::arg("rng") = py::none());
}
