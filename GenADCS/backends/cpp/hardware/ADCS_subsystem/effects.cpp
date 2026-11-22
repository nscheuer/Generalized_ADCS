#include "effects.hpp"

#include <string>
#include <utility>

namespace py = pybind11;

namespace omni::hardware {
py::object apply_effects(py::object value,
                         const py::sequence& effects,
                         double dt,
                         py::object rng) {
    for (const py::handle effect : effects) {
        value = py::reinterpret_borrow<py::object>(effect).attr("apply")(value, dt, rng);
    }
    return value;
}

py::list filter_effects(const py::sequence& effects,
                        const char* module_name,
                        const char* class_name) {
    const py::object effect_type = py::module_::import(module_name).attr(class_name);
    py::list selected;
    for (const py::handle effect : effects) {
        if (py::isinstance(effect, effect_type)) {
            selected.append(effect);
        }
    }
    return selected;
}
}
