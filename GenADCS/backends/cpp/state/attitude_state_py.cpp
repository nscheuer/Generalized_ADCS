#include "python_conversions.hpp"
using namespace omni::state_bindings;

void bind_attitude_state(py::module_& m) {
    py::class_<AttitudeState>(m, "AttitudeState")
        .def(py::init([](py::object orientation, py::object rates) {
            return AttitudeState(fixed<4>(orientation_value(orientation), "orientation"), fixed<3>(rates, "rates"));
        }), py::arg("orientation"), py::arg("rates"))
        .def_property("orientation", [](const AttitudeState& s) { return to_array(s.orientation); },
                      [](AttitudeState& s, py::object v) { s = AttitudeState(fixed<4>(v, "orientation"), s.rates); })
        .def_property("rates", [](const AttitudeState& s) { return to_array(s.rates); },
                      [](AttitudeState& s, py::object v) { s.rates = fixed<3>(v, "rates"); })
        .def_property_readonly("quaternion_array", [](const AttitudeState& s) { return to_array(s.quaternion_array()); })
        .def_static("from_python", &attitude_from_python)
        .def("to_python", [](const AttitudeState& s) { return to_python(s); });


}
