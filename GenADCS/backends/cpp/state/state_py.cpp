#include "python_conversions.hpp"
using namespace omni::state_bindings;

void bind_state(py::module_& m) {
    py::class_<State>(m, "State")
        .def(py::init<AttitudeState, OrbitalState, InternalState>(), py::arg("attitude"), py::arg("orbit"), py::arg("internal") = InternalState{})
        .def_readwrite("attitude", &State::attitude)
        .def_readwrite("orbit", &State::orbit)
        .def_readwrite("internal", &State::internal)
        .def("to_record", &state_record, py::arg("wheel_count") = py::none())
        .def_static("from_record", &from_record, py::arg("record"), py::arg("julian_date"), py::arg("ephemeris") = py::none())
        .def_static("from_python", [](py::object s) {
            return State(attitude_from_python(s.attr("attitude")),
                         orbital_from_python(s.attr("orbit")),
                         InternalState(variable(s.attr("internal").attr("wheel_momentum"), "wheel_momentum")));
        })
        .def("to_python", [](const State& s) {
            return python_states().attr("State")(to_python(s.attitude), to_python(s.orbit), to_python(s.internal));
        });

}
