#include "python_conversions.hpp"
using namespace omni::state_bindings;

void bind_internal_state(py::module_ &m) {
  py::class_<InternalState>(m, "InternalState")
      .def(py::init([](py::object v) {
             return InternalState(variable(v, "wheel_momentum"));
           }),
           py::arg("wheel_momentum") = py::list())
      .def_property(
          "wheel_momentum",
          [](const InternalState &s) { return to_array(s.wheel_momentum); },
          [](InternalState &s, py::object v) {
            s = InternalState(variable(v, "wheel_momentum"));
          })
      .def_static("from_python",
                  [](py::object s) {
                    return InternalState(
                        variable(s.attr("wheel_momentum"), "wheel_momentum"));
                  })
      .def("to_python", [](const InternalState &s) { return to_python(s); });

  py::class_<WheelMomentumState, InternalState>(m, "WheelMomentumState")
      .def(py::init([](py::object subsystem, py::object momentum) {
             if (!subsystem.is_none() && !momentum.is_none()) {
               throw py::value_error(
                   "provide subsystem or wheel_momentum, not both");
             }
             if (!subsystem.is_none()) {
               py::list values;
               for (py::handle wheel : subsystem.attr("reaction_wheels")) {
                 values.append(wheel.attr("h"));
               }
               momentum = values;
             }
             return WheelMomentumState(variable(
                 momentum.is_none() ? py::list() : momentum, "wheel_momentum"));
           }),
           py::arg("subsystem") = py::none(), py::kw_only(),
           py::arg("wheel_momentum") = py::none())
      .def_static("from_python",
                  [](py::object s) {
                    return WheelMomentumState(
                        variable(s.attr("wheel_momentum"), "wheel_momentum"));
                  })
      .def("to_python", [](const WheelMomentumState &s) {
        return python_states().attr("WheelMomentumState")(
            py::arg("wheel_momentum") = to_array(s.wheel_momentum));
      });
}
