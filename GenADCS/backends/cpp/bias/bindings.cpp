#include "bias.hpp"
#include <limits>

namespace py = pybind11;
using omni::bias::Bias;

PYBIND11_MODULE(_bias, module) {
    module.doc() = "C++ bindings for the GenADCS Bias model";
    py::class_<Bias>(module, "Bias")
        .def(py::init([](py::object bias, py::object std_bias, py::object bounds) {
            auto cls = py::module_::import("GenADCS.backends.python.bias.bias").attr("Bias");
            return Bias(cls(bias, std_bias, bounds));
        }), py::arg("bias") = 0.0, py::arg("std_bias") = 0.0,
            py::arg("bounds") = py::make_tuple(-std::numeric_limits<double>::infinity(),
                                                std::numeric_limits<double>::infinity()))
        .def_property("bias", [](const Bias& self) { return self.implementation.attr("bias"); },
                      [](Bias& self, py::object value) { self.implementation.attr("bias") = value; })
        .def_property("std_bias", [](const Bias& self) { return self.implementation.attr("std_bias"); },
                      [](Bias& self, py::object value) { self.implementation.attr("std_bias") = value; })
        .def_property("estimated_std_bias", [](const Bias& self) { return self.implementation.attr("estimated_std_bias"); },
                      [](Bias& self, py::object value) { self.implementation.attr("estimated_std_bias") = value; })
        .def_property("bounds", [](const Bias& self) { return self.implementation.attr("bounds"); },
                      [](Bias& self, py::object value) { self.implementation.attr("bounds") = value; })
        .def_property("last_bias_time", [](const Bias& self) { return self.implementation.attr("last_bias_time"); },
                      [](Bias& self, py::object value) { self.implementation.attr("last_bias_time") = value; })
        .def("__bool__", [](const Bias& self) { return self.implementation.attr("__bool__")().cast<bool>(); })
        .def("copy", &Bias::copy)
        .def("broadcast_to", &Bias::broadcast_to, py::arg("length"), py::kw_only(), py::arg("owner") = "sensor")
        .def("process_covariance", &Bias::process_covariance, py::arg("dt") = 1.0, py::kw_only(), py::arg("form") = "full")
        .def("process_psd", &Bias::process_psd, py::kw_only(), py::arg("form") = "full")
        .def("_update_bias", &Bias::update_bias, py::arg("j2000"), py::arg("rng") = py::none())
        .def("get_bias", &Bias::get_bias, py::arg("j2000"))
        .def("apply", &Bias::apply, py::arg("value"), py::arg("dt"), py::arg("rng") = py::none());
}
