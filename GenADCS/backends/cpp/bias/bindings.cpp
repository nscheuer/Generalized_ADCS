#include "bias.hpp"
#include "fixed_bias.hpp"
#include "linear_drift.hpp"
#include "random_walk_bias.hpp"
#include "scale_factor_error.hpp"

namespace py = pybind11;
using namespace omni::bias;

PYBIND11_MODULE(_bias, module) {
    py::class_<Bias>(module, "Bias")
        .def("apply", &Bias::apply, py::arg("value"), py::arg("dt"),
             py::arg("rng") = py::none());
    py::class_<FixedBias, Bias>(module, "FixedBias")
        .def(py::init<double>(), py::arg("value"))
        .def_readwrite("value", &FixedBias::value);
    py::class_<LinearDrift, Bias>(module, "LinearDrift")
        .def(py::init<double, double>(), py::arg("rate"), py::arg("initial_value") = 0.0)
        .def_readwrite("rate", &LinearDrift::rate)
        .def_readwrite("initial_value", &LinearDrift::initial_value);
    py::class_<RandomWalkBias, Bias>(module, "RandomWalkBias")
        .def(py::init<double, double>(), py::arg("standard_deviation"),
             py::arg("initial_value") = 0.0)
        .def_readwrite("standard_deviation", &RandomWalkBias::standard_deviation)
        .def_readwrite("initial_value", &RandomWalkBias::initial_value);
    py::class_<ScaleFactorError, Bias>(module, "ScaleFactorError")
        .def(py::init<double>(), py::arg("fraction"))
        .def_readwrite("fraction", &ScaleFactorError::fraction);
}
