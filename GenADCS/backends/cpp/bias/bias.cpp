#include "bias.hpp"

namespace py = pybind11;

namespace omni::bias {

[[noreturn]] void raise_not_implemented() {
    PyErr_SetString(PyExc_NotImplementedError, "");
    throw py::error_already_set();
}

py::object Bias::apply(const py::object&, double, const py::object&) const {
    raise_not_implemented();
}

}  // namespace omni::bias
