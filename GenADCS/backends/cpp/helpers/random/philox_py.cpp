#include "philox.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace random_helper = omni::helpers::random;

namespace
{
std::uint64_t seed_from_python(const py::handle seed)
{
    if (PyBool_Check(seed.ptr()) || PyIndex_Check(seed.ptr()) == 0)
        throw py::value_error("seed must be an integer in [0, 2**64)");

    PyObject *indexed_seed = PyNumber_Index(seed.ptr());
    if (indexed_seed == nullptr)
        throw py::error_already_set();
    const py::int_ integer = py::reinterpret_steal<py::int_>(indexed_seed);

    const long long signed_value = PyLong_AsLongLong(integer.ptr());
    if (PyErr_Occurred() == nullptr)
    {
        if (signed_value < 0)
            throw py::value_error("seed must be an integer in [0, 2**64)");
        return static_cast<std::uint64_t>(signed_value);
    }
    PyErr_Clear();
    const unsigned long long value = PyLong_AsUnsignedLongLong(integer.ptr());
    if (PyErr_Occurred() != nullptr)
    {
        PyErr_Clear();
        throw py::value_error("seed must be an integer in [0, 2**64)");
    }
    return static_cast<std::uint64_t>(value);
}
} // namespace

void bind_philox(py::module_ &module)
{
    module.def("_philox4x32", [](const py::object &seed, const std::uint64_t tick, const std::uint32_t sensor,
                                  const std::uint32_t block) {
        return random_helper::philox4x32(seed_from_python(seed), tick, sensor, block);
    });
    module.def("standard_normals", [](const py::object &seed, const std::uint64_t tick, const std::uint32_t sensor,
                                       const std::uint32_t count) {
        return random_helper::standard_normals(seed_from_python(seed), tick, sensor, count);
    }, py::arg("seed"), py::arg("tick"), py::arg("sensor"), py::arg("count"));
    module.def("validate_seed", [](const py::object &seed) { return seed_from_python(seed); });
}
