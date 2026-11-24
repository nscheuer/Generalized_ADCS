#include "satellite_geometry.hpp"

#include <algorithm>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace hardware = omni::hardware;

namespace {
std::array<double, 3> vector3(py::handle value) {
    auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(value);
    if (!array || array.size() != 3) throw py::value_error("expected three numeric components");
    std::array<double, 3> result{};
    std::copy(array.data(), array.data() + 3, result.begin());
    return result;
}

std::array<double, 2> vector2(py::handle value) {
    auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(value);
    if (!array || array.size() != 2) throw py::value_error("expected two numeric components");
    std::array<double, 2> result{};
    std::copy(array.data(), array.data() + 2, result.begin());
    return result;
}

py::tuple tuple3(const std::array<double, 3>& value) {
    return py::make_tuple(value[0], value[1], value[2]);
}

py::tuple tuple2(const std::array<double, 2>& value) {
    return py::make_tuple(value[0], value[1]);
}
}

void bind_satellite_panel(py::module_& module) {
    py::class_<hardware::SatellitePanel>(module, "SatellitePanel")
        .def(py::init([](py::object position, py::object orientation, py::object dimensions,
                         double eta_s, double eta_d, double eta_a, double cd) {
            return hardware::SatellitePanel(vector3(position), vector3(orientation), vector2(dimensions),
                                            eta_s, eta_d, eta_a, cd);
        }), py::arg("position"), py::arg("orientation"), py::arg("dimensions"),
            py::arg("eta_s") = 0.0, py::arg("eta_d") = 0.0,
            py::arg("eta_a") = 1.0, py::arg("cd") = 2.2)
        .def_property("position", [](const hardware::SatellitePanel& p) { return tuple3(p.position); },
                      [](hardware::SatellitePanel& p, py::object v) { p.position = vector3(v); })
        .def_property("orientation", [](const hardware::SatellitePanel& p) { return tuple3(p.orientation); },
                      [](hardware::SatellitePanel& p, py::object v) { p.orientation = vector3(v); })
        .def_property("dimensions", [](const hardware::SatellitePanel& p) { return tuple2(p.dimensions); },
                      [](hardware::SatellitePanel& p, py::object v) { p.dimensions = vector2(v); })
        .def_readwrite("eta_s", &hardware::SatellitePanel::eta_s)
        .def_readwrite("eta_d", &hardware::SatellitePanel::eta_d)
        .def_readwrite("eta_a", &hardware::SatellitePanel::eta_a)
        .def_readwrite("cd", &hardware::SatellitePanel::cd)
        .def("__eq__", [](const hardware::SatellitePanel& a, const hardware::SatellitePanel& b) {
            return a.position == b.position && a.orientation == b.orientation
                && a.dimensions == b.dimensions && a.eta_s == b.eta_s
                && a.eta_d == b.eta_d && a.eta_a == b.eta_a && a.cd == b.cd;
        });
}
