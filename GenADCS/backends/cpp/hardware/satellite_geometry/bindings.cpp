#include "satellite_geometry.hpp"
#include "satellite_panel.hpp"

#include <pybind11/pybind11.h>

#include <utility>

namespace py = pybind11;
namespace hardware = omni::hardware;

namespace {
py::list fresh_list(const py::object& value) {
    if (value.is_none()) return py::list();
    if (PyList_Check(value.ptr()) != 0) return py::reinterpret_borrow<py::list>(value);
    return py::list(value);
}

bool same(const py::object& left, const py::object& right) {
    return py::module_::import("numpy").attr("array_equal")(left, right).cast<bool>();
}
}

PYBIND11_MODULE(_satellite_geometry, module) {
    module.doc() = "C++ satellite geometry models";

    py::class_<hardware::SatellitePanel>(module, "SatellitePanel")
        .def(py::init([](py::object position, py::object orientation,
                         py::object dimensions, double eta_s, double eta_d,
                         double eta_a, double cd) {
            return hardware::SatellitePanel(std::move(position), std::move(orientation),
                std::move(dimensions), eta_s, eta_d, eta_a, cd);
        }), py::arg("position"), py::arg("orientation"), py::arg("dimensions"),
            py::arg("eta_s") = 0.0, py::arg("eta_d") = 0.0,
            py::arg("eta_a") = 1.0, py::arg("cd") = 2.2)
        .def_readwrite("position", &hardware::SatellitePanel::position)
        .def_readwrite("orientation", &hardware::SatellitePanel::orientation)
        .def_readwrite("dimensions", &hardware::SatellitePanel::dimensions)
        .def_readwrite("eta_s", &hardware::SatellitePanel::eta_s)
        .def_readwrite("eta_d", &hardware::SatellitePanel::eta_d)
        .def_readwrite("eta_a", &hardware::SatellitePanel::eta_a)
        .def_readwrite("cd", &hardware::SatellitePanel::cd)
        .def("__eq__", [](const hardware::SatellitePanel& left,
                           const hardware::SatellitePanel& right) {
            return same(left.position, right.position)
                && same(left.orientation, right.orientation)
                && same(left.dimensions, right.dimensions)
                && (left.eta_s == right.eta_s) && (left.eta_d == right.eta_d)
                && (left.eta_a == right.eta_a) && (left.cd == right.cd);
        });

    py::class_<hardware::SatelliteGeometry>(module, "SatelliteGeometry")
        .def(py::init([](double mass, py::object inertia, py::object panels,
                         py::object center_of_mass) {
            if (center_of_mass.is_none()) center_of_mass = py::make_tuple(0.0, 0.0, 0.0);
            return hardware::SatelliteGeometry(mass, std::move(inertia),
                fresh_list(panels), std::move(center_of_mass));
        }), py::arg("mass"), py::arg("inertia"), py::arg("panels") = py::none(),
            py::arg("center_of_mass") = py::none())
        .def_readwrite("mass", &hardware::SatelliteGeometry::mass)
        .def_readwrite("inertia", &hardware::SatelliteGeometry::inertia)
        .def_readwrite("panels", &hardware::SatelliteGeometry::panels)
        .def_readwrite("center_of_mass", &hardware::SatelliteGeometry::center_of_mass)
        .def("__eq__", [](const hardware::SatelliteGeometry& left,
                           const hardware::SatelliteGeometry& right) {
            return (left.mass == right.mass) && same(left.inertia, right.inertia)
                && same(left.panels, right.panels)
                && same(left.center_of_mass, right.center_of_mass);
        });
}
