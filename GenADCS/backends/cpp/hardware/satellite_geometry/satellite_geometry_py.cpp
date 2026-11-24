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

std::array<double, 9> matrix3(py::handle value) {
    auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(value);
    if (!array || array.size() != 9) throw py::value_error("inertia must contain nine numeric values");
    std::array<double, 9> result{};
    std::copy(array.data(), array.data() + 9, result.begin());
    return result;
}

py::tuple tuple3(const std::array<double, 3>& value) {
    return py::make_tuple(value[0], value[1], value[2]);
}

py::array_t<double> numpy_matrix(const std::array<double, 9>& value) {
    py::array_t<double> result({3, 3});
    std::copy(value.begin(), value.end(), result.mutable_data());
    return result;
}
}

void bind_satellite_geometry(py::module_& module) {
    py::class_<hardware::SatelliteGeometry>(module, "SatelliteGeometry")
        .def(py::init([](double mass, py::object inertia, py::object panels, py::object center) {
            std::vector<hardware::SatellitePanel> native_panels;
            if (!panels.is_none()) native_panels = panels.cast<std::vector<hardware::SatellitePanel>>();
            const auto native_center = center.is_none() ? std::array<double, 3>{0.0, 0.0, 0.0} : vector3(center);
            return hardware::SatelliteGeometry(mass, matrix3(inertia), std::move(native_panels), native_center);
        }), py::arg("mass"), py::arg("inertia"), py::arg("panels") = py::none(),
            py::arg("center_of_mass") = py::none())
        .def_readwrite("mass", &hardware::SatelliteGeometry::mass)
        .def_property("inertia", [](const hardware::SatelliteGeometry& g) { return numpy_matrix(g.inertia); },
                      [](hardware::SatelliteGeometry& g, py::object v) { g.inertia = matrix3(v); })
        .def_readwrite("panels", &hardware::SatelliteGeometry::panels)
        .def_property("center_of_mass", [](const hardware::SatelliteGeometry& g) { return tuple3(g.center_of_mass); },
                      [](hardware::SatelliteGeometry& g, py::object v) { g.center_of_mass = vector3(v); })
        .def("__eq__", [](const hardware::SatelliteGeometry& a, const hardware::SatelliteGeometry& b) {
            if (a.mass != b.mass || a.inertia != b.inertia || a.center_of_mass != b.center_of_mass
                || a.panels.size() != b.panels.size()) return false;
            for (std::size_t i = 0; i < a.panels.size(); ++i) {
                const auto& x = a.panels[i];
                const auto& y = b.panels[i];
                if (x.position != y.position || x.orientation != y.orientation
                    || x.dimensions != y.dimensions || x.eta_s != y.eta_s
                    || x.eta_d != y.eta_d || x.eta_a != y.eta_a || x.cd != y.cd) return false;
            }
            return true;
        });
}
