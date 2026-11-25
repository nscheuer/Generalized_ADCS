#include "satellite.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace {
template <std::size_t N> std::array<double, N> values(py::handle object) {
  auto array =
      py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
          object);
  if (!array || array.size() != N)
    throw py::value_error("incorrect geometry array size");
  std::array<double, N> result{};
  auto data = array.data();
  std::copy(data, data + N, result.begin());
  return result;
}
} // namespace

PYBIND11_MODULE(_satellite, module) {
  // Ensure the SatelliteGeometry caster is registered before exposing
  // Satellite.
  py::module_::import("GenADCS.backends.cpp.hardware.satellite_geometry");

  py::class_<omni::hardware::Satellite>(module, "Satellite", py::dynamic_attr())
      .def(py::init([](py::object geometry, std::string mode) {
             std::vector<omni::hardware::SatellitePanel> panels;
             for (auto panel : geometry.attr("panels")) {
               panels.emplace_back(values<3>(panel.attr("position")),
                                   values<3>(panel.attr("orientation")),
                                   values<2>(panel.attr("dimensions")),
                                   panel.attr("eta_s").cast<double>(),
                                   panel.attr("eta_d").cast<double>(),
                                   panel.attr("eta_a").cast<double>(),
                                   panel.attr("cd").cast<double>());
             }
             auto native = omni::hardware::SatelliteGeometry(
                 geometry.attr("mass").cast<double>(),
                 values<9>(geometry.attr("inertia")), std::move(panels),
                 values<3>(geometry.attr("center_of_mass")));
             return omni::hardware::Satellite(std::move(native),
                                              std::move(mode));
           }),
           py::arg("geometry"), py::arg("initial_mode") = "idle")
      .def_readwrite("geometry", &omni::hardware::Satellite::geometry)
      .def_readwrite("mode", &omni::hardware::Satellite::mode);
}
