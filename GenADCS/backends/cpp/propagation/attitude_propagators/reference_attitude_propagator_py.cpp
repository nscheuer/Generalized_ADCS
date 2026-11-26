#include "reference_attitude_propagator.hpp"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using omni::propagation::ReferenceAttitudeMode;
using omni::propagation::ReferenceAttitudePropagator;

void bind_reference_attitude_propagator(py::module_ &module) {
  py::class_<ReferenceAttitudePropagator>(module,
                                           "ReferenceAttitudePropagator")
      .def(py::init([](double dt, const std::string &mode) {
             return ReferenceAttitudePropagator(
                 dt, ReferenceAttitudePropagator::parse_mode(mode));
           }),
           py::arg("dt"), py::arg("mode"))
      .def_property_readonly("dt", &ReferenceAttitudePropagator::dt)
      .def_property_readonly("mode", [](const ReferenceAttitudePropagator &self) {
        return static_cast<int>(self.mode());
      })
      .def("apply",
           [](const ReferenceAttitudePropagator &self, Eigen::MatrixXd states,
              const std::vector<double> &times, double initial_julian_date,
              double mu) {
             self.apply(states, times, initial_julian_date, mu);
             return states;
           },
           py::arg("states"), py::arg("times"), py::arg("initial_julian_date"),
           py::arg("mu"))
      .def("attitude",
           [](const ReferenceAttitudePropagator &self,
              const Eigen::Vector3d &position,
              const Eigen::Vector3d &velocity, double mu,
              const Eigen::Vector3d &sun_position,
              const Eigen::Vector3d &sun_velocity) {
             const auto result = self.attitude(position, velocity, mu,
                                               sun_position, sun_velocity);
             return py::make_tuple(result.quaternion, result.angular_velocity);
           },
           py::arg("position"), py::arg("velocity"), py::arg("mu"),
           py::arg("sun_position") = Eigen::Vector3d::Zero(),
           py::arg("sun_velocity") = Eigen::Vector3d::Zero());
}
