#include "base.hpp"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace omni::hardware::estimator;
void bind_estimator_base(py::module_ &m) {
  py::class_<State>(m, "State")
      .def(py::init<Eigen::Vector4d, Eigen::Vector3d, Matrix6>())
      .def_readwrite("quaternion", &State::quaternion)
      .def_readwrite("rate", &State::rate)
      .def_readwrite("covariance", &State::covariance);
  py::class_<AttitudeEstimator>(m, "AttitudeEstimator")
      .def(py::init([](Eigen::Vector4d q, Eigen::Vector3d w, Matrix6 p,
                       py::object a, std::set<std::string> record) {
             Eigen::Matrix3d psd;
             if (py::isinstance<py::float_>(a) || py::isinstance<py::int_>(a)) {
               psd = Eigen::Matrix3d::Identity() * a.cast<double>();
             } else {
               psd = a.cast<Eigen::Matrix3d>();
}
             return AttitudeEstimator(q, w, p, psd, record);
           }),
           py::arg("quaternion"), py::arg("rate"), py::arg("covariance"),
           py::arg("acceleration_psd"), py::kw_only(),
           py::arg("record") = std::set<std::string>{})
      .def_readonly("quaternion", &AttitudeEstimator::quaternion)
      .def_readonly("rate", &AttitudeEstimator::rate)
      .def_readonly("covariance", &AttitudeEstimator::covariance)
      .def_readonly("acceleration_psd", &AttitudeEstimator::acceleration_psd)
      .def_readonly("record", &AttitudeEstimator::record)
      .def("prepare", [](const AttitudeEstimator &self) {
        auto s = self.prepare();
        return py::make_tuple(s.quaternion, s.rate, s.covariance,
                              self.acceleration_psd);
      });
}
