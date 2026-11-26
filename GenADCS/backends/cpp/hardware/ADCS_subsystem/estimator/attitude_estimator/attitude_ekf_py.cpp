#include "attitude_ekf.hpp"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace omni::hardware::estimator;

void bind_estimator_ekf(py::module_ &m) {
  m.def("sample_ekf", &AttitudeEKF::sample,
        py::call_guard<py::gil_scoped_release>());
  py::class_<AttitudeEKF, AttitudeEstimator>(m, "AttitudeEKF")
      .def(py::init([](Eigen::Vector4d q, Eigen::Vector3d w, Matrix7 p,
                       py::object acceleration_psd,
                       std::set<std::string> record) {
             Eigen::Matrix3d psd;
             if (py::isinstance<py::float_>(acceleration_psd) ||
                 py::isinstance<py::int_>(acceleration_psd)) {
               psd = Eigen::Matrix3d::Identity() *
                     acceleration_psd.cast<double>();
             } else {
               psd = acceleration_psd.cast<Eigen::Matrix3d>();
             }
             return AttitudeEKF(q, w, p, psd, record);
           }),
           py::arg("quaternion"), py::arg("rate"), py::arg("covariance"),
           py::arg("acceleration_psd"), py::kw_only(),
           py::arg("record") = std::set<std::string>{});
  m.def("ekf_predict", [](Eigen::Vector4d q, Eigen::Vector3d w, Matrix7 p,
                          Eigen::Matrix3d acceleration_psd, double dt) {
    const State state{q, w, p};
    const State result = AttitudeEKF::predict(state, acceleration_psd, dt);
    return py::make_tuple(result.quaternion, result.rate, result.covariance);
  });
  m.def("ekf_correct", [](Eigen::Vector4d q, Eigen::Vector3d w, Matrix7 p,
                          Eigen::Vector3d residual,
                          Eigen::Matrix<double, 3, 7> h,
                          Eigen::Matrix3d covariance) {
    const auto result =
        AttitudeEKF::correct({q, w, p}, residual, h, covariance).state;
    return py::make_tuple(result.quaternion, result.rate, result.covariance);
  });
  m.def("ekf_correct_gyro", [](Eigen::Vector4d q, Eigen::Vector3d w, Matrix7 p,
                               Eigen::Vector3d measured_rate,
                               Eigen::Matrix3d mounting,
                               Eigen::Matrix3d covariance) {
    const auto result = AttitudeEKF::correct_gyro({q, w, p}, measured_rate,
                                                  mounting, covariance)
                            .state;
    return py::make_tuple(result.quaternion, result.rate, result.covariance);
  });
  m.def("ekf_correct_star_tracker", [](Eigen::Vector4d q, Eigen::Vector3d w,
                                       Matrix7 p, Eigen::Vector4d measured,
                                       Eigen::Vector4d mounting,
                                       Eigen::Matrix3d covariance) {
    const auto result = AttitudeEKF::correct_star_tracker({q, w, p}, measured,
                                                          mounting, covariance)
                            .state;
    return py::make_tuple(result.quaternion, result.rate, result.covariance);
  });
  m.def("ekf_step", [](Eigen::Vector4d q, Eigen::Vector3d w, Matrix7 p,
                       Eigen::Matrix3d acceleration_psd, double dt,
                       const Eigen::VectorXd &readings,
                       const std::vector<Sensor> &sensors) {
    const State result =
        AttitudeEKF::step({q, w, p}, acceleration_psd, dt, readings, sensors);
    return py::make_tuple(result.quaternion, result.rate, result.covariance);
  });
}
