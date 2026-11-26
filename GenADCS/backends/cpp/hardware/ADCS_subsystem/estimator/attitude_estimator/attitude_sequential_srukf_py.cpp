#include "attitude_sequential_srukf.hpp"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace omni::hardware::estimator;

void bind_estimator_sequential_srukf(py::module_ &m) {
  m.def("sample_srukf", &AttitudeSequentialSRUKF::sample,
        py::call_guard<py::gil_scoped_release>());
  py::class_<AttitudeSequentialSRUKF, AttitudeEstimator>(
      m, "AttitudeSequentialSRUKF")
      .def(py::init([](Eigen::Vector4d q, Eigen::Vector3d w, Matrix6 p,
                       py::object acceleration_psd, double alpha, double beta,
                       double kappa, std::set<std::string> record) {
             Eigen::Matrix3d psd;
             if (py::isinstance<py::float_>(acceleration_psd) ||
                 py::isinstance<py::int_>(acceleration_psd)) {
               psd = Eigen::Matrix3d::Identity() *
                     acceleration_psd.cast<double>();
             } else {
               psd = acceleration_psd.cast<Eigen::Matrix3d>();
             }
             return AttitudeSequentialSRUKF(q, w, p, psd, alpha, beta, kappa,
                                            record);
           }),
           py::arg("quaternion"), py::arg("rate"), py::arg("covariance"),
           py::arg("acceleration_psd"), py::kw_only(), py::arg("alpha") = 1.0,
           py::arg("beta") = 2.0, py::arg("kappa") = 0.0,
           py::arg("record") = std::set<std::string>{})
      .def_property_readonly_static(
          "COVARIANCE_SIZE",
          [](py::object) { return AttitudeSequentialSRUKF::COVARIANCE_SIZE; })
      .def_readonly("alpha", &AttitudeSequentialSRUKF::alpha)
      .def_readonly("beta", &AttitudeSequentialSRUKF::beta)
      .def_readonly("kappa", &AttitudeSequentialSRUKF::kappa)
      .def_readonly("ukf_params", &AttitudeSequentialSRUKF::ukf_params);
  m.def(
      "srukf_predict",
      [](Eigen::Vector4d q, Eigen::Vector3d w, Matrix6 p,
         Eigen::Matrix3d acceleration_psd, double dt, double alpha, double beta,
         double kappa) {
        const State result = AttitudeSequentialSRUKF::predict(
            {q, w, p}, acceleration_psd, dt, alpha, beta, kappa);
        return py::make_tuple(result.quaternion, result.rate, result.covariance);
      },
      py::arg("q"), py::arg("w"), py::arg("p"), py::arg("acceleration_psd"),
      py::arg("dt"), py::arg("alpha") = 1.0, py::arg("beta") = 2.0,
      py::arg("kappa") = 0.0);
  m.def(
      "srukf_step",
      [](Eigen::Vector4d q, Eigen::Vector3d w, Matrix6 p,
         Eigen::Matrix3d acceleration_psd, double dt,
         const Eigen::VectorXd &readings, const std::vector<Sensor> &sensors,
         double alpha, double beta, double kappa) {
        const State result = AttitudeSequentialSRUKF::step(
            {q, w, p}, acceleration_psd, dt, readings, sensors, alpha, beta,
            kappa);
        return py::make_tuple(result.quaternion, result.rate, result.covariance);
      },
      py::arg("q"), py::arg("w"), py::arg("p"), py::arg("acceleration_psd"),
      py::arg("dt"), py::arg("readings"), py::arg("sensors"),
      py::arg("alpha") = 1.0, py::arg("beta") = 2.0, py::arg("kappa") = 0.0);
}
