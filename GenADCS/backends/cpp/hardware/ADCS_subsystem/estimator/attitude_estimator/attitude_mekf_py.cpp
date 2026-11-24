#include "attitude_mekf.hpp"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace omni::hardware::estimator;
void bind_estimator_mekf(py::module_& m) {
    py::class_<Sensor>(m, "Sensor")
        .def(py::init<>())
        .def_readwrite("code", &Sensor::code)
        .def_readwrite("orientation", &Sensor::orientation)
        .def_readwrite("misalignment", &Sensor::misalignment)
        .def_readwrite("bias", &Sensor::bias)
        .def_readwrite("noise_cholesky", &Sensor::noise_cholesky)
        .def_readwrite("enabled", &Sensor::enabled)
        .def_readwrite("offset", &Sensor::offset);
    py::class_<AttitudeMEKF, AttitudeEstimator>(m, "AttitudeMEKF")
        .def(py::init([](Eigen::Vector4d q, Eigen::Vector3d w, Matrix6 p, py::object a, std::set<std::string> record) {
            Eigen::Matrix3d psd;
            if (py::isinstance<py::float_>(a) || py::isinstance<py::int_>(a)) psd=Eigen::Matrix3d::Identity()*a.cast<double>();
            else psd=a.cast<Eigen::Matrix3d>();
            return AttitudeMEKF(q,w,p,psd,record);
        }), py::arg("quaternion"), py::arg("rate"), py::arg("covariance"), py::arg("acceleration_psd"), py::kw_only(), py::arg("record")=std::set<std::string>{});
    m.def("predict", [](Eigen::Vector4d q,Eigen::Vector3d w,Matrix6 p,Eigen::Matrix3d a,double dt) {
        auto s=AttitudeMEKF::predict({q,w,p},a,dt); return py::make_tuple(s.quaternion,s.rate,s.covariance);
    });
    m.def("correct", [](Eigen::Vector4d q,Eigen::Vector3d w,Matrix6 p,Eigen::Vector3d residual,Eigen::Matrix<double,3,6> h,Eigen::Matrix3d r) {
        auto s=AttitudeMEKF::correct({q,w,p},residual,h,r).state; return py::make_tuple(s.quaternion,s.rate,s.covariance);
    });
    m.def("correct_gyro", [](Eigen::Vector4d q,Eigen::Vector3d w,Matrix6 p,Eigen::Vector3d measured,Eigen::Matrix3d mounting,Eigen::Matrix3d r) {
        auto s=AttitudeMEKF::correct_gyro({q,w,p},measured,mounting,r).state; return py::make_tuple(s.quaternion,s.rate,s.covariance);
    });
    m.def("correct_star_tracker", [](Eigen::Vector4d q,Eigen::Vector3d w,Matrix6 p,Eigen::Vector4d measured,Eigen::Vector4d mounting,Eigen::Matrix3d r) {
        auto s=AttitudeMEKF::correct_star_tracker({q,w,p},measured,mounting,r).state; return py::make_tuple(s.quaternion,s.rate,s.covariance);
    });
    m.def("step", [](Eigen::Vector4d q,Eigen::Vector3d w,Matrix6 p,Eigen::Matrix3d a,double dt,Eigen::VectorXd readings,std::vector<Sensor> sensors) {
        auto s=AttitudeMEKF::step({q,w,p},a,dt,readings,sensors); return py::make_tuple(s.quaternion,s.rate,s.covariance);
    });
}
