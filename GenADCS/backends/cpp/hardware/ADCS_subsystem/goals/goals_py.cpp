#include "goals.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace py = pybind11;
namespace goals = omni::hardware::goals;

namespace {

Eigen::VectorXd to_state_vector(const py::handle &state) {
  auto arr =
      py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
          state);
  if (!arr || arr.ndim() != 1) {
    throw std::invalid_argument(
        "goal state must have a finite 13-element spacecraft-state prefix");
  }
  return Eigen::Map<const Eigen::VectorXd>(arr.data(), arr.size());
}

Eigen::Vector4d to_vec4(const py::handle &value, const char *err_msg) {
  auto arr =
      py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
          value);
  if (!arr || arr.ndim() != 1 || arr.shape(0) != 4) {
    throw std::invalid_argument(err_msg);
  }
  return Eigen::Map<const Eigen::Vector4d>(arr.data());
}

Eigen::Vector3d to_vec3(const py::handle &value, const char *err_msg) {
  auto arr =
      py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
          value);
  if (!arr || arr.ndim() != 1 || arr.shape(0) != 3) {
    throw std::invalid_argument(err_msg);
  }
  return Eigen::Map<const Eigen::Vector3d>(arr.data());
}

Eigen::Matrix<double, 2, 3> to_mat2x3(const py::handle &value,
                                      const char *err_msg) {
  auto arr =
      py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
          value);
  if (!arr || arr.ndim() != 2 || arr.shape(0) != 2 || arr.shape(1) != 3) {
    throw std::invalid_argument(err_msg);
  }
  Eigen::Matrix<double, 2, 3> mat;
  for (int r = 0; r < 2; ++r) {
    for (int c = 0; c < 3; ++c) {
      mat(r, c) = arr.at(r, c);
    }
  }
  return mat;
}

goals::GoalRuntime to_runtime(const py::handle &runtime) {
  goals::GoalRuntime rt;
  if (!runtime.is_none()) {
    rt.mu = runtime.attr("mu").cast<double>();
  }
  return rt;
}

py::tuple call_reference(const goals::Goal &goal, const py::handle &state,
                         double epoch_jd, const py::handle &runtime) {
  const Eigen::VectorXd state_vec = to_state_vector(state);
  const goals::GoalRuntime rt = to_runtime(runtime);
  const goals::GoalReference ref = goal.reference(state_vec, epoch_jd, rt);

  py::object q_obj = py::none();
  if (ref.first.has_value()) {
    q_obj = py::array_t<double>(4, ref.first->data());
  }
  py::object w_obj = py::none();
  if (ref.second.has_value()) {
    w_obj = py::array_t<double>(3, ref.second->data());
  }
  return py::make_tuple(std::move(q_obj), std::move(w_obj));
}

template <typename Vec> py::array_t<double> to_numpy(const Vec &v) {
  return py::array_t<double>(v.size(), v.data());
}

} // namespace

void bind_goals(py::module_ &module) {
  py::class_<goals::Goal, std::shared_ptr<goals::Goal>>(module, "Goal")
      .def(
          "reference",
          [](const goals::Goal &self, py::object state, double epoch_jd,
             py::object runtime) {
            return call_reference(self, state, epoch_jd, runtime);
          },
          py::arg("state"), py::arg("epoch_jd"), py::arg("runtime") = py::none())
      .def(
          "__call__",
          [](const goals::Goal &self, py::object state, double epoch_jd,
             py::object runtime) {
            return call_reference(self, state, epoch_jd, runtime);
          },
          py::arg("state"), py::arg("epoch_jd"),
          py::arg("runtime") = py::none());

  py::class_<goals::NoGoal, goals::Goal, std::shared_ptr<goals::NoGoal>>(
      module, "NoGoal")
      .def(py::init<>());

  py::class_<goals::QuaternionGoal, goals::Goal,
             std::shared_ptr<goals::QuaternionGoal>>(module, "QuaternionGoal")
      .def(py::init([](py::object quaternion, py::object rate) {
             const Eigen::Vector4d q = to_vec4(
                 quaternion, "quaternion must be a finite 4-element vector");
             std::optional<Eigen::Vector3d> r;
             if (!rate.is_none()) {
               r = to_vec3(rate, "rate must be a finite 3-element vector");
             }
             return goals::QuaternionGoal(q, r);
           }),
           py::arg("quaternion"), py::arg("rate") = py::none())
      .def_property_readonly(
          "quaternion",
          [](const goals::QuaternionGoal &self) {
            return to_numpy(self.quaternion());
          })
      .def_property_readonly("rate", [](const goals::QuaternionGoal &self) {
        return to_numpy(self.rate());
      });

  py::class_<goals::AngularRateGoal, goals::Goal,
             std::shared_ptr<goals::AngularRateGoal>>(module, "AngularRateGoal")
      .def(py::init([](py::object rate) {
             return goals::AngularRateGoal(
                 to_vec3(rate, "rate must be a finite 3-element vector"));
           }),
           py::arg("rate"))
      .def_property_readonly("rate", [](const goals::AngularRateGoal &self) {
        return to_numpy(self.rate());
      });

  py::class_<goals::VectorGoal, goals::Goal,
             std::shared_ptr<goals::VectorGoal>>(module, "VectorGoal")
      .def(py::init([](py::object inertial_vector, py::object body_axis,
                       py::object inertial_roll, py::object body_roll) {
             return goals::VectorGoal(
                 to_vec3(inertial_vector,
                         "inertial_vector must be a finite 3-element vector"),
                 to_vec3(body_axis,
                         "body_axis must be a finite 3-element vector"),
                 to_vec3(inertial_roll,
                         "inertial_roll must be a finite 3-element vector"),
                 to_vec3(body_roll,
                         "body_roll must be a finite 3-element vector"));
           }),
           py::arg("inertial_vector"),
           py::arg("body_axis") = py::make_tuple(0.0, 0.0, 1.0), py::kw_only(),
           py::arg("inertial_roll") = py::make_tuple(1.0, 0.0, 0.0),
           py::arg("body_roll") = py::make_tuple(1.0, 0.0, 0.0))
      .def_property_readonly("inertial_vector",
                             [](const goals::VectorGoal &self) {
                               return to_numpy(self.inertial_vector());
                             })
      .def_property_readonly("body_axis",
                             [](const goals::VectorGoal &self) {
                               return to_numpy(self.body_axis());
                             })
      .def_property_readonly("inertial_roll",
                             [](const goals::VectorGoal &self) {
                               return to_numpy(self.inertial_roll());
                             })
      .def_property_readonly("body_roll", [](const goals::VectorGoal &self) {
        return to_numpy(self.body_roll());
      });

  py::class_<goals::TwoVectorGoal, goals::Goal,
             std::shared_ptr<goals::TwoVectorGoal>>(module, "TwoVectorGoal")
      .def(py::init([](py::object inertial_vectors, py::object body_vectors) {
             const char *msg =
                 "inertial_vectors and body_vectors must each have shape (2, 3)";
             return goals::TwoVectorGoal(to_mat2x3(inertial_vectors, msg),
                                         to_mat2x3(body_vectors, msg));
           }),
           py::arg("inertial_vectors"), py::arg("body_vectors"))
      .def_property_readonly("inertial_first",
                             [](const goals::TwoVectorGoal &self) {
                               return to_numpy(self.inertial_first());
                             })
      .def_property_readonly("inertial_second",
                             [](const goals::TwoVectorGoal &self) {
                               return to_numpy(self.inertial_second());
                             })
      .def_property_readonly("body_first",
                             [](const goals::TwoVectorGoal &self) {
                               return to_numpy(self.body_first());
                             })
      .def_property_readonly("body_second",
                             [](const goals::TwoVectorGoal &self) {
                               return to_numpy(self.body_second());
                             });

  py::class_<goals::OrbitGoal, goals::Goal, std::shared_ptr<goals::OrbitGoal>>(
      module, "OrbitGoal");

  py::class_<goals::NadirGoal, goals::OrbitGoal,
             std::shared_ptr<goals::NadirGoal>>(module, "NadirGoal")
      .def(py::init<>());

  py::class_<goals::ZenithGoal, goals::OrbitGoal,
             std::shared_ptr<goals::ZenithGoal>>(module, "ZenithGoal")
      .def(py::init<>());

  py::class_<goals::AlongTrackGoal, goals::OrbitGoal,
             std::shared_ptr<goals::AlongTrackGoal>>(module, "AlongTrackGoal")
      .def(py::init<>());

  py::class_<goals::CrossTrackGoal, goals::OrbitGoal,
             std::shared_ptr<goals::CrossTrackGoal>>(module, "CrossTrackGoal")
      .def(py::init<>());

  py::class_<goals::VelocityGoal, goals::OrbitGoal,
             std::shared_ptr<goals::VelocityGoal>>(module, "VelocityGoal")
      .def(py::init<>());

  py::class_<goals::SunGoal, goals::OrbitGoal,
             std::shared_ptr<goals::SunGoal>>(module, "SunGoal")
      .def(py::init<>());

  py::class_<goals::CoordinateGoal, goals::Goal,
             std::shared_ptr<goals::CoordinateGoal>>(module, "CoordinateGoal")
      .def(py::init<double, double, double>(), py::arg("lat"), py::arg("lon"),
           py::arg("alt") = 0.0)
      .def_property_readonly("lat_deg", &goals::CoordinateGoal::lat_deg)
      .def_property_readonly("lon_deg", &goals::CoordinateGoal::lon_deg)
      .def_property_readonly("alt_km", &goals::CoordinateGoal::alt_km)
      .def_property_readonly("target_ecef_m",
                             [](const goals::CoordinateGoal &self) {
                               return to_numpy(self.target_ecef_m());
                             });

  py::class_<goals::CoordinateFMCGoal, goals::CoordinateGoal,
             std::shared_ptr<goals::CoordinateFMCGoal>>(module,
                                                        "CoordinateFMCGoal")
      .def(py::init([](double lat, double lon, double alt,
                       py::object fmc_level) {
             if (PyBool_Check(fmc_level.ptr()) != 0 ||
                 PyIndex_Check(fmc_level.ptr()) == 0) {
               throw std::invalid_argument(
                   "fmc_level must be an integer from 1 through 10");
             }
             const int level = fmc_level.cast<int>();
             return goals::CoordinateFMCGoal(lat, lon, alt, level);
           }),
           py::arg("lat"), py::arg("lon"), py::arg("alt") = 0.0,
           py::arg("fmc_level") = 1)
      .def_property_readonly("fmc_level",
                             &goals::CoordinateFMCGoal::fmc_level);
}
