#include "controllers.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
namespace ctrl = omni::hardware::controllers;

namespace {

Eigen::Matrix3d parse_gain_matrix(const py::handle &obj,
                                  const std::string &name, bool require_spd) {
  auto arr =
      py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
          obj);
  if (!arr) {
    throw std::invalid_argument(name +
                                " must be a scalar, 3-vector, or 3x3 matrix");
  }
  if (arr.ndim() == 0) {
    return ctrl::to_gain_matrix(*arr.data(), name, require_spd);
  }
  if (arr.ndim() == 1 && arr.shape(0) == 3) {
    Eigen::Vector3d v(arr.at(0), arr.at(1), arr.at(2));
    return ctrl::to_gain_matrix(v, name, require_spd);
  }
  if (arr.ndim() == 2 && arr.shape(0) == 3 && arr.shape(1) == 3) {
    Eigen::Matrix3d m;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        m(r, c) = arr.at(r, c);
      }
    }
    return ctrl::to_gain_matrix(m, name, require_spd);
  }
  throw std::invalid_argument(name +
                              " must be a scalar, 3-vector, or 3x3 matrix");
}

std::vector<std::string> extract_sensor_types(const py::handle &sensors) {
  std::vector<std::string> names;
  for (py::handle item : sensors) {
    names.push_back(
        py::str(py::type::handle_of(item).attr("__name__")).cast<std::string>());
  }
  return names;
}

} // namespace

void bind_controllers(py::module_ &module) {
  py::class_<ctrl::Feedforward, std::shared_ptr<ctrl::Feedforward>>(
      module, "Feedforward")
      .def_property_readonly("kind", &ctrl::Feedforward::kind)
      .def("to_params", &ctrl::Feedforward::to_params);

  py::class_<ctrl::Passthrough, ctrl::Feedforward,
             std::shared_ptr<ctrl::Passthrough>>(module, "Passthrough")
      .def(py::init<>());

  py::class_<ctrl::Feedback, std::shared_ptr<ctrl::Feedback>>(module,
                                                              "Feedback")
      .def_property_readonly("kind", &ctrl::Feedback::kind)
      .def("to_matrices", &ctrl::Feedback::to_matrices);

  py::class_<ctrl::MRPFeedbackLinearized, ctrl::Feedback,
             std::shared_ptr<ctrl::MRPFeedbackLinearized>>(
      module, "MRPFeedbackLinearized")
      .def(py::init([](const py::handle &kp, const py::handle &kd) {
             auto kp_mat = parse_gain_matrix(kp, "kp", true);
             auto kd_mat = parse_gain_matrix(kd, "kd", false);
             return std::make_shared<ctrl::MRPFeedbackLinearized>(kp_mat,
                                                                  kd_mat);
           }),
           py::arg("kp"), py::arg("kd"))
      .def_property_readonly("kp", &ctrl::MRPFeedbackLinearized::kp)
      .def_property_readonly("kd", &ctrl::MRPFeedbackLinearized::kd);

  py::class_<ctrl::Desaturation, std::shared_ptr<ctrl::Desaturation>>(
      module, "Desaturation")
      .def_property_readonly("kind", &ctrl::Desaturation::kind)
      .def("to_params", &ctrl::Desaturation::to_params);

  py::class_<ctrl::NoDesaturation, ctrl::Desaturation,
             std::shared_ptr<ctrl::NoDesaturation>>(module, "NoDesaturation")
      .def(py::init<>());

  py::class_<ctrl::HoganSchaubManagement, ctrl::Desaturation,
             std::shared_ptr<ctrl::HoganSchaubManagement>>(
      module, "HoganSchaubManagement")
      .def(py::init<double, double, double>(), py::arg("gain"),
           py::arg("target_momentum") = 0.0, py::arg("mtq_gain") = 0.0)
      .def_property_readonly("gain", &ctrl::HoganSchaubManagement::gain)
      .def_property_readonly("target_momentum",
                             &ctrl::HoganSchaubManagement::target_momentum)
      .def_property_readonly("mtq_gain",
                             &ctrl::HoganSchaubManagement::mtq_gain);

  py::class_<ctrl::Allocator, std::shared_ptr<ctrl::Allocator>>(module,
                                                                "Allocator")
      .def_property_readonly("kind", &ctrl::Allocator::kind);

  py::class_<ctrl::L2Allocator, ctrl::Allocator,
             std::shared_ptr<ctrl::L2Allocator>>(module, "L2Allocator")
      .def(py::init<>());

  py::class_<ctrl::LInfinityAllocator, ctrl::Allocator,
             std::shared_ptr<ctrl::LInfinityAllocator>>(module,
                                                        "LInfinityAllocator")
      .def(py::init<>());

  py::class_<ctrl::Controller, std::shared_ptr<ctrl::Controller>>(module,
                                                                  "Controller")
      .def_property_readonly("kind", &ctrl::Controller::kind)
      .def("validate_sensors",
           [](const ctrl::Controller &self, const py::handle &sensors) {
             self.validate_sensors(extract_sensor_types(sensors));
           })
      .def("to_params", &ctrl::Controller::to_params);

  py::class_<ctrl::NoController, ctrl::Controller,
             std::shared_ptr<ctrl::NoController>>(module, "NoController")
      .def(py::init<>());

  py::class_<ctrl::BDot, ctrl::Controller, std::shared_ptr<ctrl::BDot>>(module,
                                                                        "BDot")
      .def(py::init<double>(), py::arg("gain"))
      .def_property_readonly("gain", &ctrl::BDot::gain);

  py::class_<ctrl::BCross, ctrl::Controller, std::shared_ptr<ctrl::BCross>>(
      module, "BCross")
      .def(py::init<double>(), py::arg("gain"))
      .def_property_readonly("gain", &ctrl::BCross::gain);

  py::class_<ctrl::AttitudeController, ctrl::Controller,
             std::shared_ptr<ctrl::AttitudeController>>(module,
                                                        "AttitudeController")
      .def(py::init([](std::shared_ptr<ctrl::Feedback> feedback,
                       std::shared_ptr<ctrl::Feedforward> feedforward,
                       std::shared_ptr<ctrl::Allocator> allocator,
                       std::shared_ptr<ctrl::Desaturation> desaturation) {
             return std::make_shared<ctrl::AttitudeController>(
                 std::move(feedback), std::move(feedforward),
                 std::move(allocator), std::move(desaturation));
           }),
           py::arg("feedback"), py::arg("feedforward") = nullptr,
           py::arg("allocator") = nullptr, py::arg("desaturation") = nullptr)
      .def_property_readonly("feedback", &ctrl::AttitudeController::feedback)
      .def_property_readonly("feedforward",
                             &ctrl::AttitudeController::feedforward)
      .def_property_readonly("allocator", &ctrl::AttitudeController::allocator)
      .def_property_readonly("desaturation",
                             &ctrl::AttitudeController::desaturation);

  module.def("passthrough_step", &ctrl::passthrough_step, py::arg("q_goal"),
             py::arg("w_goal"), py::arg("q_current"));
  module.def("mrp_feedback_linearized_step",
             &ctrl::mrp_feedback_linearized_step, py::arg("q"),
             py::arg("omega"), py::arg("q_d"), py::arg("omega_d"),
             py::arg("omega_d_dot"), py::arg("ff_torque"),
             py::arg("wheel_momenta"), py::arg("total_inertia"),
             py::arg("bus_inertia"), py::arg("wheel_axes"), py::arg("kp"),
             py::arg("kd"));
  module.def("hogan_schaub_u0", &ctrl::hogan_schaub_u0,
             py::arg("wheel_momenta"), py::arg("gain"),
             py::arg("target_momentum"));
  module.def("allocate_l2", &ctrl::allocate_l2, py::arg("tau_c"),
             py::arg("a_pinv"), py::arg("null_proj"), py::arg("u_0"),
             py::arg("max_torque"));
  module.def("allocate_linfinity", &ctrl::allocate_linfinity, py::arg("tau_c"),
             py::arg("a_pinv"), py::arg("null_basis"), py::arg("null_proj"),
             py::arg("u_0"), py::arg("max_torque"));
}
