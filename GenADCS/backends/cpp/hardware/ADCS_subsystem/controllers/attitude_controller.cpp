#include "attitude_controller.hpp"

#include "allocator/l2allocator.hpp"
#include "allocator/linftyallocator.hpp"
#include "desaturation/hoganschaubdesaturation.hpp"
#include "feedback/mrpfeedbacklinearization.hpp"
#include "feedforward/passthrough.hpp"

#include <stdexcept>
#include <utility>

namespace omni::hardware::controllers {

std::tuple<Eigen::Vector3d, Eigen::Vector4d, Eigen::Vector3d, Eigen::Vector3d>
attitude_controller_step(
    const Eigen::Vector4d &q, const Eigen::Vector3d &omega,
    const Eigen::Vector4d &q_goal, const Eigen::Vector3d &w_goal,
    const Eigen::Ref<const Eigen::VectorXd> &wheel_momenta,
    const Eigen::Matrix3d &total_inertia, const Eigen::Matrix3d &bus_inertia,
    const Eigen::Ref<const Eigen::MatrixXd> &wheel_axes,
    const Eigen::Ref<const Eigen::VectorXd> &wheel_max_torque,
    const Eigen::Ref<const Eigen::MatrixXd> &a_pinv,
    const Eigen::Ref<const Eigen::MatrixXd> &null_basis,
    const Eigen::Ref<const Eigen::MatrixXd> &null_proj, int ff_kind,
    int fb_kind, int alloc_kind, int desat_kind, const Eigen::Matrix3d &kp,
    const Eigen::Matrix3d &kd, double desat_gain, double desat_target,
    double desat_mtq_gain, const Eigen::Ref<const Eigen::VectorXd> &readings,
    const SensorConfig &sensor_config,
    const Eigen::Ref<const Eigen::MatrixXd> &mtq_axes,
    const Eigen::Ref<const Eigen::VectorXd> &mtq_max_dipole,
    Eigen::Ref<Eigen::VectorXd> wheel_commands,
    Eigen::Ref<Eigen::VectorXd> dipole_commands) {
  // 1. Feedforward
  Eigen::Vector4d q_d;
  Eigen::Vector3d w_d;
  Eigen::Vector3d w_d_dot;
  Eigen::Vector3d ff_torque;
  if (ff_kind == FF_PASSTHROUGH) {
    std::tie(q_d, w_d, w_d_dot, ff_torque) =
        passthrough_step(q_goal, w_goal, q);
  } else {
    q_d = q;
    w_d.setZero();
    w_d_dot.setZero();
    ff_torque.setZero();
  }

  // 2. Feedback
  Eigen::Vector3d tau_c;
  if (fb_kind == FB_MRP_FEEDBACK_LINEARIZED) {
    tau_c = mrp_feedback_linearized_step(
        q, omega, q_d, w_d, w_d_dot, ff_torque, wheel_momenta, total_inertia,
        bus_inertia, wheel_axes, kp, kd);
  } else {
    tau_c = ff_torque;
  }

  if (wheel_commands.size() == 0 ||
      (wheel_max_torque.array() <= 0.0).all()) {
    wheel_commands.setZero();
    dipole_commands.setZero();
    return {ff_torque, q_d, w_d, tau_c};
  }

  // 3. Desaturation nullspace vector u_0 + optional magnetorquer momentum dump
  Eigen::VectorXd u_0;
  if (desat_kind == DESAT_HOGAN_SCHAUB) {
    u_0 = hogan_schaub_u0(wheel_momenta, desat_gain, desat_target);
    hogan_schaub_mtq_dump(wheel_momenta, wheel_axes, desat_mtq_gain,
                          desat_target, readings, sensor_config, mtq_axes,
                          mtq_max_dipole, dipole_commands);
  } else {
    u_0 = Eigen::VectorXd::Zero(wheel_commands.size());
    dipole_commands.setZero();
  }

  // 4. Allocator
  Eigen::VectorXd u_rw;
  if (alloc_kind == ALLOC_L2) {
    u_rw = allocate_l2(tau_c, a_pinv, null_proj, u_0, wheel_max_torque);
  } else {
    u_rw = allocate_linfinity(tau_c, a_pinv, null_basis, null_proj, u_0,
                              wheel_max_torque);
  }
  wheel_commands = u_rw;
  return {ff_torque, q_d, w_d, tau_c};
}

AttitudeController::AttitudeController(
    std::shared_ptr<Feedback> feedback,
    std::shared_ptr<Feedforward> feedforward,
    std::shared_ptr<Allocator> allocator,
    std::shared_ptr<Desaturation> desaturation)
    : feedback_(std::move(feedback)), feedforward_(std::move(feedforward)),
      allocator_(std::move(allocator)),
      desaturation_(std::move(desaturation)) {
  if (!feedback_) {
    throw std::invalid_argument("feedback must be a Feedback instance");
  }
  if (!feedforward_) {
    feedforward_ = std::make_shared<Passthrough>();
  }
  if (!allocator_) {
    allocator_ = std::make_shared<L2Allocator>();
  }
  if (!desaturation_) {
    desaturation_ = std::make_shared<NoDesaturation>();
  }
}

Eigen::VectorXd AttitudeController::to_params() const {
  const Eigen::VectorXd desat_p = desaturation_->to_params();
  Eigen::VectorXd p(7);
  p << static_cast<double>(feedforward_->kind()),
      static_cast<double>(feedback_->kind()),
      static_cast<double>(allocator_->kind()),
      static_cast<double>(desaturation_->kind()), desat_p[0], desat_p[1],
      desat_p[2];
  return p;
}

} // namespace omni::hardware::controllers
