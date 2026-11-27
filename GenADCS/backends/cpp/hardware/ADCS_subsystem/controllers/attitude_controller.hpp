#ifndef ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_CONTROLLERS_ATTITUDE_CONTROLLER_HPP
#define ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_CONTROLLERS_ATTITUDE_CONTROLLER_HPP

#include "allocator/allocator.hpp"
#include "allocator/l2allocator.hpp"
#include "controller.hpp"
#include "desaturation/desaturation.hpp"
#include "feedback/feedback.hpp"
#include "feedforward/feedforward.hpp"
#include "feedforward/passthrough.hpp"

#include <Eigen/Core>
#include <memory>
#include <tuple>
#include <vector>

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
    Eigen::Ref<Eigen::VectorXd> dipole_commands);

class AttitudeController final : public Controller {
public:
  AttitudeController(
      std::shared_ptr<Feedback> feedback,
      std::shared_ptr<Feedforward> feedforward = std::make_shared<Passthrough>(),
      std::shared_ptr<Allocator> allocator = std::make_shared<L2Allocator>(),
      std::shared_ptr<Desaturation> desaturation =
          std::make_shared<NoDesaturation>());

  int kind() const override { return CTRL_ATTITUDE; }
  void
  validate_sensors(const std::vector<std::string> & /*sensor_types*/) const override {}
  Eigen::VectorXd to_params() const override;

  const std::shared_ptr<Feedback> &feedback() const { return feedback_; }
  const std::shared_ptr<Feedforward> &feedforward() const {
    return feedforward_;
  }
  const std::shared_ptr<Allocator> &allocator() const { return allocator_; }
  const std::shared_ptr<Desaturation> &desaturation() const {
    return desaturation_;
  }

private:
  std::shared_ptr<Feedback> feedback_;
  std::shared_ptr<Feedforward> feedforward_;
  std::shared_ptr<Allocator> allocator_;
  std::shared_ptr<Desaturation> desaturation_;
};

} // namespace omni::hardware::controllers

#endif
