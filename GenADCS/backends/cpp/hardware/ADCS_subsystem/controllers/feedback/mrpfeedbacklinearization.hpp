#ifndef ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_CONTROLLERS_FEEDBACK_MRPFEEDBACKLINEARIZATION_HPP
#define ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_CONTROLLERS_FEEDBACK_MRPFEEDBACKLINEARIZATION_HPP

#include "feedback.hpp"

#include <Eigen/Core>

namespace omni::hardware::controllers {

Eigen::Vector3d mrp_feedback_linearized_step(
    const Eigen::Vector4d &q, const Eigen::Vector3d &omega,
    const Eigen::Vector4d &q_d, const Eigen::Vector3d &omega_d,
    const Eigen::Vector3d &omega_d_dot, const Eigen::Vector3d &ff_torque,
    const Eigen::Ref<const Eigen::VectorXd> &wheel_momenta,
    const Eigen::Matrix3d &total_inertia, const Eigen::Matrix3d &bus_inertia,
    const Eigen::Ref<const Eigen::MatrixXd> &wheel_axes,
    const Eigen::Matrix3d &kp, const Eigen::Matrix3d &kd);

class MRPFeedbackLinearized final : public Feedback {
public:
  MRPFeedbackLinearized(double kp, double kd);
  MRPFeedbackLinearized(double kp, const Eigen::Vector3d &kd);
  MRPFeedbackLinearized(double kp, const Eigen::Matrix3d &kd);
  MRPFeedbackLinearized(const Eigen::Vector3d &kp, double kd);
  MRPFeedbackLinearized(const Eigen::Vector3d &kp, const Eigen::Vector3d &kd);
  MRPFeedbackLinearized(const Eigen::Vector3d &kp, const Eigen::Matrix3d &kd);
  MRPFeedbackLinearized(const Eigen::Matrix3d &kp, double kd);
  MRPFeedbackLinearized(const Eigen::Matrix3d &kp, const Eigen::Vector3d &kd);
  MRPFeedbackLinearized(const Eigen::Matrix3d &kp, const Eigen::Matrix3d &kd);

  int kind() const override { return FB_MRP_FEEDBACK_LINEARIZED; }
  std::pair<Eigen::Matrix3d, Eigen::Matrix3d> to_matrices() const override {
    return {kp_, kd_};
  }

  const Eigen::Matrix3d &kp() const { return kp_; }
  const Eigen::Matrix3d &kd() const { return kd_; }

private:
  Eigen::Matrix3d kp_;
  Eigen::Matrix3d kd_;
};

} // namespace omni::hardware::controllers

#endif
