#include "check.hpp"
#include "hardware/ADCS_subsystem/controllers/feedback/mrpfeedbacklinearization.hpp"

#include <Eigen/Core>
#include <cmath>
#include <stdexcept>

using namespace omni::hardware::controllers;

int main() {
  MRPFeedbackLinearized fb(0.5, Eigen::Vector3d(1.0, 1.2, 1.4));
  const auto [kp, kd] = fb.to_matrices();
  CHECK(kp.rows() == 3 && kp.cols() == 3);
  CHECK(kd.rows() == 3 && kd.cols() == 3);

  bool threw_kp = false;
  try {
    MRPFeedbackLinearized bad_kp(0.0, 1.0);
  } catch (const std::invalid_argument &) {
    threw_kp = true;
  }
  CHECK(threw_kp);

  bool threw_kd = false;
  try {
    MRPFeedbackLinearized bad_kd(0.5, -0.1);
  } catch (const std::invalid_argument &) {
    threw_kd = true;
  }
  CHECK(threw_kd);

  const double pi = std::acos(-1.0);
  const Eigen::Vector4d q(std::cos(pi / 6.0), std::sin(pi / 6.0), 0.0, 0.0);
  const Eigen::Vector3d w = Eigen::Vector3d::Zero();
  const Eigen::Vector4d qd(1.0, 0.0, 0.0, 0.0);
  const Eigen::Vector3d wd = Eigen::Vector3d::Zero();
  const Eigen::Matrix3d inertia = Eigen::Vector3d(0.15, 0.12, 0.08).asDiagonal();
  const Eigen::Vector3d tau = mrp_feedback_linearized_step(
      q, w, qd, wd, wd, wd, Eigen::Vector3d::Zero(), inertia, inertia,
      Eigen::Matrix3d::Identity(), 0.5 * Eigen::Matrix3d::Identity(),
      1.4 * Eigen::Matrix3d::Identity());
  CHECK(tau[0] > 0.0);
  CHECK(tau.tail<2>().isApprox(Eigen::Vector2d::Zero(), 1e-12));
  return 0;
}
