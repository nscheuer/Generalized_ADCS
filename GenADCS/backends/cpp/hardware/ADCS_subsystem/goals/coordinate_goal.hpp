#ifndef ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_GOALS_COORDINATE_GOAL_HPP
#define ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_GOALS_COORDINATE_GOAL_HPP

#include "goal.hpp"
#include "helpers/math/rotations/rotation_types.hpp"

#include <Eigen/Geometry>
#include <stdexcept>

namespace omni::hardware::goals {

class CoordinateGoal : public Goal {
public:
  CoordinateGoal(double lat, double lon, double alt = 0.0);

  GoalReference reference(const Eigen::Ref<const Eigen::VectorXd> &state,
                          double epoch_jd,
                          const GoalRuntime &runtime = {}) const override;

  double lat_deg() const { return lat_deg_; }
  double lon_deg() const { return lon_deg_; }
  double alt_km() const { return alt_km_; }
  const Vector3 &target_ecef_m() const { return target_ecef_m_; }

  static Vector3 geodetic_to_ecef(double lat_deg, double lon_deg, double alt_km);

private:
  double lat_deg_;
  double lon_deg_;
  double alt_km_;
  Vector3 target_ecef_m_;
};

namespace detail {

inline Eigen::Matrix3d build_triad_dcm(const Vector3 &z_axis,
                                       const Vector3 &x_raw) {
  const Vector3 x_axis_init = x_raw.normalized();
  Vector3 y_axis = z_axis.cross(x_axis_init);
  y_axis.normalize();
  const Vector3 x_axis = y_axis.cross(z_axis);

  Eigen::Matrix3d inertial_to_body;
  inertial_to_body.row(0) = x_axis;
  inertial_to_body.row(1) = y_axis;
  inertial_to_body.row(2) = z_axis;
  return inertial_to_body;
}

template <typename DcmFn>
GoalReference central_difference_reference(
    const Eigen::Ref<const Eigen::VectorXd> &state, double epoch_jd, double mu,
    DcmFn &&dcm_fn, const char *zero_radius_msg) {
  constexpr double kSecondsPerDay = 86400.0;
  const Vector3 position = state.segment<3>(7);
  const Vector3 velocity = state.segment<3>(10);
  const Eigen::Matrix3d dcm = dcm_fn(position, velocity, epoch_jd);
  const Quaternion quaternion = omni::rotations::dcm_to_quaternion(dcm);

  const double radius = position.norm();
  if (radius == 0.0) {
    throw std::invalid_argument(zero_radius_msg);
  }
  const Vector3 acceleration = -mu * position / (radius * radius * radius);
  constexpr double step = 1.0;
  const Vector3 plus_position =
      position + step * velocity + 0.5 * step * step * acceleration;
  const Vector3 minus_position =
      position - step * velocity + 0.5 * step * step * acceleration;
  const Vector3 plus_velocity = velocity + step * acceleration;
  const Vector3 minus_velocity = velocity - step * acceleration;
  constexpr double day_step = step / kSecondsPerDay;

  const Eigen::Matrix3d dcm_plus =
      dcm_fn(plus_position, plus_velocity, epoch_jd + day_step);
  const Eigen::Matrix3d dcm_minus =
      dcm_fn(minus_position, minus_velocity, epoch_jd - day_step);
  const Eigen::Matrix3d dcm_dot = (dcm_plus - dcm_minus) / (2.0 * step);
  const Eigen::Matrix3d omega_skew_body = -dcm_dot * dcm.transpose();
  const Vector3 body_rate(
      0.5 * (omega_skew_body(2, 1) - omega_skew_body(1, 2)),
      0.5 * (omega_skew_body(0, 2) - omega_skew_body(2, 0)),
      0.5 * (omega_skew_body(1, 0) - omega_skew_body(0, 1)));
  return {quaternion, body_rate};
}

} // namespace detail

} // namespace omni::hardware::goals

#endif
