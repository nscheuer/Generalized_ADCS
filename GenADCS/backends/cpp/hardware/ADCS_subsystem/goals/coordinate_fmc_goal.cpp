#include "coordinate_fmc_goal.hpp"

#include "helpers/math/frames/conversions.hpp"

#include <Eigen/Geometry>
#include <cmath>
#include <stdexcept>

namespace omni::hardware::goals {
namespace {

constexpr double kEarthRotationRateRadPerS = 7.2921159e-5;

Eigen::Matrix3d coordinate_fmc_dcm(const Vector3 &position,
                                   const Vector3 &velocity, double epoch_jd,
                                   const Vector3 &target_ecef, int fmc_level) {
  const Vector3 target =
      omni::frames::ecef_to_gcrf_matrix(epoch_jd, epoch_jd) * target_ecef;
  const double radius = position.norm();
  if (radius == 0.0) {
    throw std::invalid_argument(
        "coordinate FMC goal is undefined at zero position");
  }
  const Vector3 radial = position / radius;
  const Vector3 nadir = -radial;

  const Vector3 target_direction = target - position;
  Vector3 azimuth = target_direction - target_direction.dot(radial) * radial;
  double azimuth_norm = azimuth.norm();
  const Vector3 earth_rate(0.0, 0.0, kEarthRotationRateRadPerS);
  const Vector3 ground_velocity = velocity - earth_rate.cross(position);
  if (azimuth_norm < 1e-12) {
    azimuth = ground_velocity - ground_velocity.dot(radial) * radial;
    azimuth_norm = azimuth.norm();
  }
  if (azimuth_norm < 1e-12) {
    throw std::invalid_argument("coordinate FMC tilt azimuth is degenerate");
  }
  azimuth /= azimuth_norm;

  const double tilt = std::acos(1.0 / static_cast<double>(fmc_level));
  const Vector3 z_axis = std::cos(tilt) * nadir + std::sin(tilt) * azimuth;

  Vector3 x_raw = ground_velocity - ground_velocity.dot(z_axis) * z_axis;
  double x_norm = x_raw.norm();
  if (x_norm < 1e-12) {
    x_raw = radial - radial.dot(z_axis) * z_axis;
    x_norm = x_raw.norm();
  }
  if (x_norm < 1e-12) {
    throw std::invalid_argument(
        "coordinate FMC roll reference is degenerate");
  }
  return detail::build_triad_dcm(z_axis, x_raw / x_norm);
}

} // namespace

CoordinateFMCGoal::CoordinateFMCGoal(double lat, double lon, double alt,
                                     int fmc_level)
    : CoordinateGoal(lat, lon, alt), fmc_level_(fmc_level) {
  if (fmc_level < 1 || fmc_level > 10) {
    throw std::invalid_argument(
        "fmc_level must be an integer from 1 through 10");
  }
}

GoalReference CoordinateFMCGoal::reference(
    const Eigen::Ref<const Eigen::VectorXd> &state, double epoch_jd,
    const GoalRuntime &runtime) const {
  validate_reference_state(state);
  if (!std::isfinite(epoch_jd)) {
    throw std::invalid_argument("epoch_jd must be finite");
  }
  return detail::central_difference_reference(
      state, epoch_jd, runtime.mu,
      [this](const Vector3 &pos, const Vector3 &vel, double jd) {
        return coordinate_fmc_dcm(pos, vel, jd, target_ecef_m(), fmc_level_);
      },
      "coordinate FMC goal is undefined at zero position");
}

} // namespace omni::hardware::goals
