#include "_orbit_goal.hpp"

#include "environment/ephemeris/de421/de421.hpp"
#include "helpers/math/rotations/rotation_types.hpp"
#include "helpers/math/time/conversions.hpp"

#include <Eigen/Geometry>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace omni::hardware::goals {
namespace {

std::pair<Vector3, Vector3> unit_rate(const Vector3 &vector,
                                      const Vector3 &derivative) {
  const double length = vector.norm();
  if (length <= 0.0) {
    throw std::invalid_argument("reference direction is undefined");
  }
  const Vector3 unit = vector / length;
  return {unit, (derivative - unit.dot(derivative) * unit) / length};
}

const omni::environment::DE421 &default_de421() {
  static const omni::environment::DE421 instance;
  return instance;
}

} // namespace

GoalReference OrbitGoal::reference(const Eigen::Ref<const Eigen::VectorXd> &state,
                                   double epoch_jd,
                                   const GoalRuntime &runtime) const {
  if (!std::isfinite(epoch_jd)) {
    throw std::invalid_argument("epoch_jd must be finite");
  }
  validate_reference_state(state);

  const Vector3 position = state.segment<3>(7);
  const Vector3 velocity = state.segment<3>(10);
  const double radius = position.norm();
  if (radius <= 0.0) {
    throw std::invalid_argument("orbital frame is undefined at zero position");
  }
  const Vector3 acceleration =
      -runtime.mu * position / (radius * radius * radius);

  const auto [radial, radial_dot] = unit_rate(position, velocity);

  const Vector3 tangent_raw = velocity - velocity.dot(radial) * radial;
  const Vector3 tangent_dot_raw =
      acceleration -
      (acceleration.dot(radial) + velocity.dot(radial_dot)) * radial -
      velocity.dot(radial) * radial_dot;
  const auto [tangent, tangent_dot] = unit_rate(tangent_raw, tangent_dot_raw);

  const Vector3 orbit_normal_raw = position.cross(velocity);
  const Vector3 orbit_normal_dot_raw = position.cross(acceleration);
  const auto [orbit_normal, orbit_normal_dot] =
      unit_rate(orbit_normal_raw, orbit_normal_dot_raw);

  Vector3 z = Vector3::Zero();
  Vector3 z_dot = Vector3::Zero();
  Vector3 x_raw = Vector3::Zero();
  Vector3 x_dot_raw = Vector3::Zero();

  switch (mode_) {
  case OrbitGoalMode::Nadir:
    z = -radial;
    z_dot = -radial_dot;
    x_raw = tangent;
    x_dot_raw = tangent_dot;
    break;
  case OrbitGoalMode::Zenith:
    z = radial;
    z_dot = radial_dot;
    x_raw = tangent;
    x_dot_raw = tangent_dot;
    break;
  case OrbitGoalMode::AlongTrack:
    z = tangent;
    z_dot = tangent_dot;
    x_raw = radial;
    x_dot_raw = radial_dot;
    break;
  case OrbitGoalMode::CrossTrack:
    z = orbit_normal;
    z_dot = orbit_normal_dot;
    x_raw = radial;
    x_dot_raw = radial_dot;
    break;
  case OrbitGoalMode::Velocity: {
    const auto [vel_unit, vel_dot] = unit_rate(velocity, acceleration);
    z = vel_unit;
    z_dot = vel_dot;
    x_raw = radial;
    x_dot_raw = radial_dot;
    break;
  }
  case OrbitGoalMode::Sun: {
    const double tdb_jd = omni::time::utc_jd_to_tdb_jd(epoch_jd);
    const omni::environment::DE421 &ephem =
        runtime.ephemeris != nullptr ? *runtime.ephemeris : default_de421();
    const Vector3 sun = ephem.position(0, tdb_jd) * 1000.0;
    const Vector3 sun_velocity = ephem.velocity(0, tdb_jd) * 1000.0;
    const auto [sun_unit, sun_dot] =
        unit_rate(sun - position, sun_velocity - velocity);
    z = sun_unit;
    z_dot = sun_dot;
    x_raw = tangent;
    x_dot_raw = tangent_dot;
    const Vector3 projected = x_raw - x_raw.dot(z) * z;
    if (projected.norm() < 1e-10) {
      x_raw = radial;
      x_dot_raw = radial_dot;
    }
    break;
  }
  }

  const double scalar = x_raw.dot(z);
  const double scalar_dot = x_dot_raw.dot(z) + x_raw.dot(z_dot);
  const Vector3 x_projected = x_raw - scalar * z;
  const Vector3 x_projected_dot = x_dot_raw - scalar_dot * z - scalar * z_dot;
  const auto [x, x_dot] = unit_rate(x_projected, x_projected_dot);

  const Vector3 y_raw = z.cross(x);
  const Vector3 y_dot_raw = z_dot.cross(x) + z.cross(x_dot);
  const auto [body_y, body_y_dot] = unit_rate(y_raw, y_dot_raw);

  const Vector3 body_x = body_y.cross(z);
  const Vector3 body_x_dot = body_y_dot.cross(z) + body_y.cross(z_dot);

  Eigen::Matrix3d dcm;
  dcm.row(0) = body_x;
  dcm.row(1) = body_y;
  dcm.row(2) = z;
  const Quaternion quaternion = omni::rotations::dcm_to_quaternion(dcm);

  Vector3 rate;
  rate[0] = -body_y_dot.dot(z);
  rate[1] = -z_dot.dot(body_x);
  rate[2] = -body_x_dot.dot(body_y);

  return {quaternion, rate};
}

} // namespace omni::hardware::goals
