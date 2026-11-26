#include "propagation/attitude_propagators/reference_attitude_propagator.hpp"
#include "helpers/math/rotations/rotation_types.hpp"

#include <cmath>
#include <stdexcept>

using omni::propagation::ReferenceAttitudeMode;
using omni::propagation::ReferenceAttitudePropagator;

void check(bool condition) {
  if (!condition) {
    throw std::runtime_error("reference attitude assertion failed");
  }
}

int main() {
  constexpr double radius = 7.0e6;
  constexpr double speed = 7500.0;
  const Eigen::Vector3d position(radius, 0.0, 0.0);
  const Eigen::Vector3d velocity(0.0, speed, 0.0);
  const double mu = speed * speed * radius;

  ReferenceAttitudePropagator nadir(0.5, ReferenceAttitudeMode::Nadir);
  const auto nadir_result = nadir.attitude(position, velocity, mu);
  const auto nadir_dcm = omni::rotations::quaternion_to_dcm(nadir_result.quaternion);
  check(nadir.dt() == 0.5);
  check((nadir_dcm.row(2).transpose() + Eigen::Vector3d::UnitX()).norm() < 1e-12);
  check(nadir_result.quaternion.allFinite());
  check(nadir_result.angular_velocity.allFinite());

  ReferenceAttitudePropagator along(1.0, ReferenceAttitudeMode::AlongTrack);
  const auto along_result = along.attitude(position, velocity, mu);
  const auto along_dcm = omni::rotations::quaternion_to_dcm(along_result.quaternion);
  check((along_dcm.row(2).transpose() - Eigen::Vector3d::UnitY()).norm() < 1e-12);

  ReferenceAttitudePropagator sun(1.0, ReferenceAttitudeMode::Sun);
  const Eigen::Vector3d sun_position(1.5e11, 0.0, 0.0);
  const auto sun_result = sun.attitude(position, velocity, mu, sun_position,
                                       Eigen::Vector3d::Zero());
  const auto sun_dcm = omni::rotations::quaternion_to_dcm(sun_result.quaternion);
  check(sun_dcm.allFinite());
  check(std::abs(sun_result.quaternion.norm() - 1.0) < 1e-12);

  bool bad_dt = false;
  try {
    ReferenceAttitudePropagator invalid(0.0, ReferenceAttitudeMode::Nadir);
  } catch (const std::invalid_argument &) {
    bad_dt = true;
  }
  check(bad_dt);
  bool bad_mode = false;
  try {
    (void)ReferenceAttitudePropagator::parse_mode("invalid");
  } catch (const std::invalid_argument &) {
    bad_mode = true;
  }
  check(bad_mode);
}
