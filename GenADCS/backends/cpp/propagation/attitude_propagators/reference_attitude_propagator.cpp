#include "reference_attitude_propagator.hpp"

#include "../../helpers/math/rotations/rotation_types.hpp"
#include "../../environment/native_models.hpp"
#include "../../helpers/math/time/conversions.hpp"

#include <Eigen/Geometry>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace omni::propagation {
namespace {
using Vec3 = Eigen::Vector3d;

std::pair<Vec3, Vec3> unit_and_rate(const Vec3 &value, const Vec3 &derivative) {
  const double length = value.norm();
  if (!(length > 0.0) || !std::isfinite(length)) {
    throw std::invalid_argument("reference direction is undefined");
  }
  const Vec3 unit = value / length;
  return {unit, (derivative - unit.dot(derivative) * unit) / length};
}
} // namespace

ReferenceAttitudePropagator::ReferenceAttitudePropagator(
    double dt, ReferenceAttitudeMode mode)
    : dt_(dt), mode_(mode) {
  if (!std::isfinite(dt_) || dt_ <= 0.0) {
    throw std::invalid_argument("dt must be finite and positive seconds");
  }
}

double ReferenceAttitudePropagator::dt() const noexcept { return dt_; }
ReferenceAttitudeMode ReferenceAttitudePropagator::mode() const noexcept {
  return mode_;
}

void ReferenceAttitudePropagator::apply(
    Eigen::Ref<Eigen::MatrixXd> states, const std::vector<double> &times,
    double initial_julian_date, double mu) const {
  if (states.rows() != static_cast<Eigen::Index>(times.size()) ||
      states.cols() < 13) {
    throw std::invalid_argument("states and times have incompatible shapes");
  }
  const std::unique_ptr<omni::environment::DE421> ephemeris =
      mode_ == ReferenceAttitudeMode::Sun
          ? std::make_unique<omni::environment::DE421>()
          : nullptr;
  for (Eigen::Index row = 0; row < states.rows(); ++row) {
    Eigen::Vector3d sun_position = Eigen::Vector3d::Zero();
    Eigen::Vector3d sun_velocity = Eigen::Vector3d::Zero();
    if (ephemeris) {
      const double tdb = omni::time::utc_jd_to_tdb_jd(
          initial_julian_date + times[static_cast<std::size_t>(row)] / 86400.0);
      sun_position = 1000.0 * ephemeris->position(0, tdb);
      sun_velocity = 1000.0 * ephemeris->velocity(0, tdb);
    }
    const ReferenceAttitude result = attitude(
        states.row(row).segment<3>(7).transpose(),
        states.row(row).segment<3>(10).transpose(), mu, sun_position,
        sun_velocity);
    states.row(row).segment<4>(0) = result.quaternion.transpose();
    states.row(row).segment<3>(4) = result.angular_velocity.transpose();
  }
}

ReferenceAttitudeMode
ReferenceAttitudePropagator::parse_mode(const std::string &mode) {
  if (mode == "nadir" || mode == "nadir-pointing") {
    return ReferenceAttitudeMode::Nadir;
  }
  if (mode == "along-track") {
    return ReferenceAttitudeMode::AlongTrack;
  }
  if (mode == "sun" || mode == "sun-pointing") {
    return ReferenceAttitudeMode::Sun;
  }
  throw std::invalid_argument("unknown reference attitude mode: " + mode);
}

ReferenceAttitude ReferenceAttitudePropagator::attitude(
    const Vec3 &position, const Vec3 &velocity, double mu,
    const Vec3 &sun_position, const Vec3 &sun_velocity) const {
  if (!position.allFinite() || !velocity.allFinite() || !std::isfinite(mu) ||
      mu <= 0.0) {
    throw std::invalid_argument("orbital state and mu must be finite and valid");
  }
  const double radius = position.norm();
  if (!(radius > 0.0)) {
    throw std::invalid_argument("orbital radius must be positive");
  }
  const Vec3 acceleration = -mu * position / (radius * radius * radius);
  const auto radial_pair = unit_and_rate(position, velocity);
  const Vec3 radial = radial_pair.first, radial_dot = radial_pair.second;
  const Vec3 tangent_raw = velocity - velocity.dot(radial) * radial;
  const Vec3 tangent_dot_raw =
      acceleration - (acceleration.dot(radial) + velocity.dot(radial_dot)) * radial -
      velocity.dot(radial) * radial_dot;
  const auto tangent_pair = unit_and_rate(tangent_raw, tangent_dot_raw);
  const Vec3 tangent = tangent_pair.first, tangent_dot = tangent_pair.second;

  Vec3 z, z_dot, x_raw, x_dot_raw;
  if (mode_ == ReferenceAttitudeMode::Nadir) {
    z = -radial;
    z_dot = -radial_dot;
    x_raw = tangent;
    x_dot_raw = tangent_dot;
  } else if (mode_ == ReferenceAttitudeMode::AlongTrack) {
    z = tangent;
    z_dot = tangent_dot;
    x_raw = radial;
    x_dot_raw = radial_dot;
  } else {
    if (!sun_position.allFinite() || !sun_velocity.allFinite()) {
      throw std::invalid_argument("Sun state must be finite");
    }
    const auto z_pair = unit_and_rate(sun_position - position,
                                      sun_velocity - velocity);
    z = z_pair.first;
    z_dot = z_pair.second;
    Vec3 reference = tangent, reference_dot = tangent_dot;
    if ((reference - reference.dot(z) * z).norm() < 1e-10) {
      reference = radial;
      reference_dot = radial_dot;
    }
    const double scalar = reference.dot(z);
    const double scalar_dot = reference_dot.dot(z) + reference.dot(z_dot);
    x_raw = reference - scalar * z;
    x_dot_raw = reference_dot - scalar_dot * z - scalar * z_dot;
  }

  const auto x_pair = unit_and_rate(x_raw, x_dot_raw);
  const Vec3 x = x_pair.first, x_dot = x_pair.second;
  const Vec3 y_raw = z.cross(x);
  const Vec3 y_dot_raw = z_dot.cross(x) + z.cross(x_dot);
  const auto y_pair = unit_and_rate(y_raw, y_dot_raw);
  const Vec3 body_y = y_pair.first, body_y_dot = y_pair.second;
  const Vec3 body_x = body_y.cross(z);
  const Vec3 body_x_dot = body_y_dot.cross(z) + body_y.cross(z_dot);

  Eigen::Matrix3d dcm;
  dcm.row(0) = body_x.transpose();
  dcm.row(1) = body_y.transpose();
  dcm.row(2) = z.transpose();
  ReferenceAttitude result;
  result.quaternion = omni::rotations::dcm_to_quaternion(dcm);
  result.angular_velocity << -body_y_dot.dot(z), -z_dot.dot(body_x),
      -body_x_dot.dot(body_y);
  return result;
}

} // namespace omni::propagation
