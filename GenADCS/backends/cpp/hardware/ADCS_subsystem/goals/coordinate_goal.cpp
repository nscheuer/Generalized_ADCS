#include "coordinate_goal.hpp"

#include "helpers/math/frames/conversions.hpp"

#include <cmath>
#include <stdexcept>

namespace omni::hardware::goals {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDegreesToRadians = kPi / 180.0;

Eigen::Matrix3d coordinate_dcm(const Vector3 &position, const Vector3 &velocity,
                               double epoch_jd, const Vector3 &target_ecef) {
  const Vector3 target =
      omni::frames::ecef_to_gcrf_matrix(epoch_jd, epoch_jd) * target_ecef;
  const Vector3 relative = target - position;
  const double distance = relative.norm();
  if (distance == 0.0) {
    throw std::invalid_argument(
        "spacecraft position coincides with coordinate target");
  }
  const Vector3 z_axis = relative / distance;

  Vector3 x_raw = velocity - velocity.dot(z_axis) * z_axis;
  double x_norm = x_raw.norm();
  if (x_norm < 1e-12) {
    const double position_norm = position.norm();
    if (position_norm == 0.0) {
      throw std::invalid_argument("coordinate goal roll reference is degenerate");
    }
    const Vector3 radial = position / position_norm;
    x_raw = radial - radial.dot(z_axis) * z_axis;
    x_norm = x_raw.norm();
  }
  if (x_norm < 1e-12) {
    throw std::invalid_argument("coordinate goal roll reference is degenerate");
  }
  return detail::build_triad_dcm(z_axis, x_raw / x_norm);
}

} // namespace

Vector3 CoordinateGoal::geodetic_to_ecef(double lat_deg, double lon_deg,
                                         double alt_km) {
  const double lat = lat_deg * kDegreesToRadians;
  const double lon = lon_deg * kDegreesToRadians;
  constexpr double semi_major = 6378137.0;
  constexpr double flattening = 1.0 / 298.257223563;
  constexpr double eccentricity_sq = flattening * (2.0 - flattening);
  const double altitude_m = alt_km * 1000.0;
  const double sin_lat = std::sin(lat);
  const double prime_vertical =
      semi_major / std::sqrt(1.0 - eccentricity_sq * sin_lat * sin_lat);
  return Vector3(
      (prime_vertical + altitude_m) * std::cos(lat) * std::cos(lon),
      (prime_vertical + altitude_m) * std::cos(lat) * std::sin(lon),
      (prime_vertical * (1.0 - eccentricity_sq) + altitude_m) * sin_lat);
}

CoordinateGoal::CoordinateGoal(double lat, double lon, double alt)
    : lat_deg_(lat), lon_deg_(lon), alt_km_(alt) {
  if (!std::isfinite(lat) || !std::isfinite(lon) || !std::isfinite(alt)) {
    throw std::invalid_argument(
        "latitude, longitude, and altitude must be finite");
  }
  if (lat < -90.0 || lat > 90.0) {
    throw std::invalid_argument("latitude must be between -90 and 90 degrees");
  }
  if (lon < -180.0 || lon > 180.0) {
    throw std::invalid_argument(
        "longitude must be between -180 and 180 degrees");
  }
  if (alt < -6378.137) {
    throw std::invalid_argument(
        "altitude is below the WGS84 ellipsoid center");
  }
  target_ecef_m_ = geodetic_to_ecef(lat, lon, alt);
}

GoalReference CoordinateGoal::reference(
    const Eigen::Ref<const Eigen::VectorXd> &state, double epoch_jd,
    const GoalRuntime &runtime) const {
  validate_reference_state(state);
  if (!std::isfinite(epoch_jd)) {
    throw std::invalid_argument("epoch_jd must be finite");
  }
  return detail::central_difference_reference(
      state, epoch_jd, runtime.mu,
      [this](const Vector3 &pos, const Vector3 &vel, double jd) {
        return coordinate_dcm(pos, vel, jd, target_ecef_m_);
      },
      "coordinate goal is undefined at zero position");
}

} // namespace omni::hardware::goals
