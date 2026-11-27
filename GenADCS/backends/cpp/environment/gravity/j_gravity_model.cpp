#include "j_gravity_model.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
namespace omni::environment {
namespace {
constexpr double kMu = 3.986004418e14;
constexpr double kRadius = 6378137.0;
constexpr std::array<double, 5> kEarthCoefficients{
    1.08262668e-3, -2.5324105e-6, -1.6198976e-6, -2.2729608e-7, 5.406812e-7};
} // namespace
Eigen::Vector3d gravity_acceleration_inertial_m_s2(
    const Eigen::Vector3d &position, double epoch_jd, double mu,
    double reference_radius, const Eigen::VectorXd &coefficients) {
  static_cast<void>(epoch_jd);
  const double x = position[0], y = position[1], z = position[2];
  const double r2 = position.squaredNorm();
  const double radius = std::sqrt(r2);
  if (radius == 0.0) {
    throw std::invalid_argument("Gravity is undefined at the body's centre");
  }
  const double inv_r = 1.0 / radius;
  const double s = z * inv_r;
  const double ratio = reference_radius * inv_r;
  const double ratio2 = ratio * ratio;
  const std::array<double, 5> p{
      0.5 * (3 * s * s - 1), 0.5 * (5 * s * s * s - 3 * s),
      (35 * std::pow(s, 4) - 30 * s * s + 3) / 8,
      (63 * std::pow(s, 5) - 70 * std::pow(s, 3) + 15 * s) / 8,
      (231 * std::pow(s, 6) - 315 * std::pow(s, 4) + 105 * s * s - 5) / 16};
  const std::array<double, 5> dp{
      3 * s, 0.5 * (15 * s * s - 3), (140 * std::pow(s, 3) - 60 * s) / 8,
      (315 * std::pow(s, 4) - 210 * s * s + 15) / 8,
      (1386 * std::pow(s, 5) - 1260 * std::pow(s, 3) + 210 * s) / 16};
  double radial = 1.0, z_gradient = 0.0, ratio_n = ratio2;
  const Eigen::Index count = std::min<Eigen::Index>(coefficients.size(), 5);
  for (Eigen::Index k = 0; k < count; ++k) {
    const double jn = coefficients[k], n = static_cast<double>(k + 2);
    radial -= (n + 1) * jn * ratio_n * p[static_cast<std::size_t>(k)];
    z_gradient += jn * ratio_n * dp[static_cast<std::size_t>(k)];
    ratio_n *= ratio;
  }
  const double mu_r3 = mu / (r2 * radius), base = -mu_r3 * radial;
  // The latitude term enters with a negative sign:
  // -mu/r^3 * sum(Jn (R/r)^n Pn'(s)) * (r z_hat - s r_vec).
  return Eigen::Vector3d(base * x - mu_r3 * z_gradient * (-s * x),
                         base * y - mu_r3 * z_gradient * (-s * y),
                         base * z - mu_r3 * z_gradient * (radius - s * z));
}
JGravityModel::JGravityModel(std::vector<int> degrees,
                             const Eigen::VectorXd &coefficients)
    : degrees_(std::move(degrees)), coefficients_(Eigen::VectorXd::Zero(5)) {
  std::sort(degrees_.begin(), degrees_.end());
  degrees_.erase(std::unique(degrees_.begin(), degrees_.end()), degrees_.end());
  for (int d : degrees_) {
    if (d < 2 || d > 6) {
      throw std::invalid_argument(
          "J-gravity degrees must be selected from 2 through 6");
    }
  }
  if (coefficients.size() != 0 && coefficients.size() < 5) {
    throw std::invalid_argument(
        "Provide five coefficients ordered as J2, J3, J4, J5, J6");
  }
  for (int d : degrees_) {
    int i = d - 2;
    coefficients_[i] = coefficients.size() == 0
                           ? kEarthCoefficients[static_cast<std::size_t>(i)]
                           : coefficients[i];
  }
}
JGravityModel::JGravityModel(int d) : JGravityModel(std::vector<int>{d}) {}
Eigen::Vector3d JGravityModel::evaluate(const Eigen::Vector3d &p,
                                        double epoch) const {
  return gravity_acceleration_inertial_m_s2(p, epoch, kMu, kRadius,
                                            coefficients_);
}
Eigen::Vector3d JGravityModel::acceleration(const Eigen::Vector3d &p,
                                            double epoch) const {
  return evaluate(p, epoch);
}
const std::vector<int> &JGravityModel::degrees() const noexcept {
  return degrees_;
}
const Eigen::VectorXd &JGravityModel::zonal_coefficients() const noexcept {
  return coefficients_;
}
int JGravityModel::degree() const noexcept { return 0; }

} // namespace omni::environment
