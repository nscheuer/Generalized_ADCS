#include "_spherical_harmonics.hpp"
#include "../../helpers/math/frames/conversions.hpp"
#include "../../helpers/math/time/conversions.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
namespace omni::environment {
constexpr double kMagneticReferenceRadiusKm = 6371.2;
constexpr double kPi = 3.14159265358979323846;
Eigen::Vector3d eval_field(const Eigen::Vector3d &position, double jd,
                           const Eigen::MatrixXd &g, const Eigen::MatrixXd &h,
                           int nmax) {
  const double jdtt = jd >= 2441317.5 ? omni::time::utc_jd_to_tt_jd(jd) : jd;
  const Eigen::Matrix3d rot = omni::frames::gcrf_to_ecef_matrix(jd, jdtt);
  const Eigen::Vector3d fixed = rot * position;
  const double radius = fixed.norm();
  if (radius <= 0.0)
    throw std::invalid_argument("position must have positive radius");
  const double theta = std::acos(std::clamp(fixed.z() / radius, -1.0, 1.0));
  const double phi = std::atan2(fixed.y(), fixed.x());
  const double st = std::sin(theta), ct = std::cos(theta), sp = std::sin(phi),
               cp = std::cos(phi);
  const double safe_theta =
      (theta == 0.0) ? 1.0e-7 : ((theta == kPi) ? kPi - 1.0e-7 : theta);
  const double safe_sin = std::sin(safe_theta);
  std::vector<double> p1(static_cast<std::size_t>(nmax + 2), 0.0), p2(p1),
      pc(p1), d1(p1), d2(p1), dc(p1);
  std::vector<double> cm(static_cast<std::size_t>(nmax + 1), 0.0), sm(cm);
  cm[0] = 1.0;
  for (int m = 1; m <= nmax; ++m) {
    const double previous_cos = cm[m - 1], previous_sin = sm[m - 1];
    cm[m] = previous_cos * cp - previous_sin * sp;
    sm[m] = previous_sin * cp + previous_cos * sp;
  }
  p1[0] = 1.0;
  const double ar = kMagneticReferenceRadiusKm / (radius / 1000.0);
  double arn = ar * ar;
  double br = 0.0, bt = 0.0, bp = 0.0;
  for (int n = 1; n <= nmax; ++n) {
    arn *= ar;
    for (int m = 0; m <= n; ++m) {
      if (n == m) {
        pc[n] = safe_sin * p1[m - 1];
        dc[n] = safe_sin * d1[m - 1] + std::cos(safe_theta) * p1[n - 1];
      } else if (n == 1) {
        pc[m] = std::cos(safe_theta) * p1[m];
        dc[m] = std::cos(safe_theta) * d1[m] - safe_sin * p1[m];
      } else {
        const double q = ((n - 1.0) * (n - 1.0) - m * m) /
                         ((2.0 * n - 1.0) * (2.0 * n - 3.0));
        pc[m] = std::cos(safe_theta) * p1[m] - q * p2[m];
        dc[m] = std::cos(safe_theta) * d1[m] - safe_sin * p1[m] - q * d2[m];
      }
    }
    pc[n + 1] = 0.0;
    dc[n + 1] = 0.0;
    for (int m = 0; m <= n; ++m) {
      const double gn = g(n, m), hn = h(n, m), cc = gn * cm[m] + hn * sm[m];
      br += arn * (n + 1.0) * cc * pc[m];
      bt -= arn * cc * dc[m];
      bp -= arn * m * (hn * cm[m] - gn * sm[m]) * pc[m];
    }
    std::vector<double> old_p2 = std::move(p2);
    p2 = std::move(p1);
    p1 = std::move(pc);
    pc = std::move(old_p2);
    std::vector<double> old_d2 = std::move(d2);
    d2 = std::move(d1);
    d1 = std::move(dc);
    dc = std::move(old_d2);
  }
  bp /= safe_sin;
  const Eigen::Vector3d bfixed((br * st * cp + bt * ct * cp - bp * sp) * 1e-9,
                               (br * st * sp + bt * ct * sp + bp * cp) * 1e-9,
                               (br * ct - bt * st) * 1e-9);
  return rot.transpose() * bfixed;
}
} // namespace omni::environment
