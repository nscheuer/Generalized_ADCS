#include "coupled_dop853_propagator.hpp"
#include "_dop853_coefficients.hpp"
#include <array>
#include <cmath>
namespace omni::propagation {
std::tuple<Vector, Vector, double>
dop853_attempt(const Vector &y, const Vector &f, double jd, double h,
               const Runtime &p, double rtol, const Vector &atol) {
  std::array<Vector, 13> k;
  k[0] = f;
  for (int stage = 1; stage < 12; ++stage) {
    Vector trial = y;
    for (int prev = 0; prev < stage; ++prev) {
      if (A[stage][prev] != 0) {
        trial += h * A[stage][prev] * k[prev];
      }
    }
    k[stage] = derivative(trial, jd + C[stage] * h / 86400., p);
  }
  Vector next = y;
  for (int i = 0; i < 12; ++i) {
    if (B[i] != 0) {
      next += h * B[i] * k[i];
    }
  }
  next.segment<4>(0).normalize();
  project(next, p);
  k[12] = derivative(next, jd + h / 86400., p);
  double n5 = 0, n3 = 0;
  for (Eigen::Index j = 0; j < y.size(); ++j) {
    double scale = atol[j] + rtol * std::max(std::abs(y[j]), std::abs(next[j])),
           e5 = 0, e3 = 0;
    for (int i = 0; i < 13; ++i) {
      e5 += E5[i] * k[i][j];
      e3 += E3[i] * k[i][j];
    }
    n5 += std::pow(e5 / scale, 2);
    n3 += std::pow(e3 / scale, 2);
  }
  double error =
      (n5 == 0 && n3 == 0)
          ? 0
          : std::abs(h) * n5 / std::sqrt((n5 + 0.01 * n3) * y.size());
  return {next, k[12], error};
}
} // namespace omni::propagation
