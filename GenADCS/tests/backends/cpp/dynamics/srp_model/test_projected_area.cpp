#include "check.hpp"
#include "dynamics/srp_model/projected_area.hpp"
using namespace omni::dynamics;
int main() {
  Eigen::MatrixXd n(1, 3), opt(1, 3), arms(1, 3);
  n << 1, 0, 0;
  opt << 0, 0, 1;
  arms << 0, 1, 0;
  Eigen::VectorXd a(1);
  a << 2;
  Q q(1, 0, 0, 0);
  V p = V::Zero(), sun(149597870700., 0, 0);
  auto f = srp(p, sun, q, n, a, opt, arms, 1, false);
  CHECK(f.allFinite() && f.norm() > 0);
  CHECK((srp(p, sun, q, n, a, opt, arms, .5, false) - .5 * f).norm() < 1e-12);
  CHECK(srp(p, sun, q, n, a, opt, arms, 0, false).norm() == 0);
  CHECK(srp(p, sun, q, n, a, opt, arms, 1, true).norm() > 0);
  bool rejected = false;
  try {
    srp(p, p, q, n, a, opt, arms, 1, false);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  CHECK(rejected);
  CHECK((srp(p, sun, q, n, a, opt, arms, 1, true) -
         V(arms.row(0).transpose()).cross(f))
            .norm() < 1e-12);
}
