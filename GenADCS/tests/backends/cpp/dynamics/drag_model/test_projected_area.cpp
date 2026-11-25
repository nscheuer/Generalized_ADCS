#include "check.hpp"
#include "dynamics/drag_model/projected_area.hpp"
#include <cmath>
using namespace omni::dynamics;
int main() {
  Eigen::MatrixXd n(1, 3), arms(1, 3);
  n << 0, 0, 1;
  arms << 1, 0, 0;
  Eigen::VectorXd areas(1), cd(1);
  areas << 2;
  cd << 2.2;
  V v(0, 0, 7500);
  Q q(1, 0, 0, 0);
  auto f = drag(v, q, 1e-11, n, areas, cd, arms, false);
  CHECK(f.z() < 0 && f.allFinite());
  CHECK(drag(-v, q, 1e-11, n, areas, cd, arms, false).norm() == 0);
  CHECK((drag(V(0, 0, 100), q, 1, n, areas, cd, arms, true) -
         V(arms.row(0).transpose())
             .cross(drag(V(0, 0, 100), q, 1, n, areas, cd, arms, false)))
            .norm() < 1e-9);
  CHECK((drag(v, q, 2e-11, n, areas, cd, arms, false) - 2 * f).norm() < 1e-12);
  CHECK(drag(v, q, 0, n, areas, cd, arms, false).norm() == 0);
  auto t = drag(V(0, 0, 100), q, 1, n, areas, cd, arms, true);
  CHECK(t.allFinite() && t.norm() > 0);
}
