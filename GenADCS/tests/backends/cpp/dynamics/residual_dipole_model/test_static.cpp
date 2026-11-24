#include "check.hpp"
#include "dynamics/residual_dipole_model/static.hpp"
using namespace omni::dynamics;
int main()
{
    V m(1, 0, 0), b(0, 0, 2e-5);
    Q q(1, 0, 0, 0);
    auto t = residual_dipole(m, b, q);
    CHECK((t - m.cross(b)).norm() < 1e-15);
}
