#include "check.hpp"
#include "dynamics/zonal_model/zonal.hpp"
using namespace omni::dynamics;
int main()
{
    Eigen::VectorXd c(5);
    c << 1.08262668e-3, -2.5324105e-6, -1.6198976e-6, -2.2729608e-7, 5.406812e-7;
    V p(7e6, 0, 0);
    auto lo = zonal(p, 10, 0, 3.986004418e14, 6378137, c);
    auto hi = zonal(V(8e6, 0, 0), 10, 0, 3.986004418e14, 6378137, c);
    CHECK(lo.allFinite() && lo.x() < 0 && lo.norm() > hi.norm());
    bool rejected = false;
    try
    {
        zonal(V::Zero(), 10, 0, 3.986e14, 6378137, c);
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    CHECK(rejected);
    CHECK((zonal(p, 20, 0, 3.986004418e14, 6378137, c) - 2 * lo).norm() < 1e-9);
}
