#include "coupled_rk4_propagator.hpp"
namespace omni::propagation
{
Vector rk4_step(const Vector &y, double jd, double dt, const Runtime &p)
{
    constexpr double day = 86400.;
    Vector k1 = derivative(y, jd, p), k2 = derivative(y + dt * 0.5 * k1, jd + dt * 0.5 / day, p),
           k3 = derivative(y + dt * 0.5 * k2, jd + dt * 0.5 / day, p), k4 = derivative(y + dt * k3, jd + dt / day, p);
    Vector out = y + (dt / 6.) * (k1 + 2 * k2 + 2 * k3 + k4);
    out.segment<4>(0).normalize();
    project(out, p);
    return out;
}
} // namespace omni::propagation
