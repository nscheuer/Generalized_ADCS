#include "static.hpp"
#include <cmath>
#include <stdexcept>
namespace omni::dynamics
{
V gravity_gradient(const V &p, const Q &q, const M &inertia, double mu)
{
    const double radius = p.norm();
    if (radius == 0)
        throw std::invalid_argument("gravity gradient is undefined at the body's centre");
    V radial = rotation(q) * (p / radius);
    return (3 * mu / std::pow(radius, 3)) * V(radial.cross(inertia * radial));
}
} // namespace omni::dynamics
