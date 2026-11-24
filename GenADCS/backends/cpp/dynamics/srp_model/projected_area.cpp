#include "projected_area.hpp"
#include <cmath>
#include <stdexcept>
namespace omni::dynamics
{
V srp(const V &position, const V &sun, const Q &q, const Eigen::MatrixXd &normals, const Eigen::VectorXd &areas,
      const Eigen::MatrixXd &optical, const Eigen::MatrixXd &arms, double illumination, bool torque)
{
    V delta = sun - position;
    const double distance = delta.norm();
    if (distance == 0)
        throw std::invalid_argument("spacecraft and Sun positions coincide");
    V direction = rotation(q) * (delta / distance);
    const double pressure = (1361.0 / 299792458.0) * std::pow(149597870700.0 / distance, 2) * illumination;
    V out = V::Zero();
    for (int i = 0; i < areas.size(); ++i)
    {
        V normal = normals.row(i);
        const double cosine = normal.dot(direction);
        if (cosine > 0)
        {
            const double spec = optical(i, 0), diff = optical(i, 1), abs = optical(i, 2);
            V force = -pressure * areas[i] * cosine *
                      ((abs + diff) * direction + (2 * spec * cosine + 2 * diff / 3) * normal);
            out += torque ? V(V(arms.row(i).transpose()).cross(force)) : force;
        }
    }
    return torque ? out : V(rotation(q).transpose() * out);
}
} // namespace omni::dynamics
