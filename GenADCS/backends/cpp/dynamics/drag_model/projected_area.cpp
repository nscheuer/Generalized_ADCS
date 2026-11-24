#include "projected_area.hpp"
namespace omni::dynamics
{
V drag(const V &velocity, const Q &q, double density, const Eigen::MatrixXd &normals, const Eigen::VectorXd &areas,
       const Eigen::VectorXd &cd, const Eigen::MatrixXd &arms, bool torque)
{
    V v = rotation(q) * velocity;
    const double speed = v.norm();
    V out = V::Zero();
    if (speed > 0 && density > 0)
        for (int i = 0; i < areas.size(); ++i)
        {
            const double incidence = normals.row(i).dot(v) / speed;
            if (incidence > 0)
            {
                V force = -0.5 * density * speed * areas[i] * cd[i] * incidence * v;
                out += torque ? V(V(arms.row(i).transpose()).cross(force)) : force;
            }
        }
    return torque ? out : V(rotation(q).transpose() * out);
}
} // namespace omni::dynamics
