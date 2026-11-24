#include "zonal.hpp"
#include "../../environment/gravity/j_gravity_model.hpp"
namespace omni::dynamics
{
V zonal(const V &position, double mass, double epoch, double mu, double radius, const Eigen::VectorXd &coefficients)
{
    return mass * omni::environment::gravity_acceleration_inertial_m_s2(position, epoch, mu, radius, coefficients);
}
} // namespace omni::dynamics
