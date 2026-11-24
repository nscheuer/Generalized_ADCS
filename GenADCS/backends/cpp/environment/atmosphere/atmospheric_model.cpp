#include "atmospheric_model.hpp"
#include <Eigen/Geometry>
#include <stdexcept>
namespace omni::environment
{
Eigen::Vector3d co_rotating_atmosphere_velocity(const Eigen::Vector3d &position, double epoch_jd,
                                                const Eigen::Vector3d &angular_velocity)
{
    static_cast<void>(epoch_jd);
    return angular_velocity.cross(position);
}
double atmospheric_density(const Eigen::Vector3d &position, double epoch_jd)
{
    static_cast<void>(position);
    static_cast<void>(epoch_jd);
    throw std::logic_error("AtmosphericModel density is not implemented; select NRLMSISE00");
}
} // namespace omni::environment
