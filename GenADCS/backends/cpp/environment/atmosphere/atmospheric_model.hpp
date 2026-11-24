#ifndef ADCS_OMNI_ENVIRONMENT_ATMOSPHERIC_MODEL_HPP
#define ADCS_OMNI_ENVIRONMENT_ATMOSPHERIC_MODEL_HPP
#include <Eigen/Core>
namespace omni::environment
{
struct AtmosphericModel
{
    virtual ~AtmosphericModel() = default;
    virtual double density(const Eigen::Vector3d &position, double epoch_jd) const;
};
Eigen::Vector3d co_rotating_atmosphere_velocity(
    const Eigen::Vector3d &position, double epoch_jd,
    const Eigen::Vector3d &angular_velocity = Eigen::Vector3d(0.0, 0.0, 7.2921150e-5));
double atmospheric_density(const Eigen::Vector3d &position, double epoch_jd);
} // namespace omni::environment
#endif
