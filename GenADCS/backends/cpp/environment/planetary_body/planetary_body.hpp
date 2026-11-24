#ifndef ADCS_OMNI_ENVIRONMENT_PLANETARY_BODY_HPP
#define ADCS_OMNI_ENVIRONMENT_PLANETARY_BODY_HPP
#include <string>
namespace omni::environment
{
struct PlanetaryBody
{
    PlanetaryBody(std::string body_name, double mu, double radius);
    std::string name;
    double gravitational_parameter;
    double reference_radius;
};
struct Earth final : PlanetaryBody
{
    Earth();
};
} // namespace omni::environment
#endif
