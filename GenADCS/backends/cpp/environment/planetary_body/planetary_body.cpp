#include "planetary_body.hpp"
#include <utility>
namespace omni::environment {
PlanetaryBody::PlanetaryBody(std::string body_name, double mu, double radius)
    : name(std::move(body_name)), gravitational_parameter(mu), reference_radius(radius) {}
}
