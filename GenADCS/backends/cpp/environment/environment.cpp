#include "environment.hpp"

#include <utility>

namespace omni::environment {

Environment::Environment()
    : Environment(std::make_shared<Earth>(), nullptr, nullptr, nullptr, nullptr) {}

Environment::Environment(std::shared_ptr<PlanetaryBody> body,
                         std::shared_ptr<AtmosphericModel> atmosphere_model,
                         std::shared_ptr<MagneticFieldModel> magnetic_model,
                         std::shared_ptr<JGravityModel> gravity_model,
                         std::shared_ptr<DE421> ephemeris_model)
    : planetary_body(body ? std::move(body) : std::make_shared<Earth>()),
      atmosphere(std::move(atmosphere_model)),
      magnetic_field(std::move(magnetic_model)),
      j_gravity(std::move(gravity_model)),
      ephemeris(std::move(ephemeris_model)) {}

} // namespace omni::environment
