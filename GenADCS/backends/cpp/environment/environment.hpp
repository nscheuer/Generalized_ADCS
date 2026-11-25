#ifndef ADCS_OMNI_ENVIRONMENT_ENVIRONMENT_HPP
#define ADCS_OMNI_ENVIRONMENT_ENVIRONMENT_HPP

#include "atmosphere/atmospheric_model.hpp"
#include "gravity/j_gravity_model.hpp"
#include "native_models.hpp"
#include "planetary_body/planetary_body.hpp"

#include <memory>

namespace omni::environment {

// Holds the native environmental models used by a simulation. Python objects
// are converted to these native model types in environment_py.cpp.
struct Environment {
  Environment();
  Environment(std::shared_ptr<PlanetaryBody> planetary_body,
              std::shared_ptr<AtmosphericModel> atmosphere,
              std::shared_ptr<MagneticFieldModel> magnetic_field,
              std::shared_ptr<JGravityModel> j_gravity,
              std::shared_ptr<DE421> ephemeris);

  std::shared_ptr<PlanetaryBody> planetary_body;
  std::shared_ptr<AtmosphericModel> atmosphere;
  std::shared_ptr<MagneticFieldModel> magnetic_field;
  std::shared_ptr<JGravityModel> j_gravity;
  std::shared_ptr<DE421> ephemeris;
};

} // namespace omni::environment
#endif
