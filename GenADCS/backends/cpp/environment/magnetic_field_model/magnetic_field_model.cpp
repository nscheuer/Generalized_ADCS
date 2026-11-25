#include "magnetic_field_model.hpp"
#include "../../helpers/math/time/conversions.hpp"
#include "../native_helpers.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace omni::environment {
Eigen::Vector3d MagneticFieldModel::field(const Eigen::Vector3d &p,
                                          double jd) const {
  return field_inertial_tesla(p, jd);
}

} // namespace omni::environment
