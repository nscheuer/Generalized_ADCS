#include "modified_rodrigues_parameters.hpp"
#include "../_parameter_base.hpp"
#include "kernels.hpp"

namespace omni::rotations {
ModifiedRodriguesParameters::ModifiedRodriguesParameters(const Vector3 &values)
    : RotationParameter(values) {
  if (parameters_.squaredNorm() > 1.0) {
    parameters_ = mrp_shadow_set(parameters_);
  }
}
ModifiedRodriguesParameters ModifiedRodriguesParameters::identity() {
  return ModifiedRodriguesParameters();
}
ModifiedRodriguesParameters
ModifiedRodriguesParameters::from_quaternion(const Quaternion &q) {
  return from_quaternion_array(q.hamilton_components());
}
ModifiedRodriguesParameters
ModifiedRodriguesParameters::from_quaternion_array(const QuaternionVector &q) {
  return ModifiedRodriguesParameters(quaternion_to_mrp_parameters(q));
}
ModifiedRodriguesParameters
ModifiedRodriguesParameters::from_axis_angle(const Vector3 &axis,
                                             double angle) {
  return from_quaternion(Quaternion::from_axis_angle(axis, angle));
}
ModifiedRodriguesParameters
ModifiedRodriguesParameters::from_rotation_vector(const Vector3 &vector) {
  return from_quaternion(Quaternion::from_rotation_vector(vector));
}
ModifiedRodriguesParameters
ModifiedRodriguesParameters::from_dcm(const Matrix3 &matrix) {
  return from_quaternion(Quaternion::from_dcm(matrix));
}
ModifiedRodriguesParameters
ModifiedRodriguesParameters::switch_shadow_set() const {
  return ModifiedRodriguesParameters(mrp_shadow_set(parameters_));
}
ModifiedRodriguesParameters ModifiedRodriguesParameters::normalized() const {
  return ModifiedRodriguesParameters(parameters_);
}
QuaternionVector ModifiedRodriguesParameters::to_hamilton() const {
  return mrp_to_quaternion_parameters(parameters_);
}
} // namespace omni::rotations
