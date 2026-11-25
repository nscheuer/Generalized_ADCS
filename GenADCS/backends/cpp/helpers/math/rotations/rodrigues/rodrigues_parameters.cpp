#include "rodrigues_parameters.hpp"
#include "../_parameter_base.hpp"
#include "kernels.hpp"

namespace omni::rotations {
RodriguesParameters::RodriguesParameters(const Vector3 &values)
    : RotationParameter(values) {}
RodriguesParameters RodriguesParameters::identity() {
  return RodriguesParameters();
}
RodriguesParameters RodriguesParameters::from_quaternion(const Quaternion &q) {
  return from_quaternion_array(q.hamilton_components());
}
RodriguesParameters
RodriguesParameters::from_quaternion_array(const QuaternionVector &q) {
  return RodriguesParameters(quaternion_to_rodrigues_parameters(q));
}
RodriguesParameters RodriguesParameters::from_axis_angle(const Vector3 &axis,
                                                         double angle) {
  return from_quaternion(Quaternion::from_axis_angle(axis, angle));
}
RodriguesParameters
RodriguesParameters::from_rotation_vector(const Vector3 &vector) {
  return from_quaternion(Quaternion::from_rotation_vector(vector));
}
RodriguesParameters RodriguesParameters::from_dcm(const Matrix3 &matrix) {
  return from_quaternion(Quaternion::from_dcm(matrix));
}
RodriguesParameters RodriguesParameters::normalized() const {
  return RodriguesParameters(parameters_);
}
QuaternionVector RodriguesParameters::to_hamilton() const {
  return rodrigues_to_quaternion_parameters(parameters_);
}
} // namespace omni::rotations
