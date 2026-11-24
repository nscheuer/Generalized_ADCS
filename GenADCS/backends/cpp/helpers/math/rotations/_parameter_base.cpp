#include "_parameter_base.hpp"
#include "dcm/dcm.hpp"
#include "quaternion/algebra.hpp"
#include "quaternion/conversions.hpp"
#include <stdexcept>
#include <string>

namespace omni::rotations
{
namespace
{
void require_finite(const Vector3 &value, const char *description)
{
    if (!value.allFinite())
        throw std::invalid_argument(std::string(description) + " must be finite");
}
} // namespace
RotationParameter::RotationParameter(const Vector3 &values) : parameters_(values)
{
    require_finite(parameters_, "rotation parameters");
}
Vector3 RotationParameter::components() const
{
    return parameters_;
}
QuaternionVector RotationParameter::hamilton_components() const
{
    return to_hamilton();
}
double RotationParameter::norm() const
{
    return parameters_.norm();
}
Vector3 RotationParameter::as_vector() const
{
    return parameters_;
}
Quaternion RotationParameter::to_quaternion() const
{
    return Quaternion(to_hamilton());
}
DCM RotationParameter::to_dcm() const
{
    return DCM(to_quaternion().to_dcm());
}
Vector3 RotationParameter::rotate(const Vector3 &vector) const
{
    return to_quaternion().rotate(vector);
}
Vector3 RotationParameter::inertial_to_body(const Vector3 &vector) const
{
    return rotate(vector);
}
Vector3 RotationParameter::body_to_inertial(const Vector3 &vector) const
{
    return to_quaternion().body_to_inertial(vector);
}
Quaternion RotationParameter::conjugate() const
{
    return to_quaternion().conjugate();
}
Quaternion RotationParameter::inverse() const
{
    return to_quaternion().inverse();
}
Quaternion RotationParameter::multiply(const RotationParameter &other) const
{
    return to_quaternion().multiply(other.to_quaternion());
}
Quaternion RotationParameter::relative_to(const RotationParameter &reference) const
{
    return to_quaternion().relative_to(reference.to_quaternion());
}
} // namespace omni::rotations
