#include "dcm.hpp"
#include "../quaternion/conversions.hpp"
#include "kernels.hpp"

#include <cmath>
#include <stdexcept>

#include <Eigen/LU>

namespace omni::rotations
{
namespace
{
void require_proper_dcm(const Matrix3 &matrix)
{
    if (!matrix.allFinite())
        throw std::invalid_argument("DCM must be finite");
    const Matrix3 residual = matrix.transpose() * matrix - Matrix3::Identity();
    if ((residual.norm() > 1e-8) || (std::abs(matrix.determinant() - 1.0) > 1e-8))
    {
        throw std::invalid_argument("DCM must be a proper rotation matrix");
    }
}
} // namespace

DCM::DCM(const Matrix3 &matrix) : matrix_(matrix)
{
    require_proper_dcm(matrix_);
}
DCM DCM::identity()
{
    return DCM();
}
DCM DCM::from_axis_angle(const Vector3 &axis, double angle)
{
    return DCM(axis_angle_to_dcm(axis, angle));
}
DCM DCM::from_rotation_vector(const Vector3 &vector)
{
    return DCM(quaternion_to_dcm(rotation_vector_to_quaternion(vector)));
}
DCM DCM::from_quaternion(const Quaternion &quaternion)
{
    return DCM(quaternion.to_dcm());
}
DCM DCM::from_dcm(const Matrix3 &matrix)
{
    return DCM(matrix);
}
Matrix3 DCM::components() const
{
    return matrix_;
}
QuaternionVector DCM::hamilton_components() const
{
    return dcm_to_quaternion(matrix_);
}
double DCM::norm() const
{
    return dcm_orthogonality_error(matrix_);
}
double DCM::orthogonality_error() const
{
    return norm();
}
Matrix3 DCM::as_matrix() const
{
    return matrix_;
}
Matrix3 DCM::to_dcm() const
{
    return matrix_;
}
Quaternion DCM::to_quaternion() const
{
    return Quaternion(dcm_to_quaternion(matrix_));
}
DCM DCM::normalized() const
{
    return DCM(normalize_dcm(matrix_));
}
DCM DCM::inverse() const
{
    return DCM(inverse_dcm(matrix_));
}
DCM DCM::conjugate() const
{
    return inverse();
}
DCM DCM::multiply(const DCM &other) const
{
    return DCM(compose_dcm(matrix_, other.matrix_));
}
Vector3 DCM::rotate(const Vector3 &vector) const
{
    return rotate_vector(matrix_, vector);
}
Vector3 DCM::inertial_to_body(const Vector3 &vector) const
{
    return rotate(vector);
}
Vector3 DCM::body_to_inertial(const Vector3 &vector) const
{
    const Matrix3 inverse_matrix = matrix_.transpose();
    return rotate_vector(inverse_matrix, vector);
}
DCM DCM::relative_to(const DCM &reference) const
{
    return DCM(relative_dcm(matrix_, reference.matrix_));
}
} // namespace omni::rotations
