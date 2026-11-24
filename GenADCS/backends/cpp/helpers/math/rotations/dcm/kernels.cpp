#include "kernels.hpp"
#include "../quaternion/algebra.hpp"
#include "../quaternion/conversions.hpp"
#include <Eigen/Geometry>
#include <stdexcept>

namespace omni::rotations
{
Matrix3 axis_angle_to_dcm(const Vector3 &axis, double angle)
{
    return quaternion_to_dcm(axis_angle_to_quaternion(axis, angle));
}
Vector3 rotate_vector(const Matrix3 &dcm, const Vector3 &vector)
{
    return dcm * vector;
}
Matrix3 inverse_dcm(const Matrix3 &dcm)
{
    return dcm.transpose();
}
Matrix3 compose_dcm(const Matrix3 &left, const Matrix3 &right)
{
    return left * right;
}
Matrix3 relative_dcm(const Matrix3 &target, const Matrix3 &reference)
{
    return target * reference.transpose();
}
double dcm_orthogonality_error(const Matrix3 &matrix)
{
    return (matrix.transpose() * matrix - Matrix3::Identity()).norm();
}

Matrix3 normalize_dcm(const Matrix3 &matrix)
{
    Vector3 first = matrix.col(0);
    const double first_norm = first.norm();
    if (first_norm == 0.0)
        throw std::invalid_argument("DCM has a degenerate first column");
    first /= first_norm;
    Vector3 second = matrix.col(1) - first.dot(matrix.col(1)) * first;
    const double second_norm = second.norm();
    if (second_norm == 0.0)
        throw std::invalid_argument("DCM has degenerate first and second columns");
    second /= second_norm;
    const Vector3 third = first.cross(second);
    Matrix3 result;
    result.col(0) = first;
    result.col(1) = second;
    result.col(2) = third;
    return result;
}
} // namespace omni::rotations
