#include "frames.hpp"

#include <stdexcept>

namespace omni::frames
{
namespace
{
Matrix3 quaternion_to_matrix(const Quaternion &quaternion)
{
    if (!quaternion.allFinite())
    {
        throw std::invalid_argument("quaternion must be finite");
    }
    const double w = quaternion[0];
    const double x = quaternion[1];
    const double y = quaternion[2];
    const double z = quaternion[3];
    Matrix3 result;
    result << (w * w) + (x * x) - (y * y) - (z * z), 2.0 * ((x * y) - (w * z)), 2.0 * ((x * z) + (w * y)),
        2.0 * ((x * y) + (w * z)), (w * w) - (x * x) + (y * y) - (z * z), 2.0 * ((y * z) - (w * x)),
        2.0 * ((x * z) - (w * y)), 2.0 * ((y * z) + (w * x)), (w * w) - (x * x) - (y * y) + (z * z);
    return result;
}
} // namespace

Matrix3 inertial_to_body_matrix(const Quaternion &quaternion_body_from_inertial)
{
    return quaternion_to_matrix(quaternion_body_from_inertial);
}

Matrix3 body_to_inertial_matrix(const Quaternion &quaternion_body_from_inertial)
{
    return quaternion_to_matrix(quaternion_body_from_inertial).transpose();
}

Vector3 inertial_to_body(const Vector3 &vector_inertial, const Quaternion &quaternion_body_from_inertial)
{
    return inertial_to_body_matrix(quaternion_body_from_inertial) * vector_inertial;
}

Vector3 body_to_inertial(const Vector3 &vector_body, const Quaternion &quaternion_body_from_inertial)
{
    return body_to_inertial_matrix(quaternion_body_from_inertial) * vector_body;
}

Vector3 transform_vector(const Vector3 &vector, const Matrix3 &source_to_target)
{
    return source_to_target * vector;
}

Matrix3 transform_covariance(const Matrix3 &covariance_source, const Matrix3 &source_to_target)
{
    return source_to_target * covariance_source * source_to_target.transpose();
}
} // namespace omni::frames
