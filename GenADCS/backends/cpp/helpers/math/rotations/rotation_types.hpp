#ifndef ADCS_OMNI_HELPERS_ROTATIONS_HPP
#define ADCS_OMNI_HELPERS_ROTATIONS_HPP

#include <string>

#include <Eigen/Core>

namespace omni::rotations {
using Vector3 = Eigen::Vector3d;
using QuaternionVector = Eigen::Vector4d;
using Matrix3 = Eigen::Matrix3d;

QuaternionVector normalize_quaternion(const QuaternionVector& value);
double quaternion_norm(const QuaternionVector& value);
QuaternionVector jpl_to_hamilton(const QuaternionVector& value);
QuaternionVector hamilton_to_jpl(const QuaternionVector& value);
QuaternionVector conjugate(const QuaternionVector& value);
QuaternionVector inverse(const QuaternionVector& value);
QuaternionVector multiply(const QuaternionVector& left, const QuaternionVector& right);
QuaternionVector relative_quaternion(const QuaternionVector& reference, const QuaternionVector& target);
Vector3 rotate_vector(const QuaternionVector& quaternion, const Vector3& vector);
Matrix3 quaternion_to_dcm(const QuaternionVector& quaternion);
QuaternionVector dcm_to_quaternion(const Matrix3& matrix);
QuaternionVector axis_angle_to_quaternion(const Vector3& axis, double angle);
QuaternionVector rotation_vector_to_quaternion(const Vector3& rotation_vector);
Vector3 quaternion_to_rotation_vector(const QuaternionVector& quaternion);
QuaternionVector slerp(const QuaternionVector& start, const QuaternionVector& finish, double fraction);

class Quaternion {
public:
    explicit Quaternion(const QuaternionVector& values = QuaternionVector(1.0, 0.0, 0.0, 0.0),
                        const std::string& convention = "Hamilton");
    static Quaternion identity();
    static Quaternion from_axis_angle(const Vector3& axis, double angle);
    static Quaternion from_rotation_vector(const Vector3& rotation_vector);
    static Quaternion from_dcm(const Matrix3& matrix);
    QuaternionVector components() const;
    QuaternionVector hamilton_components() const;
    std::string convention() const;
    double norm() const;
    QuaternionVector as_convention(const std::string& convention = "Hamilton") const;
    Quaternion to_convention(const std::string& convention = "Hamilton") const;
    Quaternion normalized() const;
    Quaternion conjugate() const;
    Quaternion inverse() const;
    Quaternion multiply(const Quaternion& other) const;
    Vector3 rotate(const Vector3& vector) const;
    Vector3 inertial_to_body(const Vector3& vector) const;
    Vector3 body_to_inertial(const Vector3& vector) const;
    Matrix3 to_dcm() const;
    Quaternion relative_to(const Quaternion& reference) const;
private:
    QuaternionVector hamilton_;
    std::string convention_;
};

Matrix3 axis_angle_to_dcm(const Vector3& axis, double angle);
Vector3 rotate_vector(const Matrix3& dcm, const Vector3& vector);
Matrix3 inverse_dcm(const Matrix3& dcm);
Matrix3 compose_dcm(const Matrix3& left, const Matrix3& right);
Matrix3 relative_dcm(const Matrix3& target, const Matrix3& reference);
double dcm_orthogonality_error(const Matrix3& matrix);
Matrix3 normalize_dcm(const Matrix3& matrix);

class DCM {
public:
    explicit DCM(const Matrix3& matrix = Matrix3::Identity());
    static DCM identity();
    static DCM from_axis_angle(const Vector3& axis, double angle);
    static DCM from_rotation_vector(const Vector3& rotation_vector);
    static DCM from_quaternion(const Quaternion& quaternion);
    static DCM from_dcm(const Matrix3& matrix);
    Matrix3 components() const;
    QuaternionVector hamilton_components() const;
    double norm() const;
    double orthogonality_error() const;
    Matrix3 as_matrix() const;
    Matrix3 to_dcm() const;
    Quaternion to_quaternion() const;
    DCM normalized() const;
    DCM inverse() const;
    DCM conjugate() const;
    DCM multiply(const DCM& other) const;
    Vector3 rotate(const Vector3& vector) const;
    Vector3 inertial_to_body(const Vector3& vector) const;
    Vector3 body_to_inertial(const Vector3& vector) const;
    DCM relative_to(const DCM& reference) const;
private:
    Matrix3 matrix_;
};

QuaternionVector mrp_to_quaternion_parameters(const Vector3& parameters);
Vector3 quaternion_to_mrp_parameters(const QuaternionVector& quaternion);
Vector3 mrp_shadow_set(const Vector3& parameters);
QuaternionVector rodrigues_to_quaternion_parameters(const Vector3& parameters);
Vector3 quaternion_to_rodrigues_parameters(const QuaternionVector& quaternion);

class RotationParameter {
public:
    virtual ~RotationParameter() noexcept = default;
    Vector3 components() const;
    QuaternionVector hamilton_components() const;
    double norm() const;
    Vector3 as_vector() const;
    Quaternion to_quaternion() const;
    DCM to_dcm() const;
    Vector3 rotate(const Vector3& vector) const;
    Vector3 inertial_to_body(const Vector3& vector) const;
    Vector3 body_to_inertial(const Vector3& vector) const;
    Quaternion conjugate() const;
    Quaternion inverse() const;
    Quaternion multiply(const RotationParameter& other) const;
    Quaternion relative_to(const RotationParameter& reference) const;
    virtual QuaternionVector to_hamilton() const = 0;
protected:
    explicit RotationParameter(const Vector3& values);
    Vector3 parameters_;
};

class ModifiedRodriguesParameters final : public RotationParameter {
public:
    explicit ModifiedRodriguesParameters(const Vector3& parameters = Vector3::Zero());
    static ModifiedRodriguesParameters identity();
    static ModifiedRodriguesParameters from_quaternion(const Quaternion& quaternion);
    static ModifiedRodriguesParameters from_quaternion_array(const QuaternionVector& quaternion);
    static ModifiedRodriguesParameters from_axis_angle(const Vector3& axis, double angle);
    static ModifiedRodriguesParameters from_rotation_vector(const Vector3& vector);
    static ModifiedRodriguesParameters from_dcm(const Matrix3& matrix);
    ModifiedRodriguesParameters switch_shadow_set() const;
    ModifiedRodriguesParameters normalized() const;
    QuaternionVector to_hamilton() const override;
};

class RodriguesParameters final : public RotationParameter {
public:
    explicit RodriguesParameters(const Vector3& parameters = Vector3::Zero());
    static RodriguesParameters identity();
    static RodriguesParameters from_quaternion(const Quaternion& quaternion);
    static RodriguesParameters from_quaternion_array(const QuaternionVector& quaternion);
    static RodriguesParameters from_axis_angle(const Vector3& axis, double angle);
    static RodriguesParameters from_rotation_vector(const Vector3& vector);
    static RodriguesParameters from_dcm(const Matrix3& matrix);
    RodriguesParameters normalized() const;
    QuaternionVector to_hamilton() const override;
};
}

#endif  // ADCS_OMNI_HELPERS_ROTATIONS_HPP
