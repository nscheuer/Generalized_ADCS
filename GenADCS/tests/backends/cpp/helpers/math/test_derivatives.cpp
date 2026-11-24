#include "derivatives.hpp"
#include "dynamics.hpp"
#include "rotation_types.hpp"
#include "vector.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <stdexcept>
#include <string>
#define CHECK(condition)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " failed: " #condition); \
    } while (false)
#define CHECK_NEAR(actual, expected, tolerance) CHECK(std::abs((actual) - (expected)) <= (tolerance))
namespace math = omni::math;
namespace rotations = omni::rotations;
template <typename Function, typename Input, typename Output>
Eigen::MatrixXd finite_difference_jacobian(Function function, const Input &input, double step = 1e-6)
{
    Eigen::MatrixXd jacobian(Output::RowsAtCompileTime, Input::RowsAtCompileTime);
    for (Eigen::Index column = 0; column < input.size(); ++column)
    {
        Input plus = input, minus = input;
        plus[column] += step;
        minus[column] -= step;
        jacobian.col(column) = (function(plus) - function(minus)) / (2.0 * step);
    }
    return jacobian;
}
void test_derivatives()
{
    Eigen::Vector4d q(0.72, -0.2, 0.31, 0.57);
    q.normalize();
    const Eigen::Vector3d omega(0.4, -0.7, 0.2);
    const Eigen::Vector4d qdot = math::quaternion_derivative(q, omega);
    CHECK(qdot.isApprox(math::quaternion_rate_matrix(q) * omega, 1e-14));
    CHECK(math::angular_velocity_from_quaternion_derivative(q, qdot).isApprox(omega, 1e-14));
    CHECK(std::abs(q.dot(qdot)) < 1e-14);

    Eigen::Vector4d raw(0.4, -1.2, 0.7, 2.0);
    const auto normalize = [](const Eigen::Vector4d &value) { return value.normalized(); };
    const Eigen::Matrix4d normalization_jacobian = math::quaternion_normalization_jacobian(raw);
    const Eigen::MatrixXd normalization_fd =
        finite_difference_jacobian<decltype(normalize), Eigen::Vector4d, Eigen::Vector4d>(normalize, raw);
    CHECK(normalization_jacobian.isApprox(normalization_fd, 2e-9));
    CHECK((normalization_jacobian * raw).norm() < 1e-13);
    CHECK(normalization_jacobian.isApprox(normalization_jacobian.transpose(), 1e-14));
    bool zero_quaternion_rejected = false;
    try
    {
        (void)math::quaternion_normalization_jacobian(Eigen::Vector4d::Zero());
    }
    catch (const std::invalid_argument &)
    {
        zero_quaternion_rejected = true;
    }
    CHECK(zero_quaternion_rejected);

    const Eigen::Vector3d vector(1.3, -0.7, 2.1);
    const auto rotate = [&vector](const Eigen::Vector4d &value) { return rotations::rotate_vector(value, vector); };
    const Eigen::Matrix<double, 3, 4> rotation_jacobian = math::rotated_vector_quaternion_jacobian(q, vector);
    const Eigen::MatrixXd rotation_fd =
        finite_difference_jacobian<decltype(rotate), Eigen::Vector4d, Eigen::Vector3d>(rotate, q);
    CHECK(rotation_jacobian.isApprox(rotation_fd, 2e-8));

    const double half = std::sqrt(0.5);
    CHECK(math::quaternion_to_euler321(Eigen::Vector4d(1, 0, 0, 0)).norm() < 1e-14);
    CHECK(math::quaternion_to_euler321(Eigen::Vector4d(half, half, 0, 0))
              .isApprox(Eigen::Vector3d(M_PI / 2, 0, 0), 1e-14));
    CHECK(math::quaternion_to_euler321(Eigen::Vector4d(half, 0, 0, half))
              .isApprox(Eigen::Vector3d(0, 0, M_PI / 2), 1e-14));
}
int main()
{
    void test_derivatives();
}
