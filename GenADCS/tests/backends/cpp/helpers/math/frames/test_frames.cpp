#include "frames.hpp"
#include <Eigen/LU>
#include <cmath>
#include <limits>
#include <stdexcept>

#define CHECK(x) do { if (!(x)) throw std::runtime_error("failed: " #x); } while (false)
using namespace omni::frames;

int main() {
    Quaternion identity; identity << 1, 0, 0, 0;
    CHECK(inertial_to_body_matrix(identity).isApprox(Matrix3::Identity(), 1e-14));
    CHECK(body_to_inertial_matrix(identity).isApprox(Matrix3::Identity(), 1e-14));
    Quaternion quarter_turn; quarter_turn << std::sqrt(0.5), 0, 0, std::sqrt(0.5);
    Vector3 x; x << 1, 0, 0;
    Vector3 y; y << 0, 1, 0;
    CHECK(inertial_to_body(x, quarter_turn).isApprox(y, 1e-14));
    CHECK(body_to_inertial(y, quarter_turn).isApprox(x, 1e-14));
    const auto rotation = inertial_to_body_matrix(quarter_turn);
    CHECK((rotation * rotation.transpose()).isApprox(Matrix3::Identity(), 1e-14));
    CHECK(std::abs(rotation.determinant() - 1.0) < 1e-14);

    Matrix3 covariance; covariance << 4, 1, 0, 1, 2, 0, 0, 0, 3;
    const auto transformed = transform_covariance(covariance, rotation);
    CHECK(transformed.isApprox(rotation * covariance * rotation.transpose(), 1e-14));
    CHECK(transformed.isApprox(transformed.transpose(), 1e-14));
    CHECK((rotation * x).isApprox(transform_vector(x, rotation), 1e-14));

    Quaternion invalid = Quaternion::Zero();
    bool rejected = false;
    try { (void)inertial_to_body_matrix(invalid); } catch (const std::exception&) { rejected = true; }
    CHECK(!rejected); // Low-level kernel is polynomial and does not normalize/validate.
    invalid[0] = std::numeric_limits<double>::quiet_NaN();
    rejected = false;
    try { (void)inertial_to_body_matrix(invalid); } catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
}
