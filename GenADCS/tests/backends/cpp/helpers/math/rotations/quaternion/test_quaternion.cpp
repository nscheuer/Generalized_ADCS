#include "../check.hpp"
int main() {
  const auto id = r::Quaternion::identity();
  CHECK(id.components().isApprox(q(1, 0, 0, 0)));
  CHECK(std::abs(id.norm() - 1) < 1e-14);
  auto a = r::Quaternion::from_axis_angle(v(0, 0, 2), M_PI / 2);
  auto b = r::Quaternion::from_rotation_vector(v(0, 0, M_PI / 2));
  CHECK(same_rotation(a.components(), b.components()));
  auto j = a.to_convention("JPL");
  CHECK(j.convention() == "JPL");
  CHECK(same_rotation(j.hamilton_components(), a.hamilton_components()));
  CHECK(j.components().isApprox(r::hamilton_to_jpl(a.components())));
  CHECK(j.to_convention("Hamilton").components().isApprox(a.components()));
  CHECK(r::jpl_to_hamilton(j.components()).isApprox(a.components()));
  CHECK(a.rotate(v(1, 0, 0)).isApprox(v(0, 1, 0), 1e-12));
  CHECK(a.inertial_to_body(v(1, 0, 0)).isApprox(v(0, 1, 0), 1e-12));
  CHECK(a.body_to_inertial(v(0, 1, 0)).isApprox(v(1, 0, 0), 1e-12));
  CHECK(
      a.inverse().multiply(a).to_dcm().isApprox(r::Matrix3::Identity(), 1e-12));
  CHECK(a.conjugate().components().isApprox(a.inverse().components()));
  CHECK(a.normalized().to_dcm().isApprox(a.to_dcm()));
  CHECK(a.relative_to(id).to_dcm().isApprox(a.to_dcm()));
  CHECK(
      r::Quaternion::from_dcm(a.to_dcm()).to_dcm().isApprox(a.to_dcm(), 1e-12));
  CHECK(r::quaternion_to_dcm(a.components()).isApprox(a.to_dcm()));
  CHECK(same_rotation(r::dcm_to_quaternion(a.to_dcm()), a.components()));
  CHECK(
      r::rotate_vector(a.components(), v(1, 0, 0)).isApprox(v(0, 1, 0), 1e-12));
  CHECK(r::multiply(a.components(), b.components())
            .isApprox(a.multiply(b).components()));
  CHECK(r::inverse(q(2, 0, 0, 0)).isApprox(q(0.5, 0, 0, 0)));
  CHECK(r::conjugate(q(1, 2, 3, 4)).isApprox(q(1, -2, -3, -4)));
  CHECK(r::quaternion_to_rotation_vector(
            r::rotation_vector_to_quaternion(v(1e-14, 0, 0)))
            .isApprox(v(1e-14, 0, 0), 1e-12));
  CHECK(r::quaternion_to_rotation_vector(a.components())
            .isApprox(v(0, 0, M_PI / 2), 1e-12));
  for (double t : {-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5}) {
    auto s = r::slerp(id.components(), a.components(), t);
    CHECK(same_rotation(s,
                        r::axis_angle_to_quaternion(v(0, 0, 1), t * M_PI / 2)));
    CHECK(std::abs(s.norm() - 1) < 1e-12);
  }
  CHECK(same_rotation(r::slerp(id.components(), -id.components(), 0.5),
                      id.components()));
  CHECK(same_rotation(r::slerp(id.components(), q(1, 1e-8, 0, 0), 0.5),
                      q(1, 0.5e-8, 0, 0)));
  REJECT(r::Quaternion(q(0, 0, 0, 0)));
  REJECT(r::normalize_quaternion(q(0, 0, 0, 0)));
  REJECT(r::inverse(q(0, 0, 0, 0)));
  REJECT(r::slerp(q(0, 0, 0, 0), id.components(), 0.5));
  REJECT(r::Quaternion(q(1, 0, 0, 0), "bad"));
  REJECT(a.as_convention("bad"));
  REJECT(r::Quaternion::from_axis_angle(v(0, 0, 0), 1));
  r::Matrix3 reflected = r::Matrix3::Identity();
  reflected(0, 0) = -1;
  REJECT(r::Quaternion::from_dcm(reflected));
  REJECT(r::Quaternion(q(1, std::numeric_limits<double>::quiet_NaN(), 0, 0)));
}
