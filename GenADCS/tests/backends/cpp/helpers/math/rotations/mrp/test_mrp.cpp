#include "../check.hpp"
int main() {
  using P = r::ModifiedRodriguesParameters;
  auto id = P::identity();
  CHECK(id.components().isZero());
  CHECK(id.norm() == 0);
  auto p = P::from_axis_angle(v(0, 0, 1), pi / 2);
  CHECK(p.rotate(v(1, 0, 0)).isApprox(v(0, 1, 0), 1e-12));
  CHECK(p.inertial_to_body(v(1, 0, 0)).isApprox(v(0, 1, 0), 1e-12));
  CHECK(p.body_to_inertial(v(0, 1, 0)).isApprox(v(1, 0, 0), 1e-12));
  CHECK(p.to_dcm().to_dcm().isApprox(p.to_quaternion().to_dcm()));
  CHECK(P::from_quaternion(p.to_quaternion())
            .components()
            .isApprox(p.components()));
  CHECK(P::from_quaternion_array(p.to_hamilton())
            .components()
            .isApprox(p.components()));
  CHECK(P::from_rotation_vector(v(0, 0, pi / 2))
            .components()
            .isApprox(p.components()));
  CHECK(P::from_dcm(p.to_dcm().to_dcm()).components().isApprox(p.components()));
  CHECK(p.inverse().to_dcm().isApprox(p.conjugate().to_dcm()));
  CHECK(p.multiply(p).to_dcm().isApprox(
      p.to_dcm().to_dcm() * p.to_dcm().to_dcm(), 1e-12));
  CHECK(p.relative_to(id).to_dcm().isApprox(p.to_dcm().to_dcm()));
  CHECK(p.normalized().components().isApprox(p.components()));
  auto canonical = P(v(2, 0, 0));
  CHECK(canonical.norm() <= 1);
  CHECK(canonical.components().isApprox(v(-0.5, 0, 0)));
  auto shadow = r::mrp_shadow_set(v(0.2, 0.3, 0.4));
  CHECK(r::mrp_shadow_set(shadow).isApprox(v(0.2, 0.3, 0.4), 1e-12));
  CHECK(same_rotation(r::mrp_to_quaternion_parameters(v(0.2, 0.3, 0.4)),
                      r::mrp_to_quaternion_parameters(shadow)));
  CHECK(same_rotation(r::mrp_to_quaternion_parameters(p.components()),
                      p.to_hamilton()));
  CHECK(r::quaternion_to_mrp_parameters(p.to_hamilton())
            .isApprox(p.components()));
  CHECK(same_rotation(p.switch_shadow_set().to_hamilton(), p.to_hamilton()));
  REJECT(r::mrp_shadow_set(v(0, 0, 0)));
  REJECT(id.switch_shadow_set());
  REJECT(P(v(std::numeric_limits<double>::quiet_NaN(), 0, 0)));
  REJECT(r::quaternion_to_mrp_parameters(q(0, 0, 0, 0)));
}
