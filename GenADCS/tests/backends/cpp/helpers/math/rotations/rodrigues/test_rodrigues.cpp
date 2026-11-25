#include "../check.hpp"
int main() {
  using P = r::RodriguesParameters;
  auto id = P::identity();
  CHECK(id.components().isZero());
  auto p = P::from_axis_angle(v(0, 0, 1), M_PI / 2);
  CHECK(p.rotate(v(1, 0, 0)).isApprox(v(0, 1, 0), 1e-12));
  CHECK(p.inertial_to_body(v(1, 0, 0)).isApprox(v(0, 1, 0), 1e-12));
  CHECK(p.body_to_inertial(v(0, 1, 0)).isApprox(v(1, 0, 0), 1e-12));
  CHECK(P::from_rotation_vector(v(0, 0, M_PI / 2))
            .components()
            .isApprox(p.components()));
  CHECK(P::from_quaternion(p.to_quaternion())
            .components()
            .isApprox(p.components()));
  CHECK(P::from_quaternion_array(p.to_hamilton())
            .components()
            .isApprox(p.components()));
  CHECK(P::from_dcm(p.to_dcm().to_dcm()).components().isApprox(p.components()));
  CHECK(p.inverse().to_dcm().isApprox(p.conjugate().to_dcm()));
  CHECK(p.relative_to(id).to_dcm().isApprox(p.to_dcm().to_dcm()));
  CHECK(p.normalized().components().isApprox(p.components()));
  CHECK(r::quaternion_to_rodrigues_parameters(p.to_hamilton())
            .isApprox(p.components()));
  CHECK(same_rotation(r::rodrigues_to_quaternion_parameters(p.components()),
                      p.to_hamilton()));
  CHECK(r::quaternion_to_rodrigues_parameters(-p.to_hamilton())
            .isApprox(p.components()));
  REJECT(r::quaternion_to_rodrigues_parameters(q(0, 1, 0, 0)));
  REJECT(P::from_axis_angle(v(1, 0, 0), M_PI));
  REJECT(P(v(std::numeric_limits<double>::infinity(), 0, 0)));
}
