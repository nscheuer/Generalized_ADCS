#include "../check.hpp"
int main() {
    const auto identity = r::DCM::identity();
    CHECK(identity.components().isApprox(r::Matrix3::Identity()));
    CHECK(identity.norm() < 1e-14);
    CHECK(identity.orthogonality_error() < 1e-14);
    auto a = r::DCM::from_axis_angle(v(0,0,2), M_PI/2);
    auto b = r::DCM::from_rotation_vector(v(0,0,M_PI/2));
    CHECK(a.as_matrix().isApprox(b.to_dcm(), 1e-12));
    CHECK(r::DCM::from_quaternion(r::Quaternion::from_axis_angle(v(0,0,1), M_PI/2)).to_dcm().isApprox(a.to_dcm()));
    CHECK(r::DCM::from_dcm(a.to_dcm()).to_dcm().isApprox(a.to_dcm()));
    CHECK(a.rotate(v(1,0,0)).isApprox(v(0,1,0), 1e-12));
    CHECK(a.inertial_to_body(v(1,0,0)).isApprox(v(0,1,0), 1e-12));
    CHECK(a.body_to_inertial(v(0,1,0)).isApprox(v(1,0,0), 1e-12));
    CHECK(a.inverse().multiply(a).to_dcm().isApprox(r::Matrix3::Identity(), 1e-12));
    CHECK(a.conjugate().to_dcm().isApprox(a.inverse().to_dcm()));
    CHECK(a.normalized().to_dcm().isApprox(a.to_dcm()));
    CHECK(a.relative_to(identity).to_dcm().isApprox(a.to_dcm()));
    CHECK(a.multiply(b).to_dcm().isApprox(a.to_dcm()*b.to_dcm(), 1e-12));
    CHECK(same_rotation(a.hamilton_components(), a.to_quaternion().hamilton_components()));
    CHECK(r::axis_angle_to_dcm(v(0,0,1), M_PI/2).isApprox(a.to_dcm()));
    CHECK(r::rotate_vector(a.to_dcm(),v(1,0,0)).isApprox(v(0,1,0),1e-12));
    CHECK(r::inverse_dcm(a.to_dcm()).isApprox(a.to_dcm().transpose()));
    CHECK(r::compose_dcm(a.to_dcm(),b.to_dcm()).isApprox(a.to_dcm()*b.to_dcm()));
    CHECK(r::relative_dcm(a.to_dcm(),b.to_dcm()).isApprox(a.to_dcm()*b.to_dcm().transpose()));
    CHECK(r::dcm_orthogonality_error(a.to_dcm()) < 1e-12);
    auto drift = a.to_dcm(); drift(0,0)+=1e-4; drift(1,1)-=1e-4;
    auto repaired = r::normalize_dcm(drift);
    CHECK(r::dcm_orthogonality_error(repaired)<1e-12);
    CHECK(std::abs(repaired.determinant()-1)<1e-12);
    r::Matrix3 reflected=r::Matrix3::Identity(); reflected(0,0)=-1;
    REJECT(r::DCM(reflected));
    auto scaled=2*r::Matrix3::Identity().eval(); REJECT(r::DCM(scaled));
    REJECT(r::axis_angle_to_dcm(v(0,0,0),1));
    REJECT(r::axis_angle_to_dcm(v(1,0,0),std::numeric_limits<double>::infinity()));
    r::Matrix3 degenerate=r::Matrix3::Zero(); REJECT(r::normalize_dcm(degenerate));
    degenerate.col(0)=v(1,0,0); degenerate.col(1)=v(2,0,0); REJECT(r::normalize_dcm(degenerate));
    r::Matrix3 nonfinite=r::Matrix3::Identity(); nonfinite(0,0)=std::numeric_limits<double>::quiet_NaN(); REJECT(r::DCM(nonfinite));
}
