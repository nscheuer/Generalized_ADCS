#include "check.hpp"
#include "hardware/ADCS_subsystem/controllers/allocator/l2allocator.hpp"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <cmath>

using namespace omni::hardware::controllers;

int main() {
  const double s3 = 1.0 / std::sqrt(3.0);
  Eigen::Matrix<double, 3, 4> a_mat;
  a_mat << 1.0, 0.0, 0.0, s3, 0.0, 1.0, 0.0, s3, 0.0, 0.0, 1.0, s3;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      a_mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::MatrixXd s_pinv = Eigen::MatrixXd::Zero(4, 3);
  for (int i = 0; i < 3; ++i) {
    s_pinv(i, i) = 1.0 / svd.singularValues()[i];
  }
  const Eigen::MatrixXd a_pinv =
      svd.matrixV() * s_pinv * svd.matrixU().transpose();
  const Eigen::MatrixXd null_proj =
      Eigen::MatrixXd::Identity(4, 4) - a_pinv * a_mat;

  const Eigen::Vector3d tau_c(0.3, -0.1, 0.2);
  const Eigen::VectorXd u =
      allocate_l2(tau_c, a_pinv, null_proj, Eigen::VectorXd::Zero(4),
                  Eigen::VectorXd::Ones(4));
  CHECK(L2Allocator().kind() == 0);
  CHECK((a_mat * u).isApprox(tau_c, 1e-12));

  const Eigen::VectorXd u_sat = allocate_l2(
      Eigen::Vector3d(2.0, 0.0, 0.0), a_pinv, null_proj,
      Eigen::VectorXd::Zero(4), Eigen::VectorXd::Constant(4, 0.2));
  CHECK((u_sat.cwiseAbs().array() <= 0.2 + 1e-12).all());
  return 0;
}
