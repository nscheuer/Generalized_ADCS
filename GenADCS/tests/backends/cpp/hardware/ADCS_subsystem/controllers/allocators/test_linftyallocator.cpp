#include "check.hpp"
#include "hardware/ADCS_subsystem/controllers/allocator/l2allocator.hpp"
#include "hardware/ADCS_subsystem/controllers/allocator/linftyallocator.hpp"

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
  const Eigen::MatrixXd null_basis = svd.matrixV().rightCols(1);
  const Eigen::VectorXd max_t = Eigen::VectorXd::Ones(4);
  const Eigen::Vector3d tau_c(0.3, 0.0, 0.0);

  const Eigen::VectorXd u_l2 =
      allocate_l2(tau_c, a_pinv, null_proj, Eigen::VectorXd::Zero(4), max_t);
  const Eigen::VectorXd u_linf = allocate_linfinity(
      tau_c, a_pinv, null_basis, null_proj, Eigen::VectorXd::Zero(4), max_t);
  CHECK(LInfinityAllocator().kind() == 1);
  CHECK((a_mat * u_linf).isApprox(tau_c, 1e-9));
  CHECK(u_linf.cwiseAbs().maxCoeff() < u_l2.cwiseAbs().maxCoeff());

  Eigen::Matrix<double, 3, 5> a5;
  a5 << 1.0, 0.0, 0.0, 0.5, -0.5, 0.0, 1.0, 0.0, 0.5, 0.5, 0.0, 0.0, 1.0,
      0.70710678, 0.70710678;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd5(
      a5, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::MatrixXd s_pinv5 = Eigen::MatrixXd::Zero(5, 3);
  for (int i = 0; i < 3; ++i) {
    s_pinv5(i, i) = 1.0 / svd5.singularValues()[i];
  }
  const Eigen::MatrixXd a_pinv5 =
      svd5.matrixV() * s_pinv5 * svd5.matrixU().transpose();
  const Eigen::MatrixXd null_proj5 =
      Eigen::MatrixXd::Identity(5, 5) - a_pinv5 * a5;
  const Eigen::MatrixXd null_basis5 = svd5.matrixV().rightCols(2);
  const Eigen::Vector3d tau5(0.4, 0.1, -0.1);
  const Eigen::VectorXd max_t5 = Eigen::VectorXd::Ones(5);

  const Eigen::VectorXd u5_l2 = allocate_l2(
      tau5, a_pinv5, null_proj5, Eigen::VectorXd::Zero(5), max_t5);
  const Eigen::VectorXd u5_linf = allocate_linfinity(
      tau5, a_pinv5, null_basis5, null_proj5, Eigen::VectorXd::Zero(5), max_t5);
  CHECK((a5 * u5_linf).isApprox(tau5, 1e-8));
  CHECK(u5_linf.cwiseAbs().maxCoeff() <=
        u5_l2.cwiseAbs().maxCoeff() + 1e-9);
  return 0;
}
