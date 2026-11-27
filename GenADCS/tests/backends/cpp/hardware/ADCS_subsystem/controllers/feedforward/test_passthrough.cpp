#include "check.hpp"
#include "hardware/ADCS_subsystem/controllers/feedforward/passthrough.hpp"

#include <Eigen/Core>
#include <limits>

using namespace omni::hardware::controllers;

int main() {
  Passthrough ff;
  CHECK(ff.to_params().size() == 1);

  const Eigen::Vector4d q_cur(1.0, 0.0, 0.0, 0.0);
  const Eigen::Vector4d q_g(0.0, 1.0, 0.0, 0.0);
  const Eigen::Vector3d w_g(0.01, -0.02, 0.03);
  const auto [qd, wd, wdd, fft] = passthrough_step(q_g, w_g, q_cur);
  CHECK(qd.isApprox(q_g, 1e-12));
  CHECK(wd.isApprox(w_g, 1e-12));
  CHECK(wdd.isApprox(Eigen::Vector3d::Zero(), 1e-12));
  CHECK(fft.isApprox(Eigen::Vector3d::Zero(), 1e-12));

  const double nan = std::numeric_limits<double>::quiet_NaN();
  const Eigen::Vector4d nan_q = Eigen::Vector4d::Constant(nan);
  const Eigen::Vector3d nan_w = Eigen::Vector3d::Constant(nan);
  const auto [qd2, wd2, wdd2, fft2] = passthrough_step(nan_q, nan_w, q_cur);
  (void)wdd2;
  (void)fft2;
  CHECK(qd2.isApprox(q_cur, 1e-12));
  CHECK(wd2.isApprox(Eigen::Vector3d::Zero(), 1e-12));
  return 0;
}
