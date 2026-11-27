#ifndef ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_CONTROLLERS_DESATURATION_HOGANSCHAUBDESATURATION_HPP
#define ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_CONTROLLERS_DESATURATION_HOGANSCHAUBDESATURATION_HPP

#include "../controller.hpp"
#include "desaturation.hpp"

#include <Eigen/Core>

namespace omni::hardware::controllers {

Eigen::VectorXd hogan_schaub_u0(
    const Eigen::Ref<const Eigen::VectorXd> &wheel_momenta, double gain,
    double target_momentum);

void hogan_schaub_mtq_dump(
    const Eigen::Ref<const Eigen::VectorXd> &wheel_momenta,
    const Eigen::Ref<const Eigen::MatrixXd> &wheel_axes, double mtq_gain,
    double target_momentum, const Eigen::Ref<const Eigen::VectorXd> &readings,
    const SensorConfig &sensor_config,
    const Eigen::Ref<const Eigen::MatrixXd> &mtq_axes,
    const Eigen::Ref<const Eigen::VectorXd> &mtq_max_dipole,
    Eigen::Ref<Eigen::VectorXd> dipole_commands);

class HoganSchaubManagement final : public Desaturation {
public:
  explicit HoganSchaubManagement(double gain, double target_momentum = 0.0,
                                 double mtq_gain = 0.0);

  int kind() const override { return DESAT_HOGAN_SCHAUB; }
  Eigen::VectorXd to_params() const override;

  double gain() const { return gain_; }
  double target_momentum() const { return target_momentum_; }
  double mtq_gain() const { return mtq_gain_; }

private:
  double gain_;
  double target_momentum_;
  double mtq_gain_;
};

} // namespace omni::hardware::controllers

#endif
