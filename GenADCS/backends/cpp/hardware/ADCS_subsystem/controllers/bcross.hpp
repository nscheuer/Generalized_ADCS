#ifndef ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_CONTROLLERS_BCROSS_HPP
#define ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_CONTROLLERS_BCROSS_HPP

#include "controller.hpp"

#include <Eigen/Core>
#include <string>
#include <vector>

namespace omni::hardware::controllers {

void bcross_step(double gain,
                 const Eigen::Ref<const Eigen::VectorXd> &readings,
                 const SensorConfig &sensor_config,
                 const Eigen::Ref<const Eigen::MatrixXd> &mtq_axes,
                 const Eigen::Ref<const Eigen::VectorXd> &mtq_max_dipole,
                 Eigen::Ref<Eigen::VectorXd> dipole_commands);

class BCross final : public Controller {
public:
  explicit BCross(double gain);

  int kind() const override { return CTRL_BCROSS; }
  void
  validate_sensors(const std::vector<std::string> &sensor_types) const override;
  Eigen::VectorXd to_params() const override;

  double gain() const { return gain_; }

private:
  double gain_;
};

} // namespace omni::hardware::controllers

#endif
