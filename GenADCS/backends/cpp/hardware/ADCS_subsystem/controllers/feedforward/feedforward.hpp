#ifndef ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_CONTROLLERS_FEEDFORWARD_FEEDFORWARD_HPP
#define ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_CONTROLLERS_FEEDFORWARD_FEEDFORWARD_HPP

#include <Eigen/Core>

namespace omni::hardware::controllers {

inline constexpr int FF_PASSTHROUGH = 0;

class Feedforward {
public:
  virtual ~Feedforward() = default;
  virtual int kind() const = 0;
  virtual Eigen::VectorXd to_params() const = 0;
};

} // namespace omni::hardware::controllers

#endif
