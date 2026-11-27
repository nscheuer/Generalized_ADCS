#ifndef ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_CONTROLLERS_DESATURATION_DESATURATION_HPP
#define ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_CONTROLLERS_DESATURATION_DESATURATION_HPP

#include <Eigen/Core>

namespace omni::hardware::controllers {

inline constexpr int DESAT_NONE = 0;
inline constexpr int DESAT_HOGAN_SCHAUB = 1;

class Desaturation {
public:
  virtual ~Desaturation() = default;
  virtual int kind() const = 0;
  virtual Eigen::VectorXd to_params() const = 0;
};

class NoDesaturation final : public Desaturation {
public:
  int kind() const override { return DESAT_NONE; }
  Eigen::VectorXd to_params() const override { return Eigen::VectorXd::Zero(3); }
};

} // namespace omni::hardware::controllers

#endif
