#ifndef ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_CONTROLLERS_FEEDBACK_FEEDBACK_HPP
#define ADCS_OMNI_HARDWARE_ADCS_SUBSYSTEM_CONTROLLERS_FEEDBACK_FEEDBACK_HPP

#include <Eigen/Core>
#include <string>
#include <utility>

namespace omni::hardware::controllers {

inline constexpr int FB_MRP_FEEDBACK_LINEARIZED = 1;

Eigen::Matrix3d to_gain_matrix(double value, const std::string &name,
                               bool require_spd);
Eigen::Matrix3d to_gain_matrix(const Eigen::Vector3d &value,
                               const std::string &name, bool require_spd);
Eigen::Matrix3d to_gain_matrix(const Eigen::Matrix3d &value,
                               const std::string &name, bool require_spd);

class Feedback {
public:
  virtual ~Feedback() = default;
  virtual int kind() const = 0;
  virtual std::pair<Eigen::Matrix3d, Eigen::Matrix3d> to_matrices() const = 0;
};

} // namespace omni::hardware::controllers

#endif
