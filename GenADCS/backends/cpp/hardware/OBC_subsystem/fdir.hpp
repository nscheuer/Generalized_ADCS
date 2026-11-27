#ifndef ADCS_OMNI_HARDWARE_OBC_SUBSYSTEM_FDIR_HPP
#define ADCS_OMNI_HARDWARE_OBC_SUBSYSTEM_FDIR_HPP

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace omni::hardware::obc {

class FDIRMonitor {
public:
  virtual ~FDIRMonitor() = default;
  virtual const std::string &name() const = 0;
};

class AngularRateMonitor final : public FDIRMonitor {
public:
  explicit AngularRateMonitor(double max_rate, std::string name = "angular_rate",
                              std::optional<std::string> target_mode = std::nullopt);

  double max_rate() const { return max_rate_; }
  const std::string &name() const override { return name_; }
  const std::optional<std::string> &target_mode() const { return target_mode_; }

private:
  double max_rate_;
  std::string name_;
  std::optional<std::string> target_mode_;
};

class FDIR {
public:
  explicit FDIR(std::vector<std::shared_ptr<FDIRMonitor>> monitors = {},
                std::string safe_mode = "safe_mode");

  const std::vector<std::shared_ptr<FDIRMonitor>> &monitors() const {
    return monitors_;
  }
  const std::string &safe_mode() const { return safe_mode_; }

private:
  std::vector<std::shared_ptr<FDIRMonitor>> monitors_;
  std::string safe_mode_;
};

} // namespace omni::hardware::obc

#endif
