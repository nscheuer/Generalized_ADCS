#include "fdir.hpp"

#include <cmath>
#include <set>
#include <stdexcept>
#include <utility>

namespace omni::hardware::obc {

AngularRateMonitor::AngularRateMonitor(double max_rate, std::string name,
                                       std::optional<std::string> target_mode)
    : max_rate_(max_rate), name_(std::move(name)),
      target_mode_(std::move(target_mode)) {
  if (!std::isfinite(max_rate_) || max_rate_ <= 0.0) {
    throw std::invalid_argument("max_rate must be finite and positive rad/s");
  }
  if (name_.empty()) {
    throw std::invalid_argument("monitor name must be nonempty");
  }
}

FDIR::FDIR(std::vector<std::shared_ptr<FDIRMonitor>> monitors,
           std::string safe_mode)
    : monitors_(std::move(monitors)), safe_mode_(std::move(safe_mode)) {
  if (safe_mode_.empty()) {
    throw std::invalid_argument("safe_mode must be a nonempty mode name");
  }
  std::set<std::string> seen;
  for (const auto &mon : monitors_) {
    if (!mon) {
      throw std::invalid_argument("FDIR monitor must not be null");
    }
    if (!seen.insert(mon->name()).second) {
      throw std::invalid_argument("duplicate FDIR monitor name: " +
                                  mon->name());
    }
  }
}

} // namespace omni::hardware::obc
