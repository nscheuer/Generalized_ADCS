#pragma once
#include "../../hardware/ADCS_subsystem/adcs_subsystem.hpp"
#include "_coupled_dynamics.hpp"
#include <array>
#include <map>
#include <string>
namespace omni::propagation {
using SampleRows = std::map<std::string, Eigen::MatrixXd>;
// Prepared once; sensor state and random streams persist between ticks.
class SensorSampler {
public:
  SensorSampler(Runtime &runtime, std::vector<omni::hardware::Sensor> devices,
                double epoch, std::uint64_t seed);
  SampleRows allocate(Eigen::Index rows) const;
  void sample(const Vector &state, double time, std::uint64_t tick,
              SampleRows &output, Eigen::Index row);
  void set_enabled(const std::vector<bool> &enabled);
private:
  Runtime &r;
  std::vector<omni::hardware::Sensor> sensors;
  double epoch;
  std::uint64_t seed;
  std::array<int, 8> counts{};
  std::vector<int> offsets;
  bool need_sun;
};
SampleRows sample_sensors(const Vector &, const Eigen::MatrixXd &, double, double,
                         Runtime &, std::vector<omni::hardware::Sensor>, std::uint64_t);
}
