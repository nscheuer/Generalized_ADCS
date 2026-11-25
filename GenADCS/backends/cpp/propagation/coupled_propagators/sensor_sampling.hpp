#pragma once
#include "../../hardware/ADCS_subsystem/adcs_subsystem.hpp"
#include "_coupled_dynamics.hpp"
#include <map>
#include <string>
namespace omni::propagation {
std::map<std::string, Eigen::MatrixXd>
sample_sensors(const Vector &, const Eigen::MatrixXd &, double, double,
               Runtime &, std::vector<omni::hardware::Sensor>, std::uint64_t);
}
