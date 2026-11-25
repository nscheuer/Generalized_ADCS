#pragma once
#include "../../native_models.hpp"
namespace omni::environment {
double nrlmsise_density_geodetic(int, double, double, double, double, double,
                                 double, double, bool,
                                 const std::array<double, 7> &);
double nrlmsise_density(const Eigen::Vector3d &, double, double, double, double,
                        bool, const std::array<double, 7> &);
} // namespace omni::environment
