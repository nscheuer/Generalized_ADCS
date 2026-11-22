#pragma once
#include <pybind11/pybind11.h>
#include <vector>
namespace omni::config {
struct DynamicsConfig { std::vector<pybind11::object> forces, torques; };
}
