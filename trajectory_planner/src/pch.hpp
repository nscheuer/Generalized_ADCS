// Precompiled header for trajectory_planner
// Contains heavy includes that rarely change to speed up compilation
#ifndef PCH_HPP
#define PCH_HPP

// Standard library
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

// Armadillo (linear algebra library - heavy include)
#include <armadillo>

// pybind11 (Python bindings - heavy include)
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#endif // PCH_HPP
