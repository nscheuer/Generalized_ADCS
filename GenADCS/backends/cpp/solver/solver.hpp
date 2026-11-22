#pragma once
#include <pybind11/pybind11.h>
namespace omni::solver {struct Solver {pybind11::object dynamics,orbital,attitude,coupled;Solver(pybind11::object,pybind11::object,pybind11::object,pybind11::object);};}
