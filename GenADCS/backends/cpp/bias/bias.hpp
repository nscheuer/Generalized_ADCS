#pragma once

#include <pybind11/pybind11.h>
#include <string>

namespace omni::bias {

// The Python object owns the shared NumPy/Covariance implementation. This
// native facade keeps the C++ backend API and forwards calls through bindings.
struct Bias {
    explicit Bias(pybind11::object implementation);
    pybind11::object implementation;

    Bias copy() const;
    Bias broadcast_to(int length, const std::string& owner) const;
    pybind11::object process_covariance(double dt, const std::string& form) const;
    pybind11::object process_psd(const std::string& form) const;
    pybind11::object update_bias(double j2000, pybind11::object rng) const;
    pybind11::object get_bias(double j2000) const;
    pybind11::object apply(pybind11::object value, double dt, pybind11::object rng) const;
};

}  // namespace omni::bias
