#pragma once

#include <pybind11/pybind11.h>

namespace omni::bias {

struct Bias {
    virtual ~Bias() = default;
    virtual pybind11::object apply(const pybind11::object& value, double dt,
                                   const pybind11::object& rng) const;
};

[[noreturn]] void raise_not_implemented();

}  // namespace omni::bias
