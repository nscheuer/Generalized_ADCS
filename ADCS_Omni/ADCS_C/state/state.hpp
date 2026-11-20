#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <vector>
#include <map>
#include <string>
#include <pybind11/pybind11.h>

namespace omni {
using Vector3 = std::array<double, 3>;
using Quaternion = std::array<double, 4>;

struct AttitudeState {
    Quaternion orientation;
    Vector3 rates;
    AttitudeState(Quaternion orientation, Vector3 rates);
    Quaternion quaternion_array() const;
};

struct InternalState {
    std::vector<double> wheel_momentum;
    explicit InternalState(std::vector<double> wheel_momentum = {});
    virtual ~InternalState() = default;
};

struct WheelMomentumState : InternalState {
    using InternalState::InternalState;
};

struct OrbitalState {
    std::optional<Vector3> position;
    std::optional<Vector3> velocity;
    double julian_date;
    pybind11::object ephemeris = pybind11::none();
    std::map<std::string, pybind11::object> body_positions;
    std::optional<std::array<double, 6>> keplerian_elements;
    OrbitalState(std::optional<Vector3> position, std::optional<Vector3> velocity,
                 double julian_date, std::optional<std::array<double, 6>> elements = std::nullopt);
    Vector3 cartesian_position() const;
    Vector3 cartesian_velocity() const;
};

struct State {
    AttitudeState attitude;
    OrbitalState orbit;
    InternalState internal;
    State(AttitudeState attitude, OrbitalState orbit, InternalState internal = InternalState{});
    void check_wheel_count(std::optional<std::size_t> count) const;
};
}
