#pragma once
#include "rotation_types.hpp"
#include <Eigen/LU>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#define CHECK(x)                                                               \
  do {                                                                         \
    if (!(x))                                                                  \
      throw std::runtime_error(std::string(__FILE__) + ":" +                   \
                               std::to_string(__LINE__) + " failed: " #x);     \
  } while (false)
#define REJECT(x)                                                              \
  do {                                                                         \
    bool caught = false;                                                       \
    try {                                                                      \
      (void)(x);                                                               \
    } catch (const std::invalid_argument &) {                                  \
      caught = true;                                                           \
    }                                                                          \
    CHECK(caught);                                                             \
  } while (false)
namespace r = omni::rotations;
inline constexpr double pi = 3.14159265358979323846;
inline r::Vector3 v(double x, double y, double z) {
  return r::Vector3(x, y, z);
}
inline r::QuaternionVector q(double w, double x, double y, double z) {
  return r::QuaternionVector(w, x, y, z);
}
inline bool same_rotation(const r::QuaternionVector &a,
                          const r::QuaternionVector &b) {
  return std::abs(r::normalize_quaternion(a).dot(r::normalize_quaternion(b))) >
         1.0 - 1e-12;
}
