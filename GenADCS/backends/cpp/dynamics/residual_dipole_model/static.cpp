#include "static.hpp"
namespace omni::dynamics {
V residual_dipole(const V &dipole, const V &field, const Q &q) {
  return dipole.cross(rotation(q) * field);
}
} // namespace omni::dynamics
