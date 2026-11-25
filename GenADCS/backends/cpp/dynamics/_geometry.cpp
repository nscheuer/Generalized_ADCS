#include "_geometry.hpp"
namespace omni::dynamics {
M rotation(const Q &q) {
  const double w = q[0], x = q[1], y = q[2], z = q[3];
  M r;
  r << w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y),
      2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x),
      2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z;
  return r;
}
} // namespace omni::dynamics
