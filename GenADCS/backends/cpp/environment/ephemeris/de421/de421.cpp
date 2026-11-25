#include "de421.hpp"
#include "../../native_helpers.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
namespace omni::environment {
constexpr double kJ2000 = 2451545.0;
constexpr double kSecondsPerDay = 86400.0;
DE421::DE421(const std::filesystem::path &supplied) {
  filepath_ = supplied.empty()
                  ? environment_data_path("ephemeris/de421/de421.bsp")
                  : supplied;
  std::ifstream file(filepath_, std::ios::binary);
  if (!file) {
    throw std::runtime_error("cannot open DE421 kernel: " + filepath_.string());
  }
  std::vector<unsigned char> raw((std::istreambuf_iterator<char>(file)), {});
  if (raw.size() < 1024U ||
      std::string(raw.begin(), raw.begin() + 8) != "DAF/SPK ") {
    throw std::runtime_error("invalid DAF/SPK kernel");
  }
  const std::string marker(raw.begin() + 88, raw.begin() + 96);
  const bool little = marker == "LTL-IEEE";
  if (!little && marker != "BIG-IEEE") {
    throw std::runtime_error("unsupported DAF byte order");
  }
  const int nd = static_cast<int32_t>(read_u32(raw, 8, little)),
            ni = static_cast<int32_t>(read_u32(raw, 12, little));
  if (nd != 2 || ni != 6) {
    throw std::runtime_error("unsupported DAF summary format");
  }
  int record = static_cast<int32_t>(read_u32(raw, 76, little));
  const std::size_t summary_size = nd * 8U + ni * 4U;
  while (record != 0) {
    const std::size_t base = (static_cast<std::size_t>(record) - 1U) * 1024U;
    if (base + 1024U > raw.size()) {
      throw std::runtime_error("invalid DAF summary record");
    }
    const int next = static_cast<int>(read_f64(raw, base, little));
    const double countd = read_f64(raw, base + 16U, little);
    if (countd < 0.0 || countd > 25.0) {
      throw std::runtime_error("invalid DAF summary count");
    }
    const int count = static_cast<int>(countd);
    for (int i = 0; i < count; ++i) {
      const std::size_t off =
          base + 24U + static_cast<std::size_t>(i) * summary_size;
      const double start = read_f64(raw, off, little),
                   end = read_f64(raw, off + 8U, little);
      const std::size_t ints = off + 16U;
      const int target = static_cast<int32_t>(read_u32(raw, ints, little));
      const int center = static_cast<int32_t>(read_u32(raw, ints + 4U, little));
      const int type = static_cast<int32_t>(read_u32(raw, ints + 12U, little));
      const int first = static_cast<int32_t>(read_u32(raw, ints + 16U, little));
      const int last = static_cast<int32_t>(read_u32(raw, ints + 20U, little));
      if (type != 2 && type != 3) {
        continue;
      }
      if (first < 1 || last < first ||
          static_cast<std::size_t>(last) * 8U > raw.size()) {
        throw std::runtime_error("invalid SPK data addresses");
      }
      const std::size_t offset = data_.size(),
                        length = static_cast<std::size_t>(last - first + 1);
      for (std::size_t j = 0; j < length; ++j) {
        data_.push_back(read_f64(
            raw, (static_cast<std::size_t>(first) - 1U + j) * 8U, little));
      }
      segments_.push_back(
          Segment{target, center, type, offset, length, start, end});
    }
    record = next;
  }
  if (segments_.empty()) {
    throw std::runtime_error("no supported SPK segments");
  }
}

DE421::State DE421::segment_state(int target, double et) const {
  const Segment *selected = nullptr;
  for (const auto &s : segments_) {
    if (s.target == target && s.start <= et && et <= s.end) {
      selected = &s;
    }
  }
  if (selected == nullptr) {
    throw std::out_of_range("no DE421 segment covers target and epoch");
  }
  const std::size_t trailer = selected->offset + selected->size - 4U;
  const double initial = data_[trailer], length = data_[trailer + 1U];
  const int record_size = static_cast<int>(data_[trailer + 2U]),
            count = static_cast<int>(data_[trailer + 3U]);
  int rec = static_cast<int>((et - initial) / length);
  rec = std::clamp(rec, 0, count - 1);
  const std::size_t base =
      selected->offset + static_cast<std::size_t>(rec * record_size);
  const double mid = data_[base], radius = data_[base + 1U],
               x = (et - mid) / radius;
  const int ncf = (record_size - 2) / (selected->type == 2 ? 3 : 6);
  State result{Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
  for (int axis = 0; axis < 3; ++axis) {
    const std::size_t po = base + 2U + static_cast<std::size_t>(axis * ncf);
    double t0 = 1.0, t1 = x, d0 = 0.0, d1 = 1.0;
    result.position[axis] = data_[po];
    if (ncf > 1) {
      result.position[axis] += data_[po + 1U] * x;
    }
    if (selected->type == 3) {
      const std::size_t vo =
          base + 2U + static_cast<std::size_t>((axis + 3) * ncf);
      result.velocity[axis] = data_[vo] + data_[vo + 1U] * x;
    }
    for (int k = 2; k < ncf; ++k) {
      const double t2 = 2.0 * x * t1 - t0, d2 = 2.0 * t1 + 2.0 * x * d1 - d0;
      result.position[axis] += data_[po + static_cast<std::size_t>(k)] * t2;
      if (selected->type == 3) {
        result.velocity[axis] +=
            data_[base + 2U + static_cast<std::size_t>((axis + 3) * ncf) +
                  static_cast<std::size_t>(k)] *
            t2;
      }
      t0 = t1;
      t1 = t2;
      d0 = d1;
      d1 = d2;
    }
    if (selected->type == 2) {
      double derivative = ncf > 1 ? data_[po + 1U] : 0.0;
      t0 = 1.0;
      t1 = x;
      d0 = 0.0;
      d1 = 1.0;
      for (int k = 2; k < ncf; ++k) {
        const double t2 = 2.0 * x * t1 - t0, d2 = 2.0 * t1 + 2.0 * x * d1 - d0;
        derivative += data_[po + static_cast<std::size_t>(k)] * d2;
        t0 = t1;
        t1 = t2;
        d0 = d1;
        d1 = d2;
      }
      result.velocity[axis] = derivative / radius;
    }
  }
  return result;
}
DE421::State DE421::barycentric_state(int target, double et) const {
  State sum{Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
  int node = target;
  for (int depth = 0; depth < 8 && node != 0; ++depth) {
    const State s = segment_state(node, et);
    sum.position += s.position;
    sum.velocity += s.velocity;
    const Segment *found = nullptr;
    for (const auto &seg : segments_) {
      if (seg.target == node && seg.start <= et && et <= seg.end) {
        found = &seg;
      }
    }
    if (found == nullptr) {
      throw std::out_of_range("missing DE421 center");
    }
    node = found->center;
  }
  if (node != 0) {
    throw std::runtime_error("DE421 center chain too deep");
  }
  return sum;
}
int DE421::body_target(const std::string &name) const {
  static const std::map<std::string, int> names{
      {"mercury", 1}, {"venus", 2},  {"earth-moon barycenter", 3},
      {"emb", 3},     {"mars", 4},   {"jupiter", 5},
      {"saturn", 6},  {"uranus", 7}, {"neptune", 8},
      {"pluto", 9},   {"moon", 301}, {"sun", 10},
      {"earth", 399}};
  auto it = names.find(name);
  if (it == names.end()) {
    throw std::invalid_argument("unsupported DE421 body: " + name);
  }
  return it->second;
}
std::pair<Eigen::Vector3d, Eigen::Vector3d>
DE421::state_eci(const std::string &target, const double jd,
                 const std::string &observer) const {
  const int t = body_target(target), o = body_target(observer);
  const double et = (jd - kJ2000) * kSecondsPerDay;
  State ts = t == 399 ? State{Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()}
                      : barycentric_state(t, et);
  State os = o == 399 ? State{Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()}
                      : barycentric_state(o, et);
  if (t == 399) {
    ts = barycentric_state(399, et);
  }
  if (o == 399) {
    os = barycentric_state(399, et);
  }
  return {ts.position - os.position, ts.velocity - os.velocity};
}
Eigen::Vector3d DE421::position_eci(const std::string &t, double jd,
                                    const std::string &o) const {
  return state_eci(t, jd, o).first;
}
Eigen::Vector3d DE421::velocity_eci(const std::string &t, double jd,
                                    const std::string &o) const {
  return state_eci(t, jd, o).second;
}
Eigen::Vector3d DE421::position(int id, double jd) const {
  if (id == 1) {
    return Eigen::Vector3d::Zero();
  }
  if (id == 0) {
    return position_eci("sun", jd);
  }
  if (id == 2) {
    return position_eci("moon", jd);
  }
  throw std::invalid_argument(
      "body_id must be BODY_SUN, BODY_EARTH, or BODY_MOON");
}
Eigen::Vector3d DE421::velocity(int id, double jd) const {
  if (id == 1) {
    return Eigen::Vector3d::Zero();
  }
  if (id == 0) {
    return velocity_eci("sun", jd);
  }
  if (id == 2) {
    return velocity_eci("moon", jd);
  }
  throw std::invalid_argument(
      "body_id must be BODY_SUN, BODY_EARTH, or BODY_MOON");
}
std::vector<std::string> DE421::available_bodies() const {
  return {"earth-moon barycenter",
          "emb",
          "jupiter",
          "mars",
          "mercury",
          "moon",
          "neptune",
          "pluto",
          "saturn",
          "sun",
          "uranus",
          "venus"};
}

} // namespace omni::environment
