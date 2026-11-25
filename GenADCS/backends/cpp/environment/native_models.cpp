#include "conversions.hpp"
#include "native_helpers.hpp"
#include "time/conversions.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>
namespace omni::environment {
uint32_t read_u32(const std::vector<unsigned char> &raw, std::size_t offset,
                  bool little) {
  if (offset + 4U > raw.size())
    throw std::runtime_error("truncated DAF integer");
  uint32_t value = 0U;
  for (unsigned int i = 0; i < 4U; ++i) {
    const std::size_t j = little ? i : 3U - i;
    value |= static_cast<uint32_t>(raw[offset + j]) << (8U * i);
  }
  return value;
}

double read_f64(const std::vector<unsigned char> &raw, std::size_t offset,
                bool little) {
  if (offset + 8U > raw.size())
    throw std::runtime_error("truncated DAF double");
  uint64_t bits = 0U;
  for (unsigned int i = 0; i < 8U; ++i) {
    const std::size_t j = little ? i : 7U - i;
    bits |= static_cast<uint64_t>(raw[offset + j]) << (8U * i);
  }
  double value = 0.0;
  static_assert(sizeof(value) == sizeof(bits), "unexpected double size");
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

HarmonicData read_shc(const std::filesystem::path &path, int nmin, int nmax) {
  std::ifstream input(path);
  if (!input)
    throw std::runtime_error("cannot open magnetic coefficient file: " +
                             path.string());
  std::string line;
  do {
    if (!std::getline(input, line))
      throw std::runtime_error("empty SHC file");
  } while (line.empty() || line[0] == '#');
  int low = 0, high = 0, count = 0, ignored1 = 0, ignored2 = 0;
  std::istringstream header(line);
  header >> low >> high >> count >> ignored1 >> ignored2;
  if (!header || low != nmin || high != nmax || count <= 0)
    throw std::runtime_error("invalid SHC header");
  do {
    if (!std::getline(input, line))
      throw std::runtime_error("missing SHC epochs");
  } while (line.empty() || line[0] == '#');
  HarmonicData result;
  std::istringstream epochs(line);
  double epoch = 0.0;
  while (epochs >> epoch)
    result.times.push_back(epoch);
  if (result.times.size() != static_cast<std::size_t>(count))
    throw std::runtime_error("invalid SHC epoch count");
  result.g.assign(static_cast<std::size_t>(count),
                  Eigen::MatrixXd::Zero(nmax + 1, nmax + 1));
  result.h.assign(static_cast<std::size_t>(count),
                  Eigen::MatrixXd::Zero(nmax + 1, nmax + 1));
  std::vector<std::vector<bool>> seen(
      static_cast<std::size_t>(nmax + 1),
      std::vector<bool>(static_cast<std::size_t>(2 * nmax + 1), false));
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    std::istringstream row(line);
    int n = 0, m = 0;
    row >> n >> m;
    if (!row || n < nmin || n > nmax || std::abs(m) > n)
      throw std::runtime_error("invalid SHC coefficient row");
    const std::size_t si = static_cast<std::size_t>(m + nmax);
    if (seen[static_cast<std::size_t>(n)][si])
      throw std::runtime_error("duplicate SHC coefficient");
    seen[static_cast<std::size_t>(n)][si] = true;
    for (int t = 0; t < count; ++t) {
      double value = 0.0;
      row >> value;
      if (!row)
        throw std::runtime_error("malformed SHC values");
      (m >= 0 ? result.g[static_cast<std::size_t>(t)]
              : result.h[static_cast<std::size_t>(t)])(n, std::abs(m)) = value;
    }
  }
  Eigen::MatrixXd scale = Eigen::MatrixXd::Zero(nmax + 1, nmax + 1);
  scale(0, 0) = 1.0;
  for (int n = 1; n <= nmax; ++n) {
    scale(n, 0) = scale(n - 1, 0) * (2.0 * n - 1.0) / n;
    for (int m = 1; m <= n; ++m)
      scale(n, m) = scale(n, m - 1) *
                    std::sqrt((n - m + 1.0) * (m == 1 ? 2.0 : 1.0) / (n + m));
  }
  for (std::size_t i = 0; i < result.g.size(); ++i) {
    result.g[i].array() *= scale.array();
    result.h[i].array() *= scale.array();
  }
  return result;
}
std::filesystem::path
environment_data_path(const std::filesystem::path &relative) {
  const std::filesystem::path root = std::filesystem::path(__FILE__)
                                         .parent_path()
                                         .parent_path()
                                         .parent_path()
                                         .parent_path() /
                                     "kernels/environment";
  return root / relative;
}
Eigen::MatrixXd schmidt_table(int maximum_degree) {
  if (maximum_degree < 0)
    throw std::invalid_argument("maximum degree must be nonnegative");
  Eigen::MatrixXd scale =
      Eigen::MatrixXd::Zero(maximum_degree + 1, maximum_degree + 1);
  scale(0, 0) = 1.0;
  for (int n = 1; n <= maximum_degree; ++n) {
    scale(n, 0) = scale(n - 1, 0) * (2.0 * n - 1.0) / n;
    for (int m = 1; m <= n; ++m)
      scale(n, m) = scale(n, m - 1) *
                    std::sqrt((n - m + 1.0) * (m == 1 ? 2.0 : 1.0) / (n + m));
  }
  return scale;
}
Eigen::Vector3d
magnetic_field_with_coefficients(const Eigen::Vector3d &position, double jd,
                                 const Eigen::MatrixXd &g,
                                 const Eigen::MatrixXd &h, int nmax) {
  return eval_field(position, jd, g, h, nmax);
}
std::tuple<std::vector<double>, std::vector<Eigen::MatrixXd>,
           std::vector<Eigen::MatrixXd>>
load_shc(const std::filesystem::path &path, int nmin, int nmax) {
  auto value = read_shc(path, nmin, nmax);
  return {std::move(value.times), std::move(value.g), std::move(value.h)};
}

} // namespace omni::environment
