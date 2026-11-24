#pragma once
#include "magnetic_field_model/_spherical_harmonics.hpp"
#include "native_models.hpp"
#include <Eigen/Dense>
#include <filesystem>
#include <vector>
namespace omni::environment
{
struct HarmonicData
{
    std::vector<double> times;
    std::vector<Eigen::MatrixXd> g, h;
};
uint32_t read_u32(const std::vector<unsigned char> &raw, std::size_t offset, bool little);
double read_f64(const std::vector<unsigned char> &raw, std::size_t offset, bool little);
HarmonicData read_shc(const std::filesystem::path &path, int nmin, int nmax);
} // namespace omni::environment
