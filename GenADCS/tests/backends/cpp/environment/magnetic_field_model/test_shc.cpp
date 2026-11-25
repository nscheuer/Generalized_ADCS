#include "check.hpp"
#include "environment/native_helpers.hpp"
#include <filesystem>
#include <fstream>
#include <stdexcept>

using namespace omni::environment;

int main() {
  const auto igrf_kernel =
      environment_data_path("magnetic_field_model/igrf14/IGRF14.shc");
  const auto chaos_core_kernel =
      environment_data_path("magnetic_field_model/chaos8/CHAOS-8.7_core.shc");
  const auto chaos_static_kernel =
      environment_data_path("magnetic_field_model/chaos8/CHAOS-8.7_static.shc");
  CHECK(std::filesystem::exists(igrf_kernel));
  CHECK(std::filesystem::exists(chaos_core_kernel));
  CHECK(std::filesystem::exists(chaos_static_kernel));
  CHECK(igrf_kernel.string().find("/GenADCS/kernels/environment/") !=
        std::string::npos);
  CHECK(chaos_core_kernel.string().find("/GenADCS/kernels/environment/") !=
        std::string::npos);
  const auto bundled = igrf_kernel;
  const auto first = read_shc(bundled, 1, 13);
  const auto second = read_shc(bundled, 1, 13);
  CHECK(first.times == second.times && first.g.size() == second.g.size());
  CHECK((first.g.front() - second.g.front()).norm() == 0.0);

  const auto path =
      std::filesystem::temp_directory_path() / "genadcs_malformed_test.shc";
  {
    std::ofstream malformed(path);
    malformed << "1 1 1 0 0\n2020\n1 0\n";
  }
  bool rejected = false;
  try {
    (void)read_shc(path, 1, 1);
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  std::filesystem::remove(path);
  CHECK(rejected);
  return 0;
}
