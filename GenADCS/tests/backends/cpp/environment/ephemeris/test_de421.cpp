#include "check.hpp"
#include "environment/native_helpers.hpp"
#include "environment/native_models.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>

using namespace omni::environment;

int main() {
    const DE421 model;
    const DE421 second;
    const auto expected_kernel = environment_data_path("ephemeris/de421/de421.bsp");
    CHECK(model.filepath() == expected_kernel);
    CHECK(std::filesystem::exists(expected_kernel));
    const double epoch = 2458849.5;
    const Eigen::Vector3d position = model.position_eci("sun", epoch);
    const Eigen::Vector3d velocity = model.velocity_eci("sun", epoch);
    const auto state = model.state_eci("sun", epoch);
    CHECK(position.allFinite() && velocity.allFinite());
    CHECK(position.norm() > 1.4e8 && position.norm() < 1.6e8);
    CHECK(velocity.norm() > 25.0 && velocity.norm() < 35.0);
    CHECK((position - state.first).norm() < 1e-10);
    CHECK((velocity - state.second).norm() < 1e-12);
    CHECK((position - second.position_eci("sun", epoch)).norm() < 1e-12);
    const auto moon_positions = model.position_eci("moon", epoch);
    CHECK(moon_positions.allFinite() && moon_positions.norm() > 3.0e5 && moon_positions.norm() < 5.0e5);

    bool invalid_body = false;
    try { (void)model.position_eci("not-a-body", epoch); }
    catch (const std::invalid_argument&) { invalid_body = true; }
    CHECK(invalid_body);
    bool invalid_id = false;
    try { (void)model.position(99, epoch); }
    catch (const std::invalid_argument&) { invalid_id = true; }
    CHECK(invalid_id);
    bool out_of_range = false;
    try { (void)model.position_eci("sun", 1'000'000.0); }
    catch (const std::out_of_range&) { out_of_range = true; }
    CHECK(out_of_range);
    const auto bad_path = std::filesystem::temp_directory_path() / "genadcs_bad_de421.bsp";
    { std::ofstream bad_kernel(bad_path, std::ios::binary); bad_kernel << "not an SPK"; }
    bool malformed_kernel = false;
    try { (void)DE421(bad_path); }
    catch (const std::runtime_error&) { malformed_kernel = true; }
    std::filesystem::remove(bad_path);
    CHECK(malformed_kernel);
    return 0;
}
