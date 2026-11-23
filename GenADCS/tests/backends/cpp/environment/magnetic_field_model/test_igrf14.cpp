#include "check.hpp"
#include "environment/native_models.hpp"
#include <stdexcept>

using namespace omni::environment;

int main() {
    const IGRF14 model;
    const Eigen::Vector3d p(6'778'137.0, 0.0, 0.0);
    const Eigen::Vector3d field = model.field(p, 2458849.5);
    CHECK(field.allFinite() && field.norm() > 1e-5 && field.norm() < 7e-5);
    CHECK((field - model.field(p, 2458849.5)).norm() == 0.0);
    bool old_epoch = false;
    try { (void)model.field(p, 2414650.5); }
    catch (const std::out_of_range&) { old_epoch = true; }
    CHECK(old_epoch);
    bool zero_position = false;
    try { (void)model.field(Eigen::Vector3d::Zero(), 2458849.5); }
    catch (const std::invalid_argument&) { zero_position = true; }
    CHECK(zero_position);
    return 0;
}
