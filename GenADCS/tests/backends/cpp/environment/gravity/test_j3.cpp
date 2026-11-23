#include "check.hpp"
#include "environment/gravity/j_gravity_model.hpp"

using namespace omni::environment;

int main() {
    const J3 model;
    const Eigen::Vector3d north = model.evaluate(Eigen::Vector3d(7e6, 0.0, 1e6));
    const Eigen::Vector3d south = model.evaluate(Eigen::Vector3d(7e6, 0.0, -1e6));
    CHECK(north.allFinite() && south.allFinite());
    CHECK((north - south).norm() > 1e-8);
    return 0;
}
