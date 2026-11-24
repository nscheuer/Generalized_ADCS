#include "effects.hpp"
#include <stdexcept>
#include <string>
#include <vector>
#define CHECK(x) do { if (!(x)) throw std::runtime_error("check failed: " #x); } while(false)
int main() {
    using omni::hardware::Effect;
    using omni::hardware::apply_effects;
    CHECK(apply_effects(4.0, std::vector<Effect<double>>{}, 0.5) == 4.0);
    std::vector<double> inputs;
    int random_state = 0;
    std::vector<Effect<double>> effects;
    for (int operation = 0; operation < 3; ++operation) {
        effects.emplace_back([&, operation](double value, double dt) {
            CHECK(dt == 0.25);
            inputs.push_back(value);
            ++random_state;
            if (operation == 0) return value * 2;
            if (operation == 1) return value + 3;
            return value * value;
        });
    }
    CHECK(apply_effects(4.0, effects, 0.25) == 121.0);
    CHECK((inputs == std::vector<double>{4, 8, 11}));
    CHECK(random_state == 3);
    bool later_ran = false, propagated = false;
    std::vector<Effect<double>> failing{
        [](double, double) -> double { throw std::runtime_error("effect failed"); },
        [&](double x, double) { later_ran = true; return x; }
    };
    try { (void)apply_effects(0.0, failing, 1.0); }
    catch (const std::runtime_error& error) { propagated = std::string(error.what()) == "effect failed"; }
    CHECK(propagated);
    CHECK(!later_ran);
}
