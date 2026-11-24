#include "solver/solver.hpp"

#include <stdexcept>
#include <string>

using TestSolver = omni::solver::BasicSolver<std::string, int, int, int>;

void check(bool condition) {
    if (!condition) throw std::runtime_error("solver test assertion failed");
}

int main() {
    const TestSolver split("dynamics", 1, 2, std::nullopt);
    check(split.dynamics == "dynamics");
    check(split.orbital == 1);
    check(split.attitude == 2);
    check(!split.coupled.has_value());

    const TestSolver coupled("dynamics", std::nullopt, std::nullopt, 3);
    check(coupled.coupled == 3);
    check(!coupled.orbital.has_value());
    check(!coupled.attitude.has_value());

    const auto rejects = [](std::optional<int> orbital,
                            std::optional<int> attitude,
                            std::optional<int> coupled_value) {
        try {
            TestSolver("dynamics", orbital, attitude, coupled_value);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    check(rejects(std::nullopt, std::nullopt, std::nullopt));
    check(rejects(1, std::nullopt, std::nullopt));
    check(rejects(std::nullopt, 2, std::nullopt));
    check(rejects(1, 2, 3));
    check(rejects(1, std::nullopt, 3));
    check(rejects(std::nullopt, 2, 3));
}
