#include "solver/coupled_solver.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace
{
struct TestPropagator
{
    double step_size;

    double dt() const
    {
        return step_size;
    }

    int step(int state, int satellite, int environment, int dynamics) const
    {
        return state + satellite + environment + dynamics;
    }
};

void check(bool condition)
{
    if (!condition)
        throw std::runtime_error("coupled solver test assertion failed");
}

template <typename Function> bool throws_invalid_argument(Function &&function)
{
    try
    {
        function();
    }
    catch (const std::invalid_argument &)
    {
        return true;
    }
    return false;
}
} // namespace

int main()
{
    using Solver = omni::solver::CoupledSolver<TestPropagator>;

    const Solver solver(TestPropagator{0.25}, 10.0, true, 20);
    check(solver.propagator.step_size == 0.25);
    check(solver.tf == 10.0);
    check(solver.progress);
    check(solver.progress_chunk_steps == 20);
    check(solver.dt() == 0.25);
    check(solver.step(1, 2, 3, 4) == 10);

    const Solver zero_duration(TestPropagator{1.0}, 0.0);
    check(zero_duration.tf == 0.0);
    check(!zero_duration.progress);
    check(zero_duration.progress_chunk_steps == 500);

    check(throws_invalid_argument([] { Solver(TestPropagator{0.1}, -1.0); }));
    check(throws_invalid_argument([] { Solver(TestPropagator{0.1}, std::numeric_limits<double>::infinity()); }));
    check(throws_invalid_argument([] { Solver(TestPropagator{0.1}, std::numeric_limits<double>::quiet_NaN()); }));
    check(throws_invalid_argument([] { Solver(TestPropagator{0.1}, 1.0, false, 0); }));
    check(throws_invalid_argument([] { Solver(TestPropagator{0.1}, 1.0, false, -1); }));
}
