#include "integrate.hpp"
#include "runtime_python.hpp"
#include <cmath>
#include <pybind11/numpy.h>
namespace py = pybind11;
using namespace omni::propagation;
namespace
{
py::tuple output(const Trajectory &result)
{
    py::array_t<double> times(result.times.size());
    py::array_t<double> states(
        {static_cast<py::ssize_t>(result.states.size()), static_cast<py::ssize_t>(result.states.front().size())});
    auto t = times.mutable_unchecked<1>();
    auto s = states.mutable_unchecked<2>();
    for (std::size_t i = 0; i < result.times.size(); ++i)
    {
        t(i) = result.times[i];
        for (Eigen::Index j = 0; j < result.states[i].size(); ++j)
            s(i, j) = result.states[i][j];
    }
    return py::make_tuple(times, states);
}
struct ProgressBar
{
    py::object bar = py::none();
    ProgressBar(bool enabled, double total, const char *description)
    {
        if (enabled)
            bar = py::module_::import("tqdm").attr("tqdm")(py::arg("total") = total, py::arg("desc") = description,
                                                           py::arg("unit") = "step");
    }
    Progress callback()
    {
        if (bar.is_none())
            return {};
        return [this](std::size_t n) { bar.attr("update")(n); };
    }
    ~ProgressBar()
    {
        if (!bar.is_none())
        {
            try
            {
                bar.attr("close")();
            }
            catch (py::error_already_set &e)
            {
                e.discard_as_unraisable("progress close");
            }
        }
    }
};
} // namespace
void bind_integrate(py::module_ &m)
{
    m.def(
        "integrate_rk4",
        [](Vector y, double jd, double dt, double tf, py::object p, bool progress, std::size_t every) {
            auto r = runtime_from_python(p);
            ProgressBar bar(progress, dt > 0 ? std::ceil(tf / dt) : 0, "Running coupled RK4");
            return output(integrate_rk4(y, jd, dt, tf, r, bar.callback(), every));
        },
        py::arg("y"), py::arg("jd"), py::arg("dt"), py::arg("tf"), py::arg("runtime"), py::arg("progress") = false,
        py::arg("record_every") = 1);
    m.def(
        "integrate_dop853",
        [](Vector y, double jd, double dt, double tf, py::object p, double rtol, Vector atol, double max_step,
           double min_step, bool progress, std::size_t every) {
            auto r = runtime_from_python(p);
            ProgressBar bar(progress, dt > 0 ? std::ceil(tf / dt) : 0, "Running coupled DOP853");
            return output(integrate_dop853(y, jd, dt, tf, r, rtol, atol, max_step, min_step, bar.callback(), every));
        },
        py::arg("y"), py::arg("jd"), py::arg("dt"), py::arg("tf"), py::arg("runtime"), py::arg("rtol"), py::arg("atol"),
        py::arg("max_step"), py::arg("min_step"), py::arg("progress") = false, py::arg("record_every") = 0);
}
