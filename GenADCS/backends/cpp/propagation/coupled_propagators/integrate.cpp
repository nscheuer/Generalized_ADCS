#include "integrate.hpp"
#include "coupled_dop853_propagator.hpp"
#include "coupled_rk4_propagator.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>
namespace omni::propagation {
static Trajectory output(const std::vector<double> &times,
                         const std::vector<Vector> &states) {
  return {times, states};
}
Trajectory integrate_rk4(Vector y, double jd, double dt, double tf,
                         const Runtime &p, const Progress &progress,
                         std::size_t record_every, const Observer &observe) {
  if (!std::isfinite(dt) || dt <= 0) {
    throw std::invalid_argument("dt must be finite and positive");
  }
  if (record_every == 0) {
    throw std::invalid_argument("record_every must be positive");
  }
  if (!std::isfinite(tf) || tf < 0) {
    throw std::invalid_argument("tf must be finite and nonnegative seconds");
  }
  project(y, p);
  std::vector<double> times{0.};
  std::vector<Vector> states{y};
  if (observe) observe(0.0, y, true);
  const auto full = static_cast<std::size_t>(std::floor(tf / dt));
  double remainder = tf - full * dt;
  if (remainder <= 1e-12 * std::max(1., tf)) {
    remainder = 0;
  }
  for (std::size_t i = 0; i < full + (remainder > 0); ++i) {
    double elapsed = i * dt, h = i == full ? remainder : dt;
    y = rk4_step(y, jd + elapsed / 86400., h, p);
    const bool recorded = (i + 1) % record_every == 0 || i + 1 == full + (remainder > 0);
    if (observe) observe(std::min((i + 1) * dt, tf), y, recorded);
    if ((i + 1) % record_every == 0 || i + 1 == full + (remainder > 0)) {
      times.push_back(std::min((i + 1) * dt, tf));
      states.push_back(y);
    }
    if (progress) {
      progress(1);
    }
  }
  times.back() = tf;
  return output(times, states);
}
Trajectory integrate_dop853(Vector y, double jd, double dt, double tf,
                            const Runtime &p, double rtol, Vector atol,
                            double max_step, double min_step,
                            const Progress &progress,
                            std::size_t record_every, const Observer &observe) {
  if (!std::isfinite(dt) || dt <= 0) {
    throw std::invalid_argument("dt must be finite and positive");
  }
  if (!std::isfinite(tf) || tf < 0) {
    throw std::invalid_argument("tf must be finite and nonnegative seconds");
  }
  project(y, p);
  const auto main_total = static_cast<std::size_t>(std::ceil(tf / dt));
  if (atol.size() == 1) {
    atol = Vector::Constant(y.size(), atol[0]);
  }
  if (atol.size() != y.size()) {
    throw std::invalid_argument("atol length differs from state size");
  }
  std::vector<double> times{0.};
  std::vector<Vector> states{y};
  if (observe) observe(0.0, y, true);
  if (tf == 0) {
    return output(times, states);
  }
  std::size_t completed_main = 0;
  double t = 0, h = std::min(dt, max_step);
  const double interval = record_every ? dt * record_every : 0.;
  double next_sample = interval;
  Vector f = derivative(y, jd, p);
  std::size_t trials = 0;
  while (t < tf) {
    if (++trials > 10000000) {
      throw std::invalid_argument("DOP853 exceeded maximum trials");
    }
    double remaining = tf - t, step = std::min({h, max_step, remaining});
    if (interval > 0 && next_sample < tf) {
      step = std::min(step, next_sample - t);
    }
    if (step <= 0 || t + step == t) {
      throw std::invalid_argument(
          "DOP853 step is too small to advance simulation time");
    }
    const bool boundary_limited =
        interval > 0 && next_sample < tf && next_sample - t <= min_step;
    if (step < min_step && remaining > min_step && !boundary_limited) {
      throw std::invalid_argument(
          "DOP853 reached min_step before satisfying tolerances");
    }
    auto [candidate, next_f, error] =
        dop853_attempt(y, f, jd + t / 86400., step, p, rtol, atol);
    if (!std::isfinite(error)) {
      throw std::invalid_argument("DOP853 produced a nonfinite error estimate");
    }
    if (error <= 1) {
      t += step;
      y = candidate;
      f = next_f;
      const bool recorded = interval == 0 ||
          std::abs(t - next_sample) <= 1e-10 * std::max(1., next_sample);
      if (observe) observe(t, y, recorded);
      if (interval == 0 ||
          std::abs(t - next_sample) <= 1e-10 * std::max(1., next_sample)) {
        times.push_back(t);
        states.push_back(y);
        next_sample += interval;
      }
      if (progress) {
        const auto main_done = std::min(
            main_total, static_cast<std::size_t>(std::floor(t / dt + 1e-12)));
        if (main_done > completed_main) {
          progress(main_done - completed_main);
          completed_main = main_done;
        }
      }
      double factor =
          error == 0 ? 10.
                     : std::min(10., std::max(.2, .9 * std::pow(error, -.125)));
      h = std::min(max_step, step * factor);
    } else {
      h = step * std::max(.2, .9 * std::pow(error, -.125));
      if (h < min_step && !boundary_limited) {
        throw std::invalid_argument(
            "DOP853 reached min_step before satisfying tolerances");
      }
    }
  }
  if (progress) {
    if (completed_main < main_total) {
      progress(main_total - completed_main);
    }
  }
  if (interval > 0 && std::abs(times.back() - tf) > 1e-10 * std::max(1., tf)) {
    if (observe) observe(tf, y, true);
    times.push_back(tf);
    states.push_back(y);
  }
  return output(times, states);
}
} // namespace omni::propagation
