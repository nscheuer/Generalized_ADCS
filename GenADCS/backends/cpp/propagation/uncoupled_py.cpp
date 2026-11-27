#include "uncoupled.hpp"
#include "attitude_propagators/simulation_sampling.hpp"
#include "control_loop.hpp"
#include "coupled_propagators/runtime_python.hpp"
#include "../hardware/ADCS_subsystem/controllers/attitude_controller.hpp"
#include "../hardware/ADCS_subsystem/controllers/bcross.hpp"
#include "../hardware/ADCS_subsystem/controllers/bdot.hpp"
#include "../hardware/ADCS_subsystem/controllers/desaturation/hoganschaubdesaturation.hpp"
#include "../hardware/ADCS_subsystem/controllers/feedback/mrpfeedbacklinearization.hpp"
#include "../hardware/ADCS_subsystem/controllers/feedforward/passthrough.hpp"
#include "../hardware/ADCS_subsystem/controllers/allocator/l2allocator.hpp"
#include "../hardware/ADCS_subsystem/controllers/allocator/linftyallocator.hpp"
#include "../hardware/ADCS_subsystem/goals/along_track_goal.hpp"
#include "../hardware/ADCS_subsystem/goals/angular_rate_goal.hpp"
#include "../hardware/ADCS_subsystem/goals/coordinate_fmc_goal.hpp"
#include "../hardware/ADCS_subsystem/goals/coordinate_goal.hpp"
#include "../hardware/ADCS_subsystem/goals/cross_track_goal.hpp"
#include "../hardware/ADCS_subsystem/goals/nadir_goal.hpp"
#include "../hardware/ADCS_subsystem/goals/no_goal.hpp"
#include "../hardware/ADCS_subsystem/goals/quaternion_goal.hpp"
#include "../hardware/ADCS_subsystem/goals/sun_goal.hpp"
#include "../hardware/ADCS_subsystem/goals/velocity_goal.hpp"
#include "../hardware/ADCS_subsystem/goals/zenith_goal.hpp"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>

namespace py = pybind11;

namespace {
namespace hardware = omni::hardware;
hardware::Sensor as_sensor(py::handle value) {
  namespace sensors = hardware::sensors;
  // Check the derived sensor before its base to preserve its concrete type.
  if (py::isinstance<sensors::FineEarthSensor>(value)) {
    return hardware::Sensor(std::in_place_type<sensors::FineEarthSensor>,
                            value.cast<const sensors::FineEarthSensor &>());
  }
  if (py::isinstance<sensors::FineSunSensor>(value)) {
    return hardware::Sensor(std::in_place_type<sensors::FineSunSensor>,
                            value.cast<const sensors::FineSunSensor &>());
  }
  if (py::isinstance<sensors::Gyroscope>(value)) {
    return hardware::Sensor(std::in_place_type<sensors::Gyroscope>,
                            value.cast<const sensors::Gyroscope &>());
  }
  if (py::isinstance<sensors::Magnetometer>(value)) {
    return hardware::Sensor(std::in_place_type<sensors::Magnetometer>,
                            value.cast<const sensors::Magnetometer &>());
  }
  if (py::isinstance<sensors::Accelerometer>(value)) {
    return hardware::Sensor(std::in_place_type<sensors::Accelerometer>,
                            value.cast<const sensors::Accelerometer &>());
  }
  if (py::isinstance<sensors::CoarseSunSensor>(value)) {
    return hardware::Sensor(std::in_place_type<sensors::CoarseSunSensor>,
                            value.cast<const sensors::CoarseSunSensor &>());
  }
  if (py::isinstance<sensors::GNSS>(value)) {
    return hardware::Sensor(std::in_place_type<sensors::GNSS>,
                            value.cast<const sensors::GNSS &>());
  }
  if (py::isinstance<sensors::AutonomousStarTracker>(value)) {
    return hardware::Sensor(
        std::in_place_type<sensors::AutonomousStarTracker>,
        value.cast<const sensors::AutonomousStarTracker &>());
  }
  throw py::type_error("sensors must contain native C++ sensor objects");
}

std::shared_ptr<hardware::goals::Goal> extract_goal(const py::handle &obj) {
  namespace g = hardware::goals;
  if (obj.is_none())
    return std::make_shared<g::NoGoal>();
  const std::string tname =
      py::str(py::type::handle_of(obj).attr("__name__")).cast<std::string>();
  if (tname == "NoGoal")
    return std::make_shared<g::NoGoal>();
  if (tname == "NadirGoal")
    return std::make_shared<g::NadirGoal>();
  if (tname == "ZenithGoal")
    return std::make_shared<g::ZenithGoal>();
  if (tname == "AlongTrackGoal")
    return std::make_shared<g::AlongTrackGoal>();
  if (tname == "CrossTrackGoal")
    return std::make_shared<g::CrossTrackGoal>();
  if (tname == "VelocityGoal")
    return std::make_shared<g::VelocityGoal>();
  if (tname == "SunGoal")
    return std::make_shared<g::SunGoal>();
  if (tname == "QuaternionGoal") {
    return std::make_shared<g::QuaternionGoal>(
        obj.attr("quaternion").cast<Eigen::Vector4d>());
  }
  if (tname == "AngularRateGoal") {
    return std::make_shared<g::AngularRateGoal>(
        obj.attr("rate").cast<Eigen::Vector3d>());
  }
  if (tname == "CoordinateGoal") {
    return std::make_shared<g::CoordinateGoal>(
        obj.attr("lat_deg").cast<double>(), obj.attr("lon_deg").cast<double>(),
        obj.attr("alt_km").cast<double>());
  }
  if (tname == "CoordinateFMCGoal") {
    return std::make_shared<g::CoordinateFMCGoal>(
        obj.attr("lat_deg").cast<double>(), obj.attr("lon_deg").cast<double>(),
        obj.attr("alt_km").cast<double>(), obj.attr("fmc_level").cast<int>());
  }
  throw py::type_error("unsupported Goal: " + tname);
}

std::shared_ptr<hardware::controllers::Controller>
extract_controller(const py::handle &obj) {
  namespace c = hardware::controllers;
  if (obj.is_none())
    return std::make_shared<c::NoController>();
  const std::string tname =
      py::str(py::type::handle_of(obj).attr("__name__")).cast<std::string>();
  if (tname == "NoController")
    return std::make_shared<c::NoController>();
  if (tname == "BDot")
    return std::make_shared<c::BDot>(obj.attr("gain").cast<double>());
  if (tname == "BCross")
    return std::make_shared<c::BCross>(obj.attr("gain").cast<double>());
  if (tname == "AttitudeController") {
    const py::object fb_obj = obj.attr("feedback");
    const py::object ff_obj = obj.attr("feedforward");
    const py::object alloc_obj = obj.attr("allocator");
    const py::object desat_obj = obj.attr("desaturation");
    auto fb = std::make_shared<c::MRPFeedbackLinearized>(
        fb_obj.attr("kp").cast<Eigen::Matrix3d>(),
        fb_obj.attr("kd").cast<Eigen::Matrix3d>());
    auto ff = std::make_shared<c::Passthrough>();
    (void)ff_obj;
    std::shared_ptr<c::Allocator> alloc;
    if (py::str(py::type::handle_of(alloc_obj).attr("__name__"))
            .cast<std::string>() == "LInfinityAllocator") {
      alloc = std::make_shared<c::LInfinityAllocator>();
    } else {
      alloc = std::make_shared<c::L2Allocator>();
    }
    std::shared_ptr<c::Desaturation> desat;
    if (py::str(py::type::handle_of(desat_obj).attr("__name__"))
            .cast<std::string>() == "HoganSchaubManagement") {
      desat = std::make_shared<c::HoganSchaubManagement>(
          desat_obj.attr("gain").cast<double>(),
          desat_obj.attr("target_momentum").cast<double>(),
          desat_obj.attr("mtq_gain").cast<double>());
    } else {
      desat = std::make_shared<c::NoDesaturation>();
    }
    return std::make_shared<c::AttitudeController>(
        std::move(fb), std::move(ff), std::move(alloc), std::move(desat));
  }
  throw py::type_error("unsupported Controller: " + tname);
}

hardware::DeviceMaskSpec extract_device_mask(const py::handle &obj) {
  hardware::DeviceMaskSpec spec;
  if (obj.is_none())
    return spec;
  for (auto item : obj.cast<py::dict>()) {
    spec[py::str(item.first).cast<std::string>()] = item.second.cast<bool>();
  }
  return spec;
}

std::vector<hardware::obc::TimelineEntry>
extract_obc_timeline(const py::handle &obj) {
  std::vector<hardware::obc::TimelineEntry> out;
  if (obj.is_none())
    return out;
  for (py::handle item : obj) {
    py::tuple tup = item.cast<py::tuple>();
    const double t = tup[0].cast<double>();
    const std::string cmd = tup[1].cast<std::string>();
    std::shared_ptr<hardware::goals::Goal> g = nullptr;
    if (tup.size() == 3 && !tup[2].is_none()) {
      g = extract_goal(tup[2]);
    }
    out.emplace_back(t, cmd, std::move(g));
  }
  return out;
}

hardware::obc::Mode extract_obc_mode(const py::handle &obj) {
  const std::string name = obj.attr("name").cast<std::string>();
  std::vector<hardware::obc::Mode> submodes;
  for (py::handle sub : obj.attr("submodes")) {
    submodes.push_back(extract_obc_mode(sub));
  }
  std::optional<std::string> initial_submode;
  if (!obj.attr("initial_submode").is_none())
    initial_submode = obj.attr("initial_submode").cast<std::string>();
  std::optional<double> duration;
  if (!obj.attr("duration").is_none())
    duration = obj.attr("duration").cast<double>();
  std::optional<std::string> auto_target;
  if (!obj.attr("auto_target").is_none())
    auto_target = obj.attr("auto_target").cast<std::string>();
  hardware::obc::ModeBindings bindings;
  const py::object b_obj = obj.attr("bindings");
  if (!b_obj.is_none()) {
    py::dict d = b_obj.cast<py::dict>();
    if (d.contains("adcs") && !d["adcs"].is_none()) {
      py::object val = d["adcs"];
      if (py::isinstance<py::str>(val)) {
        bindings.adcs = val.cast<std::string>();
      } else if (py::isinstance<py::tuple>(val)) {
        py::tuple tup = val.cast<py::tuple>();
        bindings.adcs = std::make_pair(tup[0].cast<std::string>(),
                                       extract_goal(tup[1]));
      } else {
        bindings.adcs = extract_goal(val);
      }
    }
    if (d.contains("sensors") && !d["sensors"].is_none())
      bindings.sensors = extract_device_mask(d["sensors"]);
    if (d.contains("actuators") && !d["actuators"].is_none())
      bindings.actuators = extract_device_mask(d["actuators"]);
  }
  return hardware::obc::Mode(name, std::move(submodes), initial_submode,
                             duration, auto_target, std::move(bindings));
}

omni::propagation::ControlLoopContext build_control_context(
    const py::object &adcs, const py::object &obc_obj, double mu,
    const omni::propagation::Runtime &runtime,
    const std::vector<hardware::Sensor> &sensors,
    std::unique_ptr<omni::environment::DE421> &own_eph) {
  omni::propagation::ControlLoopContext ctx;
  const py::object adcs_sm_obj = adcs.attr("state_machine");
  const py::object adcs_ctrl_obj = adcs.attr("controller");
  const bool has_obc = !obc_obj.is_none() && !obc_obj.attr("state_machine").is_none();
  const bool has_sm = !adcs_sm_obj.is_none();
  const bool has_ctrl = !adcs_ctrl_obj.is_none();
  if (!has_obc && !has_sm && !has_ctrl) {
    return ctx;
  }
  ctx.active = true;

  std::vector<std::string> sensor_names;
  ctx.sensor_enabled.reserve(sensors.size());
  for (std::size_t i = 0; i < sensors.size(); ++i) {
    int code = -1;
    Eigen::Vector4d ori(1, 0, 0, 0);
    bool en = true;
    std::string tname;
    std::visit(
        [&](const auto &dev) {
          using T = std::decay_t<decltype(dev)>;
          en = dev.enabled;
          if constexpr (std::is_same_v<T, hardware::sensors::Gyroscope>) {
            code = 0;
            tname = "Gyroscope";
          } else if constexpr (std::is_same_v<T, hardware::sensors::Magnetometer>) {
            code = 1;
            tname = "Magnetometer";
          } else if constexpr (std::is_same_v<T, hardware::sensors::AutonomousStarTracker>) {
            tname = "AutonomousStarTracker";
          } else if constexpr (std::is_same_v<T, hardware::sensors::CoarseSunSensor>) {
            tname = "CoarseSunSensor";
          } else if constexpr (std::is_same_v<T, hardware::sensors::FineSunSensor>) {
            tname = "FineSunSensor";
          } else if constexpr (std::is_same_v<T, hardware::sensors::FineEarthSensor>) {
            tname = "FineEarthSensor";
          } else if constexpr (std::is_same_v<T, hardware::sensors::GNSS>) {
            tname = "GNSS";
          } else if constexpr (std::is_same_v<T, hardware::sensors::Accelerometer>) {
            tname = "Accelerometer";
          }
          if constexpr (!std::is_same_v<T, hardware::sensors::GNSS> &&
                        !std::is_same_v<T, hardware::sensors::Accelerometer>) {
            for (int k = 0; k < 4; ++k)
              ori[k] = dev.orientation[k];
          }
        },
        sensors[i]);
    ctx.sensor_enabled.push_back(en);
    sensor_names.push_back(tname);
    if (code == 0 || code == 1) {
      // offset will be matched below
    }
  }

  // Build ctrl_sensor_config matching SimulationSampler reading_layout_ offsets.
  int total_off = 0;
  for (const auto &s : sensors) {
    std::visit(
        [&](const auto &dev) {
          using T = std::decay_t<decltype(dev)>;
          int code = -1;
          int width = 0;
          if constexpr (std::is_same_v<T, hardware::sensors::Gyroscope>) {
            code = 0;
            width = 3;
          } else if constexpr (std::is_same_v<T, hardware::sensors::Magnetometer>) {
            code = 1;
            width = 3;
          } else if constexpr (std::is_same_v<T, hardware::sensors::AutonomousStarTracker>) {
            width = 4;
          } else if constexpr (std::is_same_v<T, hardware::sensors::CoarseSunSensor>) {
            width = 1;
          } else if constexpr (std::is_same_v<T, hardware::sensors::FineSunSensor>) {
            width = 2;
          } else if constexpr (std::is_same_v<T, hardware::sensors::FineEarthSensor>) {
            width = 2;
          }
          if (width > 0) {
            if (code == 0 || code == 1) {
              hardware::controllers::SensorEntry entry;
              entry.code = code;
              entry.enabled = dev.enabled;
              entry.offset = total_off;
              if constexpr (!std::is_same_v<T, hardware::sensors::GNSS> &&
                            !std::is_same_v<T, hardware::sensors::Accelerometer>) {
                for (int k = 0; k < 4; ++k)
                  entry.orientation[k] = dev.orientation[k];
              }
              ctx.ctrl_sensor_config.push_back(entry);
            }
            total_off += width;
          }
        },
        s);
  }

  std::vector<std::string> rw_names;
  std::vector<double> rw_max_t;
  for (py::handle rw : adcs.attr("reaction_wheels")) {
    std::string nm = "ReactionWheel";
    if (py::hasattr(rw, "name") && !rw.attr("name").is_none()) {
      nm = rw.attr("name").cast<std::string>();
    }
    rw_names.push_back(nm);
    rw_max_t.push_back(rw.attr("max_torque").cast<double>());
    ctx.rw_enabled.push_back(true);
  }
  ctx.base_wheel_max = Eigen::Map<const Eigen::VectorXd>(
      rw_max_t.data(), static_cast<Eigen::Index>(rw_max_t.size()));

  std::vector<std::string> mtq_names;
  for (py::handle mtq : adcs.attr("magnetorquers")) {
    std::string nm = "Magnetorquer";
    if (py::hasattr(mtq, "name") && !mtq.attr("name").is_none()) {
      nm = mtq.attr("name").cast<std::string>();
    }
    mtq_names.push_back(nm);
    ctx.mtq_enabled.push_back(true);
  }
  ctx.base_mtq_max = runtime.mtq_max_dipole;

  hardware::GoalTable goal_table;
  hardware::controllers::ControllerTable ctrl_table;

  std::shared_ptr<hardware::ADCSStateMachine> native_adcs_sm;
  if (has_sm) {
    std::vector<hardware::ADCSMode> modes;
    for (py::handle m : adcs_sm_obj.attr("modes")) {
      modes.emplace_back(
          m.attr("name").cast<std::string>(), extract_goal(m.attr("goal")),
          extract_controller(m.attr("controller")),
          extract_device_mask(m.attr("sensors")),
          extract_device_mask(m.attr("actuators")));
    }
    native_adcs_sm = std::make_shared<hardware::ADCSStateMachine>(
        adcs_sm_obj.attr("initial").cast<std::string>(), std::move(modes));
  }
  ctx.adcs_sm_rt = hardware::prepare_adcs_sm(
      native_adcs_sm.get(), &goal_table, &ctrl_table, sensor_names, rw_names,
      mtq_names);
  ctx.adcs_work = hardware::init_adcs_work(ctx.adcs_sm_rt);
  ctx.adcs_sm_every = (has_sm || has_obc) ? adcs.attr("N").cast<int>() : 0;

  if (has_obc) {
    const py::object obc_sm_obj = obc_obj.attr("state_machine");
    std::vector<hardware::obc::Mode> obc_modes;
    for (py::handle m : obc_sm_obj.attr("modes")) {
      obc_modes.push_back(extract_obc_mode(m));
    }
    std::vector<hardware::obc::Transition> obc_trans;
    for (py::handle tr : obc_sm_obj.attr("transitions")) {
      const py::object trig_obj = tr.attr("trigger");
      const std::string tname =
          py::str(py::type::handle_of(trig_obj).attr("__name__"))
              .cast<std::string>();
      std::shared_ptr<hardware::obc::Trigger> trig;
      if (tname == "Telecommand") {
        trig = std::make_shared<hardware::obc::Telecommand>(
            trig_obj.attr("command").cast<std::string>(),
            trig_obj.attr("goal").is_none()
                ? nullptr
                : extract_goal(trig_obj.attr("goal")),
            trig_obj.attr("sensors").is_none()
                ? std::nullopt
                : std::make_optional(
                      extract_device_mask(trig_obj.attr("sensors"))),
            trig_obj.attr("actuators").is_none()
                ? std::nullopt
                : std::make_optional(
                      extract_device_mask(trig_obj.attr("actuators"))));
      } else if (tname == "InFOV") {
        trig = std::make_shared<hardware::obc::InFOV>(
            trig_obj.attr("lat").cast<double>(),
            trig_obj.attr("lon").cast<double>(),
            trig_obj.attr("alt").cast<double>(),
            trig_obj.attr("max_off_nadir_deg").cast<double>(),
            trig_obj.attr("negate").cast<bool>(),
            trig_obj.attr("goal").is_none()
                ? nullptr
                : extract_goal(trig_obj.attr("goal")));
      } else if (tname == "Completion") {
        trig = std::make_shared<hardware::obc::Completion>();
      } else if (tname == "After") {
        trig = std::make_shared<hardware::obc::After>(
            trig_obj.attr("duration").cast<double>());
      } else if (tname == "FDIRTriggered") {
        std::optional<std::string> mon;
        if (!trig_obj.attr("monitor").is_none())
          mon = trig_obj.attr("monitor").cast<std::string>();
        trig = std::make_shared<hardware::obc::FDIRTriggered>(mon);
      } else {
        throw py::type_error("unsupported Trigger: " + tname);
      }
      obc_trans.emplace_back(tr.attr("source").cast<std::string>(),
                             tr.attr("target").cast<std::string>(),
                             std::move(trig), tr.attr("priority").cast<int>());
    }
    auto native_obc_sm = std::make_shared<hardware::obc::StateMachine>(
        obc_sm_obj.attr("initial").cast<std::string>(), std::move(obc_modes),
        std::move(obc_trans),
        extract_obc_timeline(obc_sm_obj.attr("timeline")));

    std::shared_ptr<hardware::obc::FDIR> native_fdir;
    const py::object fdir_obj = obc_obj.attr("fdir");
    if (!fdir_obj.is_none()) {
      std::vector<std::shared_ptr<hardware::obc::FDIRMonitor>> mons;
      for (py::handle mon : fdir_obj.attr("monitors")) {
        std::optional<std::string> tm;
        if (!mon.attr("target_mode").is_none())
          tm = mon.attr("target_mode").cast<std::string>();
        mons.push_back(std::make_shared<hardware::obc::AngularRateMonitor>(
            mon.attr("max_rate").cast<double>(),
            mon.attr("name").cast<std::string>(), tm));
      }
      native_fdir = std::make_shared<hardware::obc::FDIR>(
          std::move(mons), fdir_obj.attr("safe_mode").cast<std::string>());
    }
    auto native_obc = std::make_shared<hardware::obc::OBCSubsystem>(
        std::move(native_obc_sm), std::move(native_fdir),
        obc_obj.attr("N").cast<int>(),
        extract_obc_timeline(obc_obj.attr("timeline")));

    std::map<std::string, int> adcs_mode_to_id;
    for (std::size_t i = 0; i < ctx.adcs_sm_rt.mode_names.size(); ++i) {
      adcs_mode_to_id[ctx.adcs_sm_rt.mode_names[i]] = static_cast<int>(i);
    }
    // Map OBC goals through ADCS GoalTable so ADCSSMRuntime can evaluate them!
    hardware::obc::GoalTable obc_gt;
    ctx.obc_rt = hardware::obc::prepare_obc(
        native_obc.get(), adcs_mode_to_id, &obc_gt, sensor_names, rw_names,
        mtq_names);
    if (obc_gt.objects.size() > 1) {
      std::vector<int> remap(obc_gt.objects.size(), 0);
      for (std::size_t i = 1; i < obc_gt.objects.size(); ++i) {
        remap[i] = goal_table.register_goal(obc_gt.objects[i]);
      }
      for (auto &g : ctx.obc_rt.state_adcs_goal)
        if (g >= 0)
          g = remap[static_cast<std::size_t>(g)];
      for (auto &g : ctx.obc_rt.tc_trans_goal)
        if (g >= 0)
          g = remap[static_cast<std::size_t>(g)];
      for (auto &g : ctx.obc_rt.fov_goal)
        if (g >= 0)
          g = remap[static_cast<std::size_t>(g)];
      for (auto &g : ctx.obc_rt.tc_goals)
        if (g >= 0)
          g = remap[static_cast<std::size_t>(g)];
    }
    ctx.adcs_sm_rt.goal_kinds = goal_table.kinds;
    ctx.adcs_sm_rt.goal_params = goal_table.params;
    ctx.adcs_sm_rt.goal_objects = goal_table.objects;
    ctx.adcs_sm_rt.goal_names = goal_table.names;

    ctx.obc_work = hardware::obc::init_obc_work(ctx.obc_rt);
    ctx.obc_every = native_obc->N;
  }

  auto default_ctrl = extract_controller(adcs_ctrl_obj);
  ctx.default_ctrl_idx = ctrl_table.register_controller(default_ctrl);
  ctx.ctrl_rt = hardware::controllers::prepare_controllers(
      {}, sensor_names, runtime.mtq_axes, runtime.mtq_max_dipole, &ctrl_table,
      &runtime.wheel_axes, &ctx.base_wheel_max, &runtime.inertia,
      &runtime.inertia);
  // Compute bus_inertia = inertia - sum(J_w * a_w a_w^T) for AttitudeController:
  Eigen::Matrix3d bus_inertia = runtime.inertia;
  for (Eigen::Index i = 0; i < runtime.wheel_inertias.size(); ++i) {
    const Eigen::Vector3d ax = runtime.wheel_axes.row(i).transpose();
    bus_inertia -= runtime.wheel_inertias[i] * (ax * ax.transpose());
  }
  ctx.ctrl_rt.bus_inertia = bus_inertia;
  ctx.ctrl_work = hardware::controllers::init_controller_work();

  ctx.goal_runtime.mu = mu;
  if (runtime.ephemeris) {
    ctx.goal_runtime.ephemeris = runtime.ephemeris.get();
  } else {
    for (int k : ctx.adcs_sm_rt.goal_kinds) {
      if (k == hardware::GOAL_SUN) {
        own_eph = std::make_unique<omni::environment::DE421>();
        ctx.goal_runtime.ephemeris = own_eph.get();
        break;
      }
    }
  }
  return ctx;
}

} // namespace

void bind_uncoupled(py::module_ &module) {
  module.def("simulate_uncoupled",
      [](const Eigen::VectorXd &initial, double mu, double epoch,
         double dt, double tf, int record_every, py::object runtime,
         py::object adcs, std::uint64_t seed, std::set<std::string> fields,
         const std::string &attitude_mode,
         bool progress, std::size_t chunk_steps, py::object obc) {
        if (chunk_steps == 0)
          throw py::value_error("progress_chunk_steps must be positive");
        auto native_runtime = omni::propagation::runtime_from_python(runtime);
        std::vector<hardware::Sensor> sensors;
        for (py::handle sensor : adcs.attr("sensors"))
          sensors.push_back(as_sensor(sensor));
        const py::object estimator_object = adcs.attr("estimator");
        const auto *estimator = estimator_object.is_none() ? nullptr :
            estimator_object.cast<const hardware::estimator::AttitudeEstimator *>();
        const int sensor_every = adcs.attr("N").cast<int>();
        std::unique_ptr<omni::environment::DE421> own_eph;
        auto ctrl_ctx = build_control_context(adcs, obc, mu, native_runtime,
                                              sensors, own_eph);
        py::gil_scoped_release release;
        return omni::propagation::simulate_uncoupled(initial, mu, epoch,
            dt, tf, record_every, native_runtime, std::move(sensors), estimator,
            sensor_every, seed, std::move(fields), attitude_mode,
            progress, chunk_steps, &ctrl_ctx);
      }, py::arg("initial"), py::arg("mu"), py::arg("initial_julian_date"),
      py::arg("dt"), py::arg("tf"), py::arg("record_every"), py::arg("runtime"),
      py::arg("adcs"), py::arg("seed"), py::arg("fields"),
      py::arg("attitude_mode"),
      py::arg("progress") = false, py::arg("progress_chunk_steps") = 500,
      py::arg("obc") = py::none());
  module.def("simulate_coupled",
      [](const Eigen::VectorXd &initial, double epoch,
         double dt, double tf, int record_every, py::object runtime,
         py::object adcs, std::uint64_t seed, std::set<std::string> fields,
         bool adaptive, double rtol, Eigen::VectorXd atol, double max_step, double min_step,
         bool progress, std::size_t chunk_steps) {
        if (chunk_steps == 0)
          throw py::value_error("progress_chunk_steps must be positive");
        auto native_runtime = omni::propagation::runtime_from_python(runtime);
        std::vector<hardware::Sensor> sensors;
        for (py::handle sensor : adcs.attr("sensors"))
          sensors.push_back(as_sensor(sensor));
        const py::object estimator_object = adcs.attr("estimator");
        const auto *estimator = estimator_object.is_none() ? nullptr :
            estimator_object.cast<const hardware::estimator::AttitudeEstimator *>();
        const int sensor_every = adcs.attr("N").cast<int>();
        py::gil_scoped_release release;
        return omni::propagation::simulate_coupled(initial, epoch,
            dt, tf, record_every, native_runtime, std::move(sensors), estimator,
            sensor_every, seed, std::move(fields), adaptive, rtol, atol, max_step, min_step, progress, chunk_steps);
      }, py::arg("initial"), py::arg("initial_julian_date"),
      py::arg("dt"), py::arg("tf"), py::arg("record_every"), py::arg("runtime"),
      py::arg("adcs"), py::arg("seed"), py::arg("fields"),
      py::arg("adaptive"), py::arg("rtol"), py::arg("atol"),
      py::arg("max_step"), py::arg("min_step"),
      py::arg("progress") = false, py::arg("progress_chunk_steps") = 500);
  module.def("integrate_separate_reference",
             [](const Eigen::VectorXd &initial, double mu, double epoch,
                double dt, double tf, int record_every,
                const std::string &mode, bool progress,
                std::size_t chunk_steps) {
               if (chunk_steps == 0) throw py::value_error("progress_chunk_steps must be positive");
               std::size_t completed = 0;
               const auto started = std::chrono::steady_clock::now();
               omni::propagation::SeparateProgress update;
               if (progress) update = [&](std::size_t count) {
                 completed += count;
                 if (completed % chunk_steps == 0 || completed * dt >= tf) {
                   const double seconds = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - started).count();
                   std::cerr << "\rRunning Keplerian + reference: " << completed
                             << "/" << static_cast<std::size_t>(std::ceil(tf / dt))
                             << " steps (" << completed / std::max(seconds, 1e-9)
                             << " step/s)" << std::flush;
                 }
               };
               py::gil_scoped_release release;
               auto result = omni::propagation::integrate_separate_reference(
                   initial, mu, epoch, dt, tf, record_every, mode, update);
               if (progress) std::cerr << '\n';
               return result;
             },
             py::arg("initial"), py::arg("mu"), py::arg("initial_julian_date"),
             py::arg("dt"), py::arg("tf"), py::arg("record_every"),
             py::arg("mode"), py::arg("progress") = false,
             py::arg("progress_chunk_steps") = 500);
  module.def("integrate_separate_rk4",
             [](const Eigen::VectorXd &initial, double mu,
                double initial_julian_date, double dt, double tf,
                int record_every, py::object runtime, bool progress,
                py::object progress_callback, std::size_t chunk_steps) {
               if (chunk_steps == 0) throw py::value_error("progress_chunk_steps must be positive");
               auto rt = omni::propagation::runtime_from_python(runtime);
               auto result = omni::propagation::integrate_separate_rk4(
                   initial, mu, initial_julian_date, dt, tf, record_every, rt);
               // Notify Python only after the native integration has completed.
               if (progress && !progress_callback.is_none())
                 progress_callback(static_cast<std::size_t>(std::ceil(tf / dt)));
               return result;
             }, py::arg("initial"), py::arg("mu"),
             py::arg("initial_julian_date"), py::arg("dt"), py::arg("tf"),
             py::arg("record_every"), py::arg("runtime"),
             py::arg("progress") = false,
             py::arg("progress_callback") = py::none(),
             py::arg("progress_chunk_steps") = 500);
}
