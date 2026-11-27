use std::collections::BTreeMap;
use std::sync::Arc;

use adcs_subsystem::goals::{Goal, NoGoal};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyTuple};

use super::fdir::{AngularRateMonitor as NativeAngularRateMonitor, FDIR as NativeFDIR};
use super::obc_subsystem::OBCSubsystem as NativeOBCSubsystem;
use super::state_machine::{
    ADCSBinding, After as NativeAfter, Completion as NativeCompletion,
    FDIRTriggered as NativeFDIRTriggered, HardwareMaskSpec, InFOV as NativeInFOV,
    Mode as NativeMode, ModeBindings, StateMachine as NativeStateMachine,
    Telecommand as NativeTelecommand, TimelineEntry, Transition as NativeTransition,
    Trigger as NativeTrigger,
};

fn parse_mask(obj: Option<&PyAny>) -> PyResult<Option<HardwareMaskSpec>> {
    let Some(val) = obj else {
        return Ok(None);
    };
    if val.is_none() {
        return Ok(None);
    }
    let dict = val.downcast::<PyDict>()?;
    let mut map = BTreeMap::new();
    for (k, v) in dict.iter() {
        map.insert(k.extract::<String>()?, v.extract::<bool>()?);
    }
    Ok(Some(map))
}

fn parse_goal_placeholder(goal: Option<Py<PyAny>>) -> Option<Arc<dyn Goal + Send + Sync>> {
    goal.map(|_| Arc::new(NoGoal) as Arc<dyn Goal + Send + Sync>)
}

fn parse_timeline(py: Python<'_>, obj: Option<&PyAny>) -> PyResult<Vec<TimelineEntry>> {
    let Some(val) = obj else {
        return Ok(Vec::new());
    };
    if val.is_none() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for item in val.iter()? {
        let tup = item?.downcast::<PyTuple>()?;
        if tup.len() != 2 && tup.len() != 3 {
            return Err(PyValueError::new_err(
                "timeline entries must be (time, command) or (time, command, goal)",
            ));
        }
        let t: f64 = tup.get_item(0)?.extract()?;
        let cmd: String = tup.get_item(1)?.extract()?;
        let g = if tup.len() == 3 && !tup.get_item(2)?.is_none() {
            Some(Arc::new(NoGoal) as Arc<dyn Goal + Send + Sync>)
        } else {
            None
        };
        let _ = py;
        out.push((t, cmd, g));
    }
    Ok(out)
}

#[pyclass(name = "FDIRMonitor", subclass)]
pub struct PyFDIRMonitor;

#[pyclass(name = "AngularRateMonitor", extends = PyFDIRMonitor)]
#[derive(Clone)]
pub struct PyAngularRateMonitor {
    pub(crate) inner: NativeAngularRateMonitor,
}

#[pymethods]
impl PyAngularRateMonitor {
    #[new]
    #[pyo3(signature = (max_rate, name="angular_rate".to_string(), target_mode=None))]
    fn new(
        max_rate: f64,
        name: String,
        target_mode: Option<String>,
    ) -> PyResult<(Self, PyFDIRMonitor)> {
        let inner = NativeAngularRateMonitor::new(max_rate, Some(name), target_mode)
            .map_err(PyValueError::new_err)?;
        Ok((Self { inner }, PyFDIRMonitor))
    }

    #[getter]
    fn max_rate(&self) -> f64 {
        self.inner.max_rate
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn target_mode(&self) -> Option<String> {
        self.inner.target_mode.clone()
    }
}

#[pyclass(name = "FDIR")]
#[derive(Clone)]
pub struct PyFDIR {
    pub(crate) inner: NativeFDIR,
    monitors_py: Vec<Py<PyAny>>,
}

#[pymethods]
impl PyFDIR {
    #[new]
    #[pyo3(signature = (monitors=Vec::new(), safe_mode="safe_mode".to_string()))]
    fn new(
        py: Python<'_>,
        monitors: Vec<Py<PyAny>>,
        safe_mode: String,
    ) -> PyResult<Self> {
        let mut native_mons = Vec::with_capacity(monitors.len());
        for m in &monitors {
            if let Ok(arm) = m.extract::<PyRef<PyAngularRateMonitor>>(py) {
                native_mons.push(arm.inner.clone());
            } else {
                return Err(PyTypeError::new_err("unsupported FDIR monitor"));
            }
        }
        let inner = NativeFDIR::new(native_mons, Some(safe_mode)).map_err(PyValueError::new_err)?;
        Ok(Self {
            inner,
            monitors_py: monitors,
        })
    }

    #[getter]
    fn monitors(&self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.monitors_py.iter().map(|m| m.clone_ref(py)).collect()
    }

    #[getter]
    fn safe_mode(&self) -> String {
        self.inner.safe_mode.clone()
    }
}

#[pyclass(name = "Trigger", subclass)]
pub struct PyTrigger;

#[pyclass(name = "Telecommand", extends = PyTrigger)]
pub struct PyTelecommand {
    pub(crate) inner: NativeTelecommand,
    goal_py: Option<Py<PyAny>>,
    sensors_py: Option<Py<PyAny>>,
    actuators_py: Option<Py<PyAny>>,
}

#[pymethods]
impl PyTelecommand {
    #[new]
    #[pyo3(signature = (command, goal=None, sensors=None, actuators=None))]
    fn new(
        py: Python<'_>,
        command: String,
        goal: Option<Py<PyAny>>,
        sensors: Option<Py<PyAny>>,
        actuators: Option<Py<PyAny>>,
    ) -> PyResult<(Self, PyTrigger)> {
        let s_mask = parse_mask(sensors.as_ref().map(|o| o.as_ref(py)))?;
        let a_mask = parse_mask(actuators.as_ref().map(|o| o.as_ref(py)))?;
        let g_native = parse_goal_placeholder(goal.as_ref().map(|g| g.clone_ref(py)));
        let inner = NativeTelecommand::new(command, g_native, s_mask, a_mask)
            .map_err(PyValueError::new_err)?;
        Ok((
            Self {
                inner,
                goal_py: goal,
                sensors_py: sensors,
                actuators_py: actuators,
            },
            PyTrigger,
        ))
    }

    #[getter]
    fn command(&self) -> String {
        self.inner.command.clone()
    }

    #[getter]
    fn goal(&self, py: Python<'_>) -> Py<PyAny> {
        match &self.goal_py {
            Some(g) => g.clone_ref(py),
            None => py.None(),
        }
    }

    #[getter]
    fn sensors(&self, py: Python<'_>) -> Py<PyAny> {
        match &self.sensors_py {
            Some(s) => s.clone_ref(py),
            None => py.None(),
        }
    }

    #[getter]
    fn actuators(&self, py: Python<'_>) -> Py<PyAny> {
        match &self.actuators_py {
            Some(a) => a.clone_ref(py),
            None => py.None(),
        }
    }
}

#[pyclass(name = "InFOV", extends = PyTrigger)]
pub struct PyInFOV {
    pub(crate) inner: NativeInFOV,
    goal_py: Option<Py<PyAny>>,
}

#[pymethods]
impl PyInFOV {
    #[new]
    #[pyo3(signature = (lat, lon, alt=0.0, max_off_nadir_deg=45.0, negate=false, goal=None))]
    fn new(
        py: Python<'_>,
        lat: f64,
        lon: f64,
        alt: f64,
        max_off_nadir_deg: f64,
        negate: bool,
        goal: Option<Py<PyAny>>,
    ) -> PyResult<(Self, PyTrigger)> {
        let g_native = parse_goal_placeholder(goal.as_ref().map(|g| g.clone_ref(py)));
        let inner = NativeInFOV::new(lat, lon, alt, max_off_nadir_deg, negate, g_native)
            .map_err(PyValueError::new_err)?;
        Ok((
            Self {
                inner,
                goal_py: goal,
            },
            PyTrigger,
        ))
    }

    fn __invert__(&self, py: Python<'_>) -> PyResult<Py<PyInFOV>> {
        Py::new(
            py,
            (
                Self {
                    inner: self.inner.inverted(),
                    goal_py: None,
                },
                PyTrigger,
            ),
        )
    }

    #[getter]
    fn lat(&self) -> f64 {
        self.inner.lat
    }

    #[getter]
    fn lon(&self) -> f64 {
        self.inner.lon
    }

    #[getter]
    fn alt(&self) -> f64 {
        self.inner.alt
    }

    #[getter]
    fn max_off_nadir_deg(&self) -> f64 {
        self.inner.max_off_nadir_deg
    }

    #[getter]
    fn negate(&self) -> bool {
        self.inner.negate
    }

    #[getter]
    fn goal(&self, py: Python<'_>) -> Py<PyAny> {
        match &self.goal_py {
            Some(g) => g.clone_ref(py),
            None => py.None(),
        }
    }
}

#[pyclass(name = "Completion", extends = PyTrigger)]
#[derive(Clone)]
pub struct PyCompletion;

#[pymethods]
impl PyCompletion {
    #[new]
    fn new() -> (Self, PyTrigger) {
        (Self, PyTrigger)
    }
}

#[pyclass(name = "After", extends = PyTrigger)]
#[derive(Clone)]
pub struct PyAfter {
    pub(crate) inner: NativeAfter,
}

#[pymethods]
impl PyAfter {
    #[new]
    fn new(duration: f64) -> PyResult<(Self, PyTrigger)> {
        let inner = NativeAfter::new(duration).map_err(PyValueError::new_err)?;
        Ok((Self { inner }, PyTrigger))
    }

    #[getter]
    fn duration(&self) -> f64 {
        self.inner.duration
    }
}

#[pyclass(name = "FDIRTriggered", extends = PyTrigger)]
#[derive(Clone)]
pub struct PyFDIRTriggered {
    pub(crate) inner: NativeFDIRTriggered,
}

#[pymethods]
impl PyFDIRTriggered {
    #[new]
    #[pyo3(signature = (monitor=None))]
    fn new(monitor: Option<String>) -> (Self, PyTrigger) {
        (
            Self {
                inner: NativeFDIRTriggered { monitor },
            },
            PyTrigger,
        )
    }

    #[getter]
    fn monitor(&self) -> Option<String> {
        self.inner.monitor.clone()
    }
}

#[pyclass(name = "Mode")]
#[derive(Clone)]
pub struct PyMode {
    pub(crate) inner: NativeMode,
    submodes_py: Vec<PyMode>,
    bindings_py: Option<Py<PyAny>>,
}

#[pymethods]
impl PyMode {
    #[new]
    #[pyo3(signature = (name, submodes=Vec::new(), initial_submode=None, duration=None, auto_target=None, bindings=None))]
    fn new(
        py: Python<'_>,
        name: String,
        submodes: Vec<PyMode>,
        initial_submode: Option<String>,
        duration: Option<f64>,
        auto_target: Option<String>,
        bindings: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let mut native_bindings = ModeBindings::default();
        if let Some(ref b_obj) = bindings {
            let b_ref = b_obj.as_ref(py);
            if !b_ref.is_none() {
                let d = b_ref.downcast::<PyDict>()?;
                if let Some(adcs_val) = d.get_item("adcs")? {
                    if let Ok(s) = adcs_val.extract::<String>() {
                        native_bindings.adcs = Some(ADCSBinding::ModeName(s));
                    } else if let Ok(tup) = adcs_val.downcast::<PyTuple>() {
                        if tup.len() != 2 {
                            return Err(PyValueError::new_err(
                                "adcs tuple binding must be (mode_name, Goal)",
                            ));
                        }
                        let m_name: String = tup.get_item(0)?.extract()?;
                        native_bindings.adcs = Some(ADCSBinding::ModeAndGoal(
                            m_name,
                            Arc::new(NoGoal),
                        ));
                    } else if !adcs_val.is_none() {
                        native_bindings.adcs = Some(ADCSBinding::GoalOnly(Arc::new(NoGoal)));
                    }
                }
                if let Some(s_val) = d.get_item("sensors")? {
                    native_bindings.sensors = parse_mask(Some(s_val))?;
                }
                if let Some(a_val) = d.get_item("actuators")? {
                    native_bindings.actuators = parse_mask(Some(a_val))?;
                }
            }
        }
        let native_subs: Vec<NativeMode> = submodes.iter().map(|m| m.inner.clone()).collect();
        let inner = NativeMode::new(
            name,
            native_subs,
            initial_submode,
            duration,
            auto_target,
            native_bindings,
        )
        .map_err(PyValueError::new_err)?;
        Ok(Self {
            inner,
            submodes_py: submodes,
            bindings_py: bindings,
        })
    }

    #[getter]
    fn submodes(&self) -> Vec<PyMode> {
        self.submodes_py.clone()
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn initial_submode(&self) -> Option<String> {
        self.inner.initial_submode.clone()
    }

    #[getter]
    fn duration(&self) -> Option<f64> {
        self.inner.duration
    }

    #[getter]
    fn auto_target(&self) -> Option<String> {
        self.inner.auto_target.clone()
    }

    #[getter]
    fn bindings(&self, py: Python<'_>) -> Py<PyAny> {
        match &self.bindings_py {
            Some(b) => b.clone_ref(py),
            None => PyDict::new(py).into_py(py),
        }
    }
}

#[pyclass(name = "Transition")]
pub struct PyTransition {
    pub(crate) inner: NativeTransition,
    trigger_py: Py<PyAny>,
}

#[pymethods]
impl PyTransition {
    #[new]
    #[pyo3(signature = (source, target, trigger, priority=0))]
    fn new(
        py: Python<'_>,
        source: String,
        target: String,
        trigger: Py<PyAny>,
        priority: i32,
    ) -> PyResult<Self> {
        let t_ref = trigger.as_ref(py);
        let native_trig = if let Ok(tc) = t_ref.extract::<PyRef<PyTelecommand>>() {
            NativeTrigger::Telecommand(tc.inner.clone())
        } else if let Ok(fov) = t_ref.extract::<PyRef<PyInFOV>>() {
            NativeTrigger::InFOV(fov.inner.clone())
        } else if t_ref.extract::<PyRef<PyCompletion>>().is_ok() {
            NativeTrigger::Completion(NativeCompletion)
        } else if let Ok(after) = t_ref.extract::<PyRef<PyAfter>>() {
            NativeTrigger::After(after.inner)
        } else if let Ok(fdir) = t_ref.extract::<PyRef<PyFDIRTriggered>>() {
            NativeTrigger::FDIRTriggered(fdir.inner.clone())
        } else {
            return Err(PyTypeError::new_err("invalid trigger"));
        };
        let inner = NativeTransition::new(source, target, native_trig, priority)
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            inner,
            trigger_py: trigger,
        })
    }

    #[getter]
    fn source(&self) -> String {
        self.inner.source.clone()
    }

    #[getter]
    fn target(&self) -> String {
        self.inner.target.clone()
    }

    #[getter]
    fn trigger(&self, py: Python<'_>) -> Py<PyAny> {
        self.trigger_py.clone_ref(py)
    }

    #[getter]
    fn priority(&self) -> i32 {
        self.inner.priority
    }
}

#[pyclass(name = "StateMachine")]
#[derive(Clone)]
pub struct PyStateMachine {
    pub(crate) inner: NativeStateMachine,
    modes_py: Vec<PyMode>,
    transitions_py: Vec<Py<PyAny>>,
    timeline_py: Py<PyAny>,
}

#[pymethods]
impl PyStateMachine {
    #[new]
    #[pyo3(signature = (initial, modes, transitions=Vec::new(), timeline=None))]
    fn new(
        py: Python<'_>,
        initial: String,
        modes: Vec<PyMode>,
        transitions: Vec<Py<PyAny>>,
        timeline: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let native_modes: Vec<NativeMode> = modes.iter().map(|m| m.inner.clone()).collect();
        let mut native_trans = Vec::with_capacity(transitions.len());
        for t in &transitions {
            let tr = t.as_ref(py).extract::<PyRef<PyTransition>>()?;
            native_trans.push(tr.inner.clone());
        }
        let tl_ref = timeline.as_ref().map(|o| o.as_ref(py));
        let native_tl = parse_timeline(py, tl_ref)?;
        let inner = NativeStateMachine::new(initial, native_modes, native_trans, native_tl)
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            inner,
            modes_py: modes,
            transitions_py: transitions,
            timeline_py: timeline.unwrap_or_else(|| py.None()),
        })
    }

    #[getter]
    fn initial(&self) -> String {
        self.inner.initial.clone()
    }

    #[getter]
    fn modes(&self) -> Vec<PyMode> {
        self.modes_py.clone()
    }

    #[getter]
    fn transitions(&self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.transitions_py.iter().map(|t| t.clone_ref(py)).collect()
    }

    #[getter]
    fn timeline(&self, py: Python<'_>) -> Py<PyAny> {
        self.timeline_py.clone_ref(py)
    }
}

#[pyclass(name = "OBCSubsystem")]
pub struct PyOBCSubsystem {
    pub(crate) inner: NativeOBCSubsystem,
    sm_py: Option<Py<PyAny>>,
    fdir_py: Option<Py<PyAny>>,
    timeline_py: Py<PyAny>,
}

#[pymethods]
impl PyOBCSubsystem {
    #[new]
    #[pyo3(signature = (state_machine=None, fdir=None, N=None, timeline=None))]
    #[allow(non_snake_case)]
    fn new(
        py: Python<'_>,
        state_machine: Option<Py<PyAny>>,
        fdir: Option<Py<PyAny>>,
        N: Option<&PyAny>,
        timeline: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let n_val: usize = match N {
            None => 1,
            Some(n_obj) => {
                if n_obj.is_instance_of::<PyBool>() {
                    return Err(PyValueError::new_err("N must be a positive integer"));
                }
                n_obj
                    .extract()
                    .map_err(|_| PyValueError::new_err("N must be a positive integer"))?
            }
        };
        let native_sm = match &state_machine {
            Some(obj) if !obj.is_none(py) => {
                Some(obj.extract::<PyRef<PyStateMachine>>(py)?.inner.clone())
            }
            _ => None,
        };
        let native_fdir = match &fdir {
            Some(obj) if !obj.is_none(py) => {
                Some(obj.extract::<PyRef<PyFDIR>>(py)?.inner.clone())
            }
            _ => None,
        };
        let tl_ref = timeline.as_ref().map(|o| o.as_ref(py));
        let native_tl = parse_timeline(py, tl_ref)?;
        let inner = NativeOBCSubsystem::new(native_sm, native_fdir, n_val, native_tl)
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            inner,
            sm_py: state_machine,
            fdir_py: fdir,
            timeline_py: timeline.unwrap_or_else(|| py.None()),
        })
    }

    #[getter]
    fn state_machine(&self, py: Python<'_>) -> Py<PyAny> {
        match &self.sm_py {
            Some(s) => s.clone_ref(py),
            None => py.None(),
        }
    }

    #[getter]
    fn fdir(&self, py: Python<'_>) -> Py<PyAny> {
        match &self.fdir_py {
            Some(f) => f.clone_ref(py),
            None => py.None(),
        }
    }

    #[getter]
    #[allow(non_snake_case)]
    fn N(&self) -> usize {
        self.inner.N
    }

    #[getter]
    fn timeline(&self, py: Python<'_>) -> Py<PyAny> {
        self.timeline_py.clone_ref(py)
    }
}

pub fn register(module: &PyModule) -> PyResult<()> {
    module.add_class::<PyFDIRMonitor>()?;
    module.add_class::<PyAngularRateMonitor>()?;
    module.add_class::<PyFDIR>()?;
    module.add_class::<PyTrigger>()?;
    module.add_class::<PyTelecommand>()?;
    module.add_class::<PyInFOV>()?;
    module.add_class::<PyCompletion>()?;
    module.add_class::<PyAfter>()?;
    module.add_class::<PyFDIRTriggered>()?;
    module.add_class::<PyMode>()?;
    module.add_class::<PyTransition>()?;
    module.add_class::<PyStateMachine>()?;
    module.add_class::<PyOBCSubsystem>()?;
    Ok(())
}
