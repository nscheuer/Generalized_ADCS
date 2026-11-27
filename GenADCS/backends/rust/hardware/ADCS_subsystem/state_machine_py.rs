use std::collections::BTreeMap;
use std::sync::Arc;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use super::controllers::{Controller, NoController};
use super::controllers::controllers_py::{
    PyAttitudeController, PyBCross, PyBDot, PyNoController,
};
use super::goals::{Goal, NoGoal};
use super::state_machine::{
    ADCSMode as NativeADCSMode, ADCSStateMachine as NativeADCSStateMachine, ADCSTimelineTarget,
    DeviceMaskSpec, GOAL_NONE, GOAL_PARAM_WIDTH,
};

fn parse_mask(obj: Option<&PyAny>) -> PyResult<DeviceMaskSpec> {
    let mut map = BTreeMap::new();
    let Some(val) = obj else {
        return Ok(map);
    };
    if val.is_none() {
        return Ok(map);
    }
    let dict = val.downcast::<PyDict>()?;
    for (k, v) in dict.iter() {
        map.insert(k.extract::<String>()?, v.extract::<bool>()?);
    }
    Ok(map)
}

#[pyclass(name = "ADCSMode")]
#[derive(Clone)]
pub struct PyADCSMode {
    pub(crate) inner: NativeADCSMode,
    goal_py: Py<PyAny>,
    controller_py: Py<PyAny>,
}

#[pymethods]
impl PyADCSMode {
    #[new]
    #[pyo3(signature = (name, goal=None, controller=None, sensors=None, actuators=None))]
    fn new(
        py: Python<'_>,
        name: String,
        goal: Option<Py<PyAny>>,
        controller: Option<Py<PyAny>>,
        sensors: Option<&PyAny>,
        actuators: Option<&PyAny>,
    ) -> PyResult<Self> {
        let goal_py = match goal {
            Some(g) => g,
            None => py
                .import("GenADCS.backends.rust.hardware.ADCS_subsystem.goals")?
                .getattr("NoGoal")?
                .call0()?
                .into_py(py),
        };
        let ctrl_py = match controller {
            Some(c) => {
                let c_ref = c.as_ref(py);
                let is_ctrl = c_ref.extract::<PyRef<PyNoController>>().is_ok()
                    || c_ref.extract::<PyRef<PyBDot>>().is_ok()
                    || c_ref.extract::<PyRef<PyBCross>>().is_ok()
                    || c_ref.extract::<PyRef<PyAttitudeController>>().is_ok();
                if !is_ctrl {
                    return Err(PyTypeError::new_err(
                        "ADCSMode controller must be a Controller instance",
                    ));
                }
                c
            }
            None => Py::new(py, PyNoController::new())?.into_py(py),
        };
        let g_native: Arc<dyn Goal + Send + Sync> = Arc::new(NoGoal);
        let c_native: Arc<dyn Controller> = Arc::new(NoController);
        let inner = NativeADCSMode::new(
            name,
            Some(g_native),
            GOAL_NONE,
            [0.0; GOAL_PARAM_WIDTH],
            "NoGoal".to_string(),
            Some(c_native),
            parse_mask(sensors)?,
            parse_mask(actuators)?,
        )
        .map_err(PyValueError::new_err)?;
        Ok(Self {
            inner,
            goal_py,
            controller_py: ctrl_py,
        })
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn goal(&self, py: Python<'_>) -> Py<PyAny> {
        self.goal_py.clone_ref(py)
    }

    #[getter]
    fn controller(&self, py: Python<'_>) -> Py<PyAny> {
        self.controller_py.clone_ref(py)
    }
}

#[pyclass(name = "ADCSStateMachine")]
pub struct PyADCSStateMachine {
    pub(crate) inner: NativeADCSStateMachine,
    modes_py: Vec<PyADCSMode>,
}

#[pymethods]
impl PyADCSStateMachine {
    #[new]
    #[pyo3(signature = (initial, modes, timeline=None))]
    fn new(
        initial: String,
        modes: Vec<PyADCSMode>,
        timeline: Option<&PyAny>,
    ) -> PyResult<Self> {
        let mut native_tl = Vec::new();
        if let Some(tl) = timeline {
            if !tl.is_none() {
                for item in tl.iter()? {
                    let tup = item?.downcast::<PyTuple>()?;
                    if tup.len() != 2 {
                        return Err(PyValueError::new_err(
                            "ADCS timeline entries must be (time, mode_or_goal)",
                        ));
                    }
                    let t: f64 = tup.get_item(0)?.extract()?;
                    if let Ok(m_name) = tup.get_item(1)?.extract::<String>() {
                        native_tl.push((t, ADCSTimelineTarget::ModeName(m_name)));
                    } else {
                        native_tl.push((
                            t,
                            ADCSTimelineTarget::GoalSpec {
                                goal: Arc::new(NoGoal),
                                kind: GOAL_NONE,
                                params: [0.0; GOAL_PARAM_WIDTH],
                                label: "Goal".to_string(),
                            },
                        ));
                    }
                }
            }
        }
        let native_modes: Vec<NativeADCSMode> = modes.iter().map(|m| m.inner.clone()).collect();
        let inner = NativeADCSStateMachine::new(initial, native_modes, native_tl)
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            inner,
            modes_py: modes,
        })
    }

    #[getter]
    fn initial(&self) -> String {
        self.inner.initial.clone()
    }

    #[getter]
    fn modes(&self) -> Vec<PyADCSMode> {
        self.modes_py.clone()
    }
}

pub fn register(module: &PyModule) -> PyResult<()> {
    module.add_class::<PyADCSMode>()?;
    module.add_class::<PyADCSStateMachine>()?;
    Ok(())
}
