use std::path::Path;
use std::sync::{Arc, OnceLock};

use ephemeris_native::NativeEphemeris;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBool;

use super::along_track_goal::AlongTrackGoal as NativeAlongTrackGoal;
use super::angular_rate_goal::AngularRateGoal as NativeAngularRateGoal;
use super::coordinate_fmc_goal::CoordinateFMCGoal as NativeCoordinateFMCGoal;
use super::coordinate_goal::CoordinateGoal as NativeCoordinateGoal;
use super::cross_track_goal::CrossTrackGoal as NativeCrossTrackGoal;
use super::goal::{Goal, GoalReference, DEFAULT_MU};
use super::nadir_goal::NadirGoal as NativeNadirGoal;
use super::no_goal::NoGoal as NativeNoGoal;
use super::quaternion_goal::QuaternionGoal as NativeQuaternionGoal;
use super::sun_goal::SunGoal as NativeSunGoal;
use super::two_vector_goal::TwoVectorGoal as NativeTwoVectorGoal;
use super::vector_goal::VectorGoal as NativeVectorGoal;
use super::velocity_goal::VelocityGoal as NativeVelocityGoal;
use super::zenith_goal::ZenithGoal as NativeZenithGoal;

static DEFAULT_SUN_EPHEMERIS: OnceLock<Arc<NativeEphemeris>> = OnceLock::new();

fn default_ephemeris(py: Python<'_>) -> PyResult<Arc<NativeEphemeris>> {
    if let Some(eph) = DEFAULT_SUN_EPHEMERIS.get() {
        return Ok(Arc::clone(eph));
    }
    let package_file: String = py.import("GenADCS")?.getattr("__file__")?.extract()?;
    let package_dir = Path::new(&package_file)
        .parent()
        .ok_or_else(|| PyIOError::new_err("cannot locate GenADCS package directory"))?;
    let kernel_path = package_dir.join("kernels/environment/ephemeris/de421/de421.bsp");
    let eph = Arc::new(NativeEphemeris::from_path(&kernel_path).map_err(PyValueError::new_err)?);
    let _ = DEFAULT_SUN_EPHEMERIS.set(Arc::clone(&eph));
    Ok(eph)
}

fn extract_mu(runtime: Option<&PyAny>) -> PyResult<f64> {
    match runtime {
        None => Ok(DEFAULT_MU),
        Some(rt) => rt.getattr("mu")?.extract::<f64>(),
    }
}

fn extract_state(py: Python<'_>, state: &PyAny) -> PyResult<Vec<f64>> {
    let array = py
        .import("numpy")?
        .call_method1("asarray", (state, "float64"))?;
    let readonly = array.extract::<PyReadonlyArray1<f64>>().map_err(|_| {
        PyValueError::new_err("goal state must have a finite 13-element spacecraft-state prefix")
    })?;
    Ok(readonly.to_vec()?)
}

fn extract_vec<const N: usize>(
    py: Python<'_>,
    value: &PyAny,
    err_msg: &str,
) -> PyResult<[f64; N]> {
    let array = py
        .import("numpy")?
        .call_method1("asarray", (value, "float64"))?;
    let readonly = array
        .extract::<PyReadonlyArray1<f64>>()
        .map_err(|_| PyValueError::new_err(err_msg.to_string()))?;
    let vec = readonly.to_vec()?;
    vec.try_into()
        .map_err(|_| PyValueError::new_err(err_msg.to_string()))
}

fn extract_mat2x3(py: Python<'_>, value: &PyAny) -> PyResult<[[f64; 3]; 2]> {
    let err_msg = "inertial_vectors and body_vectors must each have shape (2, 3)";
    let array = py
        .import("numpy")?
        .call_method1("asarray", (value, "float64"))?;
    let readonly = array
        .extract::<PyReadonlyArray2<f64>>()
        .map_err(|_| PyValueError::new_err(err_msg))?;
    if readonly.shape() != [2, 3] {
        return Err(PyValueError::new_err(err_msg));
    }
    let view = readonly.as_array();
    Ok([
        [view[[0, 0]], view[[0, 1]], view[[0, 2]]],
        [view[[1, 0]], view[[1, 1]], view[[1, 2]]],
    ])
}

fn eval_goal_py(
    py: Python<'_>,
    goal: &dyn Goal,
    state: &PyAny,
    epoch_jd: f64,
    runtime: Option<&PyAny>,
) -> PyResult<(PyObject, PyObject)> {
    let state_vec = extract_state(py, state)?;
    let mu = extract_mu(runtime)?;
    let (q_opt, r_opt): GoalReference = goal
        .reference(&state_vec, epoch_jd, mu)
        .map_err(PyValueError::new_err)?;
    let q_py = match q_opt {
        Some(q) => q.to_pyarray(py).into_py(py),
        None => py.None(),
    };
    let r_py = match r_opt {
        Some(r) => r.to_pyarray(py).into_py(py),
        None => py.None(),
    };
    Ok((q_py, r_py))
}

#[pyclass(name = "Goal", subclass)]
pub struct PyGoal;

#[pyclass(name = "NoGoal", extends = PyGoal)]
pub struct PyNoGoal {
    inner: NativeNoGoal,
}

#[pymethods]
impl PyNoGoal {
    #[new]
    fn new() -> (Self, PyGoal) {
        (
            Self {
                inner: NativeNoGoal::new(),
            },
            PyGoal,
        )
    }

    #[pyo3(signature = (state, epoch_jd, runtime=None))]
    fn reference(
        &self,
        py: Python<'_>,
        state: &PyAny,
        epoch_jd: f64,
        runtime: Option<&PyAny>,
    ) -> PyResult<(PyObject, PyObject)> {
        eval_goal_py(py, &self.inner, state, epoch_jd, runtime)
    }

    #[pyo3(signature = (state, epoch_jd, runtime=None))]
    fn __call__(
        &self,
        py: Python<'_>,
        state: &PyAny,
        epoch_jd: f64,
        runtime: Option<&PyAny>,
    ) -> PyResult<(PyObject, PyObject)> {
        eval_goal_py(py, &self.inner, state, epoch_jd, runtime)
    }
}

#[pyclass(name = "QuaternionGoal", extends = PyGoal)]
pub struct PyQuaternionGoal {
    inner: NativeQuaternionGoal,
}

#[pymethods]
impl PyQuaternionGoal {
    #[new]
    #[pyo3(signature = (quaternion, rate=None))]
    fn new(
        py: Python<'_>,
        quaternion: &PyAny,
        rate: Option<&PyAny>,
    ) -> PyResult<(Self, PyGoal)> {
        let q = extract_vec::<4>(py, quaternion, "quaternion must be a finite 4-element vector")?;
        let r = match rate {
            Some(val) => Some(extract_vec::<3>(
                py,
                val,
                "rate must be a finite 3-element vector",
            )?),
            None => None,
        };
        let inner = NativeQuaternionGoal::new(q, r).map_err(PyValueError::new_err)?;
        Ok((Self { inner }, PyGoal))
    }

    #[getter]
    fn quaternion(&self, py: Python<'_>) -> PyObject {
        self.inner.quaternion.to_pyarray(py).into_py(py)
    }

    #[getter]
    fn rate(&self, py: Python<'_>) -> PyObject {
        self.inner.rate.to_pyarray(py).into_py(py)
    }

    #[pyo3(signature = (state, epoch_jd, runtime=None))]
    fn reference(
        &self,
        py: Python<'_>,
        state: &PyAny,
        epoch_jd: f64,
        runtime: Option<&PyAny>,
    ) -> PyResult<(PyObject, PyObject)> {
        eval_goal_py(py, &self.inner, state, epoch_jd, runtime)
    }

    #[pyo3(signature = (state, epoch_jd, runtime=None))]
    fn __call__(
        &self,
        py: Python<'_>,
        state: &PyAny,
        epoch_jd: f64,
        runtime: Option<&PyAny>,
    ) -> PyResult<(PyObject, PyObject)> {
        eval_goal_py(py, &self.inner, state, epoch_jd, runtime)
    }
}

#[pyclass(name = "AngularRateGoal", extends = PyGoal)]
pub struct PyAngularRateGoal {
    inner: NativeAngularRateGoal,
}

#[pymethods]
impl PyAngularRateGoal {
    #[new]
    fn new(py: Python<'_>, rate: &PyAny) -> PyResult<(Self, PyGoal)> {
        let r = extract_vec::<3>(py, rate, "rate must be a finite 3-element vector")?;
        let inner = NativeAngularRateGoal::new(r).map_err(PyValueError::new_err)?;
        Ok((Self { inner }, PyGoal))
    }

    #[getter]
    fn rate(&self, py: Python<'_>) -> PyObject {
        self.inner.rate.to_pyarray(py).into_py(py)
    }

    #[pyo3(signature = (state, epoch_jd, runtime=None))]
    fn reference(
        &self,
        py: Python<'_>,
        state: &PyAny,
        epoch_jd: f64,
        runtime: Option<&PyAny>,
    ) -> PyResult<(PyObject, PyObject)> {
        eval_goal_py(py, &self.inner, state, epoch_jd, runtime)
    }

    #[pyo3(signature = (state, epoch_jd, runtime=None))]
    fn __call__(
        &self,
        py: Python<'_>,
        state: &PyAny,
        epoch_jd: f64,
        runtime: Option<&PyAny>,
    ) -> PyResult<(PyObject, PyObject)> {
        eval_goal_py(py, &self.inner, state, epoch_jd, runtime)
    }
}

#[pyclass(name = "VectorGoal", extends = PyGoal)]
pub struct PyVectorGoal {
    inner: NativeVectorGoal,
}

#[pymethods]
impl PyVectorGoal {
    #[new]
    #[pyo3(signature = (inertial_vector, body_axis=None, *, inertial_roll=None, body_roll=None))]
    fn new(
        py: Python<'_>,
        inertial_vector: &PyAny,
        body_axis: Option<&PyAny>,
        inertial_roll: Option<&PyAny>,
        body_roll: Option<&PyAny>,
    ) -> PyResult<(Self, PyGoal)> {
        let iv = extract_vec::<3>(
            py,
            inertial_vector,
            "inertial_vector must be a finite 3-element vector",
        )?;
        let ba = body_axis
            .map(|v| extract_vec::<3>(py, v, "body_axis must be a finite 3-element vector"))
            .transpose()?;
        let ir = inertial_roll
            .map(|v| extract_vec::<3>(py, v, "inertial_roll must be a finite 3-element vector"))
            .transpose()?;
        let br = body_roll
            .map(|v| extract_vec::<3>(py, v, "body_roll must be a finite 3-element vector"))
            .transpose()?;
        let inner = NativeVectorGoal::new(iv, ba, ir, br).map_err(PyValueError::new_err)?;
        Ok((Self { inner }, PyGoal))
    }

    #[getter]
    fn inertial_vector(&self, py: Python<'_>) -> PyObject {
        self.inner.inertial_vector.to_pyarray(py).into_py(py)
    }

    #[getter]
    fn body_axis(&self, py: Python<'_>) -> PyObject {
        self.inner.body_axis.to_pyarray(py).into_py(py)
    }

    #[getter]
    fn inertial_roll(&self, py: Python<'_>) -> PyObject {
        self.inner.inertial_roll.to_pyarray(py).into_py(py)
    }

    #[getter]
    fn body_roll(&self, py: Python<'_>) -> PyObject {
        self.inner.body_roll.to_pyarray(py).into_py(py)
    }

    #[pyo3(signature = (state, epoch_jd, runtime=None))]
    fn reference(
        &self,
        py: Python<'_>,
        state: &PyAny,
        epoch_jd: f64,
        runtime: Option<&PyAny>,
    ) -> PyResult<(PyObject, PyObject)> {
        eval_goal_py(py, &self.inner, state, epoch_jd, runtime)
    }

    #[pyo3(signature = (state, epoch_jd, runtime=None))]
    fn __call__(
        &self,
        py: Python<'_>,
        state: &PyAny,
        epoch_jd: f64,
        runtime: Option<&PyAny>,
    ) -> PyResult<(PyObject, PyObject)> {
        eval_goal_py(py, &self.inner, state, epoch_jd, runtime)
    }
}

#[pyclass(name = "TwoVectorGoal", extends = PyGoal)]
pub struct PyTwoVectorGoal {
    inner: NativeTwoVectorGoal,
}

#[pymethods]
impl PyTwoVectorGoal {
    #[new]
    fn new(
        py: Python<'_>,
        inertial_vectors: &PyAny,
        body_vectors: &PyAny,
    ) -> PyResult<(Self, PyGoal)> {
        let iv = extract_mat2x3(py, inertial_vectors)?;
        let bv = extract_mat2x3(py, body_vectors)?;
        let inner = NativeTwoVectorGoal::new(iv, bv).map_err(PyValueError::new_err)?;
        Ok((Self { inner }, PyGoal))
    }

    #[pyo3(signature = (state, epoch_jd, runtime=None))]
    fn reference(
        &self,
        py: Python<'_>,
        state: &PyAny,
        epoch_jd: f64,
        runtime: Option<&PyAny>,
    ) -> PyResult<(PyObject, PyObject)> {
        eval_goal_py(py, &self.inner, state, epoch_jd, runtime)
    }

    #[pyo3(signature = (state, epoch_jd, runtime=None))]
    fn __call__(
        &self,
        py: Python<'_>,
        state: &PyAny,
        epoch_jd: f64,
        runtime: Option<&PyAny>,
    ) -> PyResult<(PyObject, PyObject)> {
        eval_goal_py(py, &self.inner, state, epoch_jd, runtime)
    }
}

macro_rules! define_py_simple_orbit_goal {
    ($py_name:ident, $native_name:ident, $py_str_name:literal) => {
        #[pyclass(name = $py_str_name, extends = PyGoal)]
        pub struct $py_name {
            inner: $native_name,
        }

        #[pymethods]
        impl $py_name {
            #[new]
            fn new() -> (Self, PyGoal) {
                (
                    Self {
                        inner: $native_name::new(),
                    },
                    PyGoal,
                )
            }

            #[pyo3(signature = (state, epoch_jd, runtime=None))]
            fn reference(
                &self,
                py: Python<'_>,
                state: &PyAny,
                epoch_jd: f64,
                runtime: Option<&PyAny>,
            ) -> PyResult<(PyObject, PyObject)> {
                eval_goal_py(py, &self.inner, state, epoch_jd, runtime)
            }

            #[pyo3(signature = (state, epoch_jd, runtime=None))]
            fn __call__(
                &self,
                py: Python<'_>,
                state: &PyAny,
                epoch_jd: f64,
                runtime: Option<&PyAny>,
            ) -> PyResult<(PyObject, PyObject)> {
                eval_goal_py(py, &self.inner, state, epoch_jd, runtime)
            }
        }
    };
}

define_py_simple_orbit_goal!(PyNadirGoal, NativeNadirGoal, "NadirGoal");
define_py_simple_orbit_goal!(PyZenithGoal, NativeZenithGoal, "ZenithGoal");
define_py_simple_orbit_goal!(PyAlongTrackGoal, NativeAlongTrackGoal, "AlongTrackGoal");
define_py_simple_orbit_goal!(PyCrossTrackGoal, NativeCrossTrackGoal, "CrossTrackGoal");
define_py_simple_orbit_goal!(PyVelocityGoal, NativeVelocityGoal, "VelocityGoal");

#[pyclass(name = "SunGoal", extends = PyGoal)]
pub struct PySunGoal {
    inner: NativeSunGoal,
}

#[pymethods]
impl PySunGoal {
    #[new]
    fn new(py: Python<'_>) -> PyResult<(Self, PyGoal)> {
        let eph = default_ephemeris(py)?;
        Ok((
            Self {
                inner: NativeSunGoal::from_ephemeris(eph),
            },
            PyGoal,
        ))
    }

    #[pyo3(signature = (state, epoch_jd, runtime=None))]
    fn reference(
        &self,
        py: Python<'_>,
        state: &PyAny,
        epoch_jd: f64,
        runtime: Option<&PyAny>,
    ) -> PyResult<(PyObject, PyObject)> {
        eval_goal_py(py, &self.inner, state, epoch_jd, runtime)
    }

    #[pyo3(signature = (state, epoch_jd, runtime=None))]
    fn __call__(
        &self,
        py: Python<'_>,
        state: &PyAny,
        epoch_jd: f64,
        runtime: Option<&PyAny>,
    ) -> PyResult<(PyObject, PyObject)> {
        eval_goal_py(py, &self.inner, state, epoch_jd, runtime)
    }
}

#[pyclass(name = "CoordinateGoal", extends = PyGoal, subclass)]
pub struct PyCoordinateGoal {
    inner: NativeCoordinateGoal,
}

#[pymethods]
impl PyCoordinateGoal {
    #[new]
    #[pyo3(signature = (lat, lon, alt=0.0))]
    fn new(lat: f64, lon: f64, alt: f64) -> PyResult<(Self, PyGoal)> {
        let inner = NativeCoordinateGoal::new(lat, lon, Some(alt)).map_err(PyValueError::new_err)?;
        Ok((Self { inner }, PyGoal))
    }

    #[getter]
    fn lat_deg(&self) -> f64 {
        self.inner.lat_deg
    }

    #[getter]
    fn lon_deg(&self) -> f64 {
        self.inner.lon_deg
    }

    #[getter]
    fn alt_km(&self) -> f64 {
        self.inner.alt_km
    }

    #[getter]
    fn target_ecef_m(&self, py: Python<'_>) -> PyObject {
        self.inner.target_ecef_m.to_pyarray(py).into_py(py)
    }

    #[pyo3(signature = (state, epoch_jd, runtime=None))]
    fn reference(
        &self,
        py: Python<'_>,
        state: &PyAny,
        epoch_jd: f64,
        runtime: Option<&PyAny>,
    ) -> PyResult<(PyObject, PyObject)> {
        eval_goal_py(py, &self.inner, state, epoch_jd, runtime)
    }

    #[pyo3(signature = (state, epoch_jd, runtime=None))]
    fn __call__(
        &self,
        py: Python<'_>,
        state: &PyAny,
        epoch_jd: f64,
        runtime: Option<&PyAny>,
    ) -> PyResult<(PyObject, PyObject)> {
        eval_goal_py(py, &self.inner, state, epoch_jd, runtime)
    }
}

#[pyclass(name = "CoordinateFMCGoal", extends = PyGoal)]
pub struct PyCoordinateFMCGoal {
    inner: NativeCoordinateFMCGoal,
}

#[pymethods]
impl PyCoordinateFMCGoal {
    #[new]
    #[pyo3(signature = (lat, lon, alt=0.0, fmc_level=None))]
    fn new(lat: f64, lon: f64, alt: f64, fmc_level: Option<&PyAny>) -> PyResult<(Self, PyGoal)> {
        let level = match fmc_level {
            None => 1,
            Some(val) => {
                if val.is_instance_of::<PyBool>() {
                    return Err(PyValueError::new_err(
                        "fmc_level must be an integer from 1 through 10",
                    ));
                }
                val.extract::<i64>().map_err(|_| {
                    PyValueError::new_err("fmc_level must be an integer from 1 through 10")
                })?
            }
        };
        let inner = NativeCoordinateFMCGoal::new(lat, lon, Some(alt), Some(level))
            .map_err(PyValueError::new_err)?;
        Ok((Self { inner }, PyGoal))
    }

    #[getter]
    fn lat_deg(&self) -> f64 {
        self.inner.coordinate.lat_deg
    }

    #[getter]
    fn lon_deg(&self) -> f64 {
        self.inner.coordinate.lon_deg
    }

    #[getter]
    fn alt_km(&self) -> f64 {
        self.inner.coordinate.alt_km
    }

    #[getter]
    fn target_ecef_m(&self, py: Python<'_>) -> PyObject {
        self.inner.coordinate.target_ecef_m.to_pyarray(py).into_py(py)
    }

    #[getter]
    fn fmc_level(&self) -> i64 {
        self.inner.fmc_level
    }

    #[pyo3(signature = (state, epoch_jd, runtime=None))]
    fn reference(
        &self,
        py: Python<'_>,
        state: &PyAny,
        epoch_jd: f64,
        runtime: Option<&PyAny>,
    ) -> PyResult<(PyObject, PyObject)> {
        eval_goal_py(py, &self.inner, state, epoch_jd, runtime)
    }

    #[pyo3(signature = (state, epoch_jd, runtime=None))]
    fn __call__(
        &self,
        py: Python<'_>,
        state: &PyAny,
        epoch_jd: f64,
        runtime: Option<&PyAny>,
    ) -> PyResult<(PyObject, PyObject)> {
        eval_goal_py(py, &self.inner, state, epoch_jd, runtime)
    }
}

pub fn register(module: &pyo3::types::PyModule) -> PyResult<()> {
    module.add_class::<PyGoal>()?;
    module.add_class::<PyNoGoal>()?;
    module.add_class::<PyQuaternionGoal>()?;
    module.add_class::<PyAngularRateGoal>()?;
    module.add_class::<PyVectorGoal>()?;
    module.add_class::<PyTwoVectorGoal>()?;
    module.add_class::<PyNadirGoal>()?;
    module.add_class::<PyZenithGoal>()?;
    module.add_class::<PyAlongTrackGoal>()?;
    module.add_class::<PyCrossTrackGoal>()?;
    module.add_class::<PyVelocityGoal>()?;
    module.add_class::<PySunGoal>()?;
    module.add_class::<PyCoordinateGoal>()?;
    module.add_class::<PyCoordinateFMCGoal>()?;
    Ok(())
}
