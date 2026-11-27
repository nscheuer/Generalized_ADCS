use std::sync::Arc;

use nalgebra::DMatrix;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use super::allocator::{
    allocate_l2 as native_allocate_l2, allocate_linfinity as native_allocate_linfinity,
    Allocator as NativeAllocatorTrait, L2Allocator as NativeL2Allocator,
    LInfinityAllocator as NativeLInfinityAllocator, ALLOC_L2, ALLOC_LINFINITY,
};
use super::attitude_controller::AttitudeController as NativeAttitudeController;
use super::bcross::BCross as NativeBCross;
use super::bdot::BDot as NativeBDot;
use super::controller::{
    Controller as NativeControllerTrait, NoController as NativeNoController, CTRL_ATTITUDE,
    CTRL_BCROSS, CTRL_BDOT, CTRL_NONE,
};
use super::desaturation::{
    hogan_schaub_u0 as native_hogan_schaub_u0, Desaturation as NativeDesaturationTrait,
    HoganSchaubManagement as NativeHoganSchaubManagement, NoDesaturation as NativeNoDesaturation,
    DESAT_HOGAN_SCHAUB, DESAT_NONE,
};
use super::feedback::{
    mrp_feedback_linearized_step as native_mrp_feedback_linearized_step, to_gain_matrix,
    Feedback as NativeFeedbackTrait, GainInput, MRPFeedbackLinearized as NativeMRPFeedbackLinearized,
    Mat3, FB_MRP_FEEDBACK_LINEARIZED,
};
use super::feedforward::{
    passthrough_step as native_passthrough_step, Feedforward as NativeFeedforwardTrait,
    Passthrough as NativePassthrough, FF_PASSTHROUGH,
};

fn extract_gain_input(py: Python<'_>, value: &PyAny, name: &str) -> PyResult<GainInput> {
    let arr = py
        .import("numpy")?
        .call_method1("asarray", (value, "float64"))?;
    let ndim: usize = arr.getattr("ndim")?.extract()?;
    if ndim == 0 {
        let s: f64 = arr.call_method0("item")?.extract()?;
        return Ok(GainInput::Scalar(s));
    }
    if ndim == 1 {
        let ro = arr.extract::<PyReadonlyArray1<f64>>()?;
        let sl = ro.as_slice()?;
        if sl.len() == 3 {
            return Ok(GainInput::Vector([sl[0], sl[1], sl[2]]));
        }
    }
    if ndim == 2 {
        let ro = arr.extract::<PyReadonlyArray2<f64>>()?;
        let sh = ro.shape();
        if sh[0] == 3 && sh[1] == 3 {
            let v = ro.as_array();
            return Ok(GainInput::Matrix([
                [v[(0, 0)], v[(0, 1)], v[(0, 2)]],
                [v[(1, 0)], v[(1, 1)], v[(1, 2)]],
                [v[(2, 0)], v[(2, 1)], v[(2, 2)]],
            ]));
        }
    }
    Err(PyValueError::new_err(format!(
        "{name} must be a scalar, 3-vector, or 3x3 matrix"
    )))
}

fn mat3_to_py<'py>(py: Python<'py>, m: &Mat3) -> &'py PyArray2<f64> {
    PyArray2::from_vec2(py, &[m[0].to_vec(), m[1].to_vec(), m[2].to_vec()])
        .expect("3x3 matrix conversion")
}

fn extract_vec3(py: Python<'_>, val: &PyAny) -> PyResult<[f64; 3]> {
    let arr = py.import("numpy")?.call_method1("asarray", (val, "float64"))?;
    let ro = arr.extract::<PyReadonlyArray1<f64>>()?;
    let sl = ro.as_slice()?;
    if sl.len() != 3 {
        return Err(PyValueError::new_err("expected 3-element vector"));
    }
    Ok([sl[0], sl[1], sl[2]])
}

fn extract_vec4(py: Python<'_>, val: &PyAny) -> PyResult<[f64; 4]> {
    let arr = py.import("numpy")?.call_method1("asarray", (val, "float64"))?;
    let ro = arr.extract::<PyReadonlyArray1<f64>>()?;
    let sl = ro.as_slice()?;
    if sl.len() != 4 {
        return Err(PyValueError::new_err("expected 4-element vector"));
    }
    Ok([sl[0], sl[1], sl[2], sl[3]])
}

fn extract_mat3(py: Python<'_>, val: &PyAny) -> PyResult<Mat3> {
    let arr = py.import("numpy")?.call_method1("asarray", (val, "float64"))?;
    let ro = arr.extract::<PyReadonlyArray2<f64>>()?;
    let sh = ro.shape();
    if sh[0] != 3 || sh[1] != 3 {
        return Err(PyValueError::new_err("expected 3x3 matrix"));
    }
    let v = ro.as_array();
    Ok([
        [v[(0, 0)], v[(0, 1)], v[(0, 2)]],
        [v[(1, 0)], v[(1, 1)], v[(1, 2)]],
        [v[(2, 0)], v[(2, 1)], v[(2, 2)]],
    ])
}

fn extract_dmatrix(py: Python<'_>, val: &PyAny) -> PyResult<DMatrix<f64>> {
    let arr = py.import("numpy")?.call_method1("asarray", (val, "float64"))?;
    let ro = arr.extract::<PyReadonlyArray2<f64>>()?;
    let sh = ro.shape();
    let v = ro.as_array();
    Ok(DMatrix::from_fn(sh[0], sh[1], |r, c| v[(r, c)]))
}

fn extract_sensor_names(sensors: &PyAny) -> PyResult<Vec<String>> {
    let mut names = Vec::new();
    for item in sensors.iter()? {
        let obj = item?;
        let name: String = obj.get_type().getattr("__name__")?.extract()?;
        names.push(name);
    }
    Ok(names)
}

#[pyclass(name = "Feedforward", subclass)]
pub struct PyFeedforward;

#[pyclass(name = "Passthrough", extends = PyFeedforward)]
#[derive(Clone)]
pub struct PyPassthrough;

#[pymethods]
impl PyPassthrough {
    #[new]
    fn new() -> (Self, PyFeedforward) {
        (Self, PyFeedforward)
    }

    #[getter]
    fn kind(&self) -> i32 {
        FF_PASSTHROUGH
    }

    fn to_params<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        NativePassthrough.to_params().to_pyarray(py)
    }
}

#[pyclass(name = "Feedback", subclass)]
pub struct PyFeedback;

#[pyclass(name = "MRPFeedbackLinearized", extends = PyFeedback)]
#[derive(Clone)]
pub struct PyMRPFeedbackLinearized {
    inner: NativeMRPFeedbackLinearized,
}

#[pymethods]
impl PyMRPFeedbackLinearized {
    #[new]
    fn new(py: Python<'_>, kp: &PyAny, kd: &PyAny) -> PyResult<(Self, PyFeedback)> {
        let kp_in = extract_gain_input(py, kp, "kp")?;
        let kd_in = extract_gain_input(py, kd, "kd")?;
        let kp_mat = to_gain_matrix(kp_in, "kp", true).map_err(PyValueError::new_err)?;
        let kd_mat = to_gain_matrix(kd_in, "kd", false).map_err(PyValueError::new_err)?;
        Ok((
            Self {
                inner: NativeMRPFeedbackLinearized {
                    kp: kp_mat,
                    kd: kd_mat,
                },
            },
            PyFeedback,
        ))
    }

    #[getter]
    fn kind(&self) -> i32 {
        FB_MRP_FEEDBACK_LINEARIZED
    }

    #[getter]
    fn kp<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        mat3_to_py(py, &self.inner.kp)
    }

    #[getter]
    fn kd<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        mat3_to_py(py, &self.inner.kd)
    }

    fn to_matrices<'py>(
        &self,
        py: Python<'py>,
    ) -> (&'py PyArray2<f64>, &'py PyArray2<f64>) {
        (mat3_to_py(py, &self.inner.kp), mat3_to_py(py, &self.inner.kd))
    }
}

#[pyclass(name = "Desaturation", subclass)]
pub struct PyDesaturation;

#[pyclass(name = "NoDesaturation", extends = PyDesaturation)]
#[derive(Clone)]
pub struct PyNoDesaturation;

#[pymethods]
impl PyNoDesaturation {
    #[new]
    fn new() -> (Self, PyDesaturation) {
        (Self, PyDesaturation)
    }

    #[getter]
    fn kind(&self) -> i32 {
        DESAT_NONE
    }

    fn to_params<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        NativeNoDesaturation.to_params().to_pyarray(py)
    }
}

#[pyclass(name = "HoganSchaubManagement", extends = PyDesaturation)]
#[derive(Clone)]
pub struct PyHoganSchaubManagement {
    inner: NativeHoganSchaubManagement,
}

#[pymethods]
impl PyHoganSchaubManagement {
    #[new]
    #[pyo3(signature = (gain, target_momentum=0.0, mtq_gain=0.0))]
    fn new(
        gain: f64,
        target_momentum: f64,
        mtq_gain: f64,
    ) -> PyResult<(Self, PyDesaturation)> {
        let inner = NativeHoganSchaubManagement::new(gain, target_momentum, mtq_gain)
            .map_err(PyValueError::new_err)?;
        Ok((Self { inner }, PyDesaturation))
    }

    #[getter]
    fn kind(&self) -> i32 {
        DESAT_HOGAN_SCHAUB
    }

    #[getter]
    fn gain(&self) -> f64 {
        self.inner.gain
    }

    #[getter]
    fn target_momentum(&self) -> f64 {
        self.inner.target_momentum
    }

    #[getter]
    fn mtq_gain(&self) -> f64 {
        self.inner.mtq_gain
    }

    fn to_params<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        self.inner.to_params().to_pyarray(py)
    }
}

#[pyclass(name = "Allocator", subclass)]
pub struct PyAllocator;

#[pyclass(name = "L2Allocator", extends = PyAllocator)]
#[derive(Clone)]
pub struct PyL2Allocator;

#[pymethods]
impl PyL2Allocator {
    #[new]
    fn new() -> (Self, PyAllocator) {
        (Self, PyAllocator)
    }

    #[getter]
    fn kind(&self) -> i32 {
        ALLOC_L2
    }
}

#[pyclass(name = "LInfinityAllocator", extends = PyAllocator)]
#[derive(Clone)]
pub struct PyLInfinityAllocator;

#[pymethods]
impl PyLInfinityAllocator {
    #[new]
    fn new() -> (Self, PyAllocator) {
        (Self, PyAllocator)
    }

    #[getter]
    fn kind(&self) -> i32 {
        ALLOC_LINFINITY
    }
}

#[pyclass(name = "Controller", subclass)]
pub struct PyController;

#[pyclass(name = "NoController", extends = PyController)]
#[derive(Clone)]
pub struct PyNoController;

#[pymethods]
impl PyNoController {
    #[new]
    pub(crate) fn new() -> (Self, PyController) {
        (Self, PyController)
    }

    #[getter]
    fn kind(&self) -> i32 {
        CTRL_NONE
    }

    fn validate_sensors(&self, _sensors: &PyAny) -> PyResult<()> {
        Ok(())
    }

    fn to_params<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        NativeNoController.to_params().to_pyarray(py)
    }
}

#[pyclass(name = "BDot", extends = PyController)]
#[derive(Clone)]
pub struct PyBDot {
    inner: NativeBDot,
}

#[pymethods]
impl PyBDot {
    #[new]
    fn new(gain: f64) -> PyResult<(Self, PyController)> {
        let inner = NativeBDot::new(gain).map_err(PyValueError::new_err)?;
        Ok((Self { inner }, PyController))
    }

    #[getter]
    fn kind(&self) -> i32 {
        CTRL_BDOT
    }

    #[getter]
    fn gain(&self) -> f64 {
        self.inner.gain
    }

    fn validate_sensors(&self, sensors: &PyAny) -> PyResult<()> {
        let names = extract_sensor_names(sensors)?;
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        self.inner
            .validate_sensors(&refs)
            .map_err(PyValueError::new_err)
    }

    fn to_params<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        self.inner.to_params().to_pyarray(py)
    }
}

#[pyclass(name = "BCross", extends = PyController)]
#[derive(Clone)]
pub struct PyBCross {
    inner: NativeBCross,
}

#[pymethods]
impl PyBCross {
    #[new]
    fn new(gain: f64) -> PyResult<(Self, PyController)> {
        let inner = NativeBCross::new(gain).map_err(PyValueError::new_err)?;
        Ok((Self { inner }, PyController))
    }

    #[getter]
    fn kind(&self) -> i32 {
        CTRL_BCROSS
    }

    #[getter]
    fn gain(&self) -> f64 {
        self.inner.gain
    }

    fn validate_sensors(&self, sensors: &PyAny) -> PyResult<()> {
        let names = extract_sensor_names(sensors)?;
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        self.inner
            .validate_sensors(&refs)
            .map_err(PyValueError::new_err)
    }

    fn to_params<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        self.inner.to_params().to_pyarray(py)
    }
}

#[pyclass(name = "AttitudeController", extends = PyController)]
pub struct PyAttitudeController {
    inner: NativeAttitudeController,
    feedback_py: Py<PyAny>,
    feedforward_py: Py<PyAny>,
    allocator_py: Py<PyAny>,
    desaturation_py: Py<PyAny>,
}

#[pymethods]
impl PyAttitudeController {
    #[new]
    #[pyo3(signature = (feedback, feedforward=None, allocator=None, desaturation=None))]
    fn new(
        py: Python<'_>,
        feedback: Py<PyAny>,
        feedforward: Option<Py<PyAny>>,
        allocator: Option<Py<PyAny>>,
        desaturation: Option<Py<PyAny>>,
    ) -> PyResult<(Self, PyController)> {
        let fb_arc: Arc<dyn NativeFeedbackTrait> =
            if let Ok(mrp) = feedback.extract::<PyRef<PyMRPFeedbackLinearized>>(py) {
                Arc::new(mrp.inner.clone())
            } else {
                return Err(PyTypeError::new_err("feedback must be a Feedback instance"));
            };

        let ff_py = match feedforward {
            Some(obj) => obj,
            None => Py::new(py, PyPassthrough::new())?.into_py(py),
        };
        let ff_arc: Arc<dyn NativeFeedforwardTrait> =
            if ff_py.extract::<PyRef<PyPassthrough>>(py).is_ok() {
                Arc::new(NativePassthrough)
            } else {
                return Err(PyTypeError::new_err(
                    "feedforward must be a Feedforward instance",
                ));
            };

        let alloc_py = match allocator {
            Some(obj) => obj,
            None => Py::new(py, PyL2Allocator::new())?.into_py(py),
        };
        let alloc_arc: Arc<dyn NativeAllocatorTrait> =
            if alloc_py.extract::<PyRef<PyL2Allocator>>(py).is_ok() {
                Arc::new(NativeL2Allocator)
            } else if alloc_py.extract::<PyRef<PyLInfinityAllocator>>(py).is_ok() {
                Arc::new(NativeLInfinityAllocator)
            } else {
                return Err(PyTypeError::new_err(
                    "allocator must be an Allocator instance",
                ));
            };

        let desat_py = match desaturation {
            Some(obj) => obj,
            None => Py::new(py, PyNoDesaturation::new())?.into_py(py),
        };
        let desat_arc: Arc<dyn NativeDesaturationTrait> =
            if desat_py.extract::<PyRef<PyNoDesaturation>>(py).is_ok() {
                Arc::new(NativeNoDesaturation)
            } else if let Ok(hs) = desat_py.extract::<PyRef<PyHoganSchaubManagement>>(py) {
                Arc::new(hs.inner)
            } else {
                return Err(PyTypeError::new_err(
                    "desaturation must be a Desaturation instance",
                ));
            };

        let inner = NativeAttitudeController::new(
            fb_arc,
            Some(ff_arc),
            Some(alloc_arc),
            Some(desat_arc),
        );
        Ok((
            Self {
                inner,
                feedback_py: feedback,
                feedforward_py: ff_py,
                allocator_py: alloc_py,
                desaturation_py: desat_py,
            },
            PyController,
        ))
    }

    #[getter]
    fn kind(&self) -> i32 {
        CTRL_ATTITUDE
    }

    #[getter]
    fn feedback(&self, py: Python<'_>) -> Py<PyAny> {
        self.feedback_py.clone_ref(py)
    }

    #[getter]
    fn feedforward(&self, py: Python<'_>) -> Py<PyAny> {
        self.feedforward_py.clone_ref(py)
    }

    #[getter]
    fn allocator(&self, py: Python<'_>) -> Py<PyAny> {
        self.allocator_py.clone_ref(py)
    }

    #[getter]
    fn desaturation(&self, py: Python<'_>) -> Py<PyAny> {
        self.desaturation_py.clone_ref(py)
    }

    fn validate_sensors(&self, _sensors: &PyAny) -> PyResult<()> {
        Ok(())
    }

    fn to_params<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        self.inner.to_params().to_pyarray(py)
    }
}

#[pyfunction]
fn passthrough_step<'py>(
    py: Python<'py>,
    q_goal: &PyAny,
    w_goal: &PyAny,
    q_current: &PyAny,
) -> PyResult<(
    &'py PyArray1<f64>,
    &'py PyArray1<f64>,
    &'py PyArray1<f64>,
    &'py PyArray1<f64>,
)> {
    let qg = extract_vec4(py, q_goal)?;
    let wg = extract_vec3(py, w_goal)?;
    let qc = extract_vec4(py, q_current)?;
    let (qd, wd, wdd, fft) = native_passthrough_step(qg, wg, qc);
    Ok((
        qd.to_pyarray(py),
        wd.to_pyarray(py),
        wdd.to_pyarray(py),
        fft.to_pyarray(py),
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn mrp_feedback_linearized_step<'py>(
    py: Python<'py>,
    q: &PyAny,
    omega: &PyAny,
    q_d: &PyAny,
    omega_d: &PyAny,
    omega_d_dot: &PyAny,
    ff_torque: &PyAny,
    wheel_momenta: &PyAny,
    total_inertia: &PyAny,
    bus_inertia: &PyAny,
    wheel_axes: &PyAny,
    kp: &PyAny,
    kd: &PyAny,
) -> PyResult<&'py PyArray1<f64>> {
    let q_v = extract_vec4(py, q)?;
    let w_v = extract_vec3(py, omega)?;
    let qd_v = extract_vec4(py, q_d)?;
    let wd_v = extract_vec3(py, omega_d)?;
    let wdd_v = extract_vec3(py, omega_d_dot)?;
    let fft_v = extract_vec3(py, ff_torque)?;
    let hm_arr = py
        .import("numpy")?
        .call_method1("asarray", (wheel_momenta, "float64"))?
        .extract::<PyReadonlyArray1<f64>>()?;
    let ti = extract_mat3(py, total_inertia)?;
    let bi = extract_mat3(py, bus_inertia)?;
    let wa_ro = py
        .import("numpy")?
        .call_method1("asarray", (wheel_axes, "float64"))?
        .extract::<PyReadonlyArray2<f64>>()?;
    let wa_arr = wa_ro.as_array();
    let mut wa = Vec::with_capacity(wa_ro.shape()[0]);
    for r in 0..wa_ro.shape()[0] {
        wa.push([wa_arr[(r, 0)], wa_arr[(r, 1)], wa_arr[(r, 2)]]);
    }
    let kp_m = extract_mat3(py, kp)?;
    let kd_m = extract_mat3(py, kd)?;
    let out = native_mrp_feedback_linearized_step(
        q_v,
        w_v,
        qd_v,
        wd_v,
        wdd_v,
        fft_v,
        hm_arr.as_slice()?,
        &ti,
        &bi,
        &wa,
        &kp_m,
        &kd_m,
    );
    Ok(out.to_pyarray(py))
}

#[pyfunction]
fn hogan_schaub_u0<'py>(
    py: Python<'py>,
    wheel_momenta: &PyAny,
    gain: f64,
    target_momentum: f64,
) -> PyResult<&'py PyArray1<f64>> {
    let hm = py
        .import("numpy")?
        .call_method1("asarray", (wheel_momenta, "float64"))?
        .extract::<PyReadonlyArray1<f64>>()?;
    Ok(native_hogan_schaub_u0(hm.as_slice()?, gain, target_momentum).to_pyarray(py))
}

#[pyfunction]
fn allocate_l2<'py>(
    py: Python<'py>,
    tau_c: &PyAny,
    a_pinv: &PyAny,
    null_proj: &PyAny,
    u_0: &PyAny,
    max_torque: &PyAny,
) -> PyResult<&'py PyArray1<f64>> {
    let tc = extract_vec3(py, tau_c)?;
    let ap = extract_dmatrix(py, a_pinv)?;
    let np_m = extract_dmatrix(py, null_proj)?;
    let u0_ro = py
        .import("numpy")?
        .call_method1("asarray", (u_0, "float64"))?
        .extract::<PyReadonlyArray1<f64>>()?;
    let mt_ro = py
        .import("numpy")?
        .call_method1("asarray", (max_torque, "float64"))?
        .extract::<PyReadonlyArray1<f64>>()?;
    Ok(native_allocate_l2(tc, &ap, &np_m, u0_ro.as_slice()?, mt_ro.as_slice()?).to_pyarray(py))
}

#[pyfunction]
fn allocate_linfinity<'py>(
    py: Python<'py>,
    tau_c: &PyAny,
    a_pinv: &PyAny,
    null_basis: &PyAny,
    null_proj: &PyAny,
    u_0: &PyAny,
    max_torque: &PyAny,
) -> PyResult<&'py PyArray1<f64>> {
    let tc = extract_vec3(py, tau_c)?;
    let ap = extract_dmatrix(py, a_pinv)?;
    let nb = extract_dmatrix(py, null_basis)?;
    let np_m = extract_dmatrix(py, null_proj)?;
    let u0_ro = py
        .import("numpy")?
        .call_method1("asarray", (u_0, "float64"))?
        .extract::<PyReadonlyArray1<f64>>()?;
    let mt_ro = py
        .import("numpy")?
        .call_method1("asarray", (max_torque, "float64"))?
        .extract::<PyReadonlyArray1<f64>>()?;
    Ok(
        native_allocate_linfinity(tc, &ap, &nb, &np_m, u0_ro.as_slice()?, mt_ro.as_slice()?)
            .to_pyarray(py),
    )
}

pub fn register(module: &PyModule) -> PyResult<()> {
    module.add_class::<PyFeedforward>()?;
    module.add_class::<PyPassthrough>()?;
    module.add_class::<PyFeedback>()?;
    module.add_class::<PyMRPFeedbackLinearized>()?;
    module.add_class::<PyDesaturation>()?;
    module.add_class::<PyNoDesaturation>()?;
    module.add_class::<PyHoganSchaubManagement>()?;
    module.add_class::<PyAllocator>()?;
    module.add_class::<PyL2Allocator>()?;
    module.add_class::<PyLInfinityAllocator>()?;
    module.add_class::<PyController>()?;
    module.add_class::<PyNoController>()?;
    module.add_class::<PyBDot>()?;
    module.add_class::<PyBCross>()?;
    module.add_class::<PyAttitudeController>()?;
    module.add_function(wrap_pyfunction!(passthrough_step, module)?)?;
    module.add_function(wrap_pyfunction!(mrp_feedback_linearized_step, module)?)?;
    module.add_function(wrap_pyfunction!(hogan_schaub_u0, module)?)?;
    module.add_function(wrap_pyfunction!(allocate_l2, module)?)?;
    module.add_function(wrap_pyfunction!(allocate_linfinity, module)?)?;
    Ok(())
}
