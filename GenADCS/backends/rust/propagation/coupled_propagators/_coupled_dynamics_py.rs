use super::{derivative_vec, project_wheel_momentum, Runtime};
use numpy::ToPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(super) fn numbers(py: Python<'_>, value: &PyAny) -> PyResult<Vec<f64>> {
    let values: Vec<f64> = py
        .import("numpy")?
        .call_method1("asarray", (value,))?
        .call_method1("astype", ("float64",))?
        .call_method0("tolist")?
        .extract()?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("values must be finite"));
    }
    Ok(values)
}

fn matrix3(py: Python<'_>, value: &PyAny) -> PyResult<[[f64; 3]; 3]> {
    let rows: Vec<Vec<f64>> = py
        .import("numpy")?
        .call_method1("asarray", (value,))?
        .call_method1("astype", ("float64",))?
        .call_method0("tolist")?
        .extract()?;
    if rows.len() != 3
        || rows
            .iter()
            .any(|row| row.len() != 3 || row.iter().any(|value| !value.is_finite()))
    {
        return Err(PyValueError::new_err("matrix must be a finite 3x3 matrix"));
    }
    Ok([
        rows[0].clone().try_into().unwrap(),
        rows[1].clone().try_into().unwrap(),
        rows[2].clone().try_into().unwrap(),
    ])
}

fn rows3(py: Python<'_>, value: &PyAny) -> PyResult<Vec<[f64; 3]>> {
    let rows: Vec<Vec<f64>> = py
        .import("numpy")?
        .call_method1("asarray", (value,))?
        .call_method1("astype", ("float64",))?
        .call_method0("tolist")?
        .extract()?;
    rows.into_iter()
        .map(|row| {
            row.try_into()
                .map_err(|_| PyValueError::new_err("rows must contain three values"))
        })
        .collect()
}

pub(super) fn py_array(py: Python<'_>, values: &[f64]) -> PyObject {
    values.to_pyarray(py).into_py(py)
}

impl Runtime {
    pub(super) fn from_py(py: Python<'_>, value: &PyAny) -> PyResult<Self> {
        let normals = rows3(py, value.getattr("normals")?)?;
        let arms = rows3(py, value.getattr("arms")?)?;
        let areas = numbers(py, value.getattr("areas")?)?;
        let drag_coefficients = numbers(py, value.getattr("drag_coefficients")?)?;
        let optical_coefficients = rows3(py, value.getattr("optical_coefficients")?)?;
        let dipole_values = numbers(py, value.getattr("dipole")?)?;
        let dipole: [f64; 3] = dipole_values
            .try_into()
            .map_err(|_| PyValueError::new_err("dipole must be a finite 3-vector"))?;
        let srp_force = value.getattr("srp_force")?.extract()?;
        let srp_torque = value.getattr("srp_torque")?.extract()?;
        let third_body = value.getattr("third_body")?.extract()?;
        let residual_dipole: bool = value.getattr("residual_dipole")?.extract()?;

        let ephemeris = if srp_force || srp_torque || third_body {
            let path: String = match value.getattr("ephemeris_path") {
                Ok(path) => path.extract()?,
                Err(_) => {
                    let package_file: String =
                        py.import("GenADCS")?.getattr("__file__")?.extract()?;
                    std::path::Path::new(&package_file)
                        .parent()
                        .ok_or_else(|| {
                            PyValueError::new_err("cannot locate GenADCS package directory")
                        })?
                        .join("kernels/environment/ephemeris/de421/de421.bsp")
                        .to_string_lossy()
                        .into_owned()
                }
            };
            Some(
                ephemeris_native::NativeEphemeris::from_path(std::path::Path::new(&path))
                    .map_err(PyValueError::new_err)?,
            )
        } else {
            None
        };

        let mtq_axes = rows3(py, value.getattr("mtq_axes")?)?;
        let mtq_max_dipole = numbers(py, value.getattr("mtq_max_dipole")?)?;
        let dipole_commands = numbers(py, value.getattr("dipole_commands")?)?;
        let magnetic = if residual_dipole || !dipole_commands.is_empty() {
            let kind: i32 = value.getattr("magnetic_kind")?.extract()?;
            Some(match kind {
                1 => {
                    magnetic_native::NativeMagneticField::igrf14().map_err(PyValueError::new_err)?
                }
                2 => magnetic_native::NativeMagneticField::chaos8(
                    value.getattr("magnetic_include_static")?.extract()?,
                )
                .map_err(PyValueError::new_err)?,
                _ => return Err(PyValueError::new_err("unsupported magnetic model")),
            })
        } else {
            None
        };

        let wheel_axes = rows3(py, value.getattr("wheel_axes")?)?;
        let wheel_inertias = numbers(py, value.getattr("wheel_inertias")?)?;
        let inverse_bus_inertia = matrix3(py, value.getattr("inverse_bus_inertia")?)?;
        let wheel_commands = numbers(py, value.getattr("wheel_commands")?)?;
        let wheel_max_momentum = numbers(py, value.getattr("wheel_max_momentum")?)?;

        let runtime = Self {
            mu: value.getattr("mu")?.extract()?,
            radius: value.getattr("radius")?.extract()?,
            mass: value.getattr("mass")?.extract()?,
            inertia: matrix3(py, value.getattr("inertia")?)?,
            inverse_inertia: matrix3(py, value.getattr("inverse_inertia")?)?,
            zonals: numbers(py, value.getattr("zonals")?)?,
            normals,
            arms,
            areas,
            drag_coefficients,
            optical_coefficients,
            dipole,
            drag_force: value.getattr("drag_force")?.extract()?,
            drag_torque: value.getattr("drag_torque")?.extract()?,
            srp_force,
            srp_torque,
            gravity_gradient: value.getattr("gravity_gradient")?.extract()?,
            third_body,
            ephemeris,
            magnetic,
            wheel_axes,
            wheel_inertias,
            inverse_bus_inertia,
            wheel_commands,
            wheel_max_momentum,
            mtq_axes,
            mtq_max_dipole,
            dipole_commands,
        };
        runtime.validate().map_err(PyValueError::new_err)?;
        Ok(runtime)
    }
}

#[pyfunction]
pub(super) fn derivative(
    py: Python<'_>,
    y: &PyAny,
    jd: f64,
    runtime: &PyAny,
) -> PyResult<PyObject> {
    let y = numbers(py, y)?;
    let runtime = Runtime::from_py(py, runtime)?;
    let result = derivative_vec(&y, jd, &runtime).map_err(PyValueError::new_err)?;
    Ok(py_array(py, &result))
}

#[pyfunction]
pub(super) fn project(py: Python<'_>, y: &PyAny, runtime: &PyAny) -> PyResult<PyObject> {
    let mut y = numbers(py, y)?;
    let runtime = Runtime::from_py(py, runtime)?;
    project_wheel_momentum(&mut y, &runtime).map_err(PyValueError::new_err)?;
    Ok(py_array(py, &y))
}
