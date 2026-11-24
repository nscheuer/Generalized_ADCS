use super::*;
use pyo3::types::PyList;
fn vec(py: Python<'_>, x: &[f64]) -> PyObject {
    x.to_vec().into_py(py)
}
#[test]
fn drag_force_torque_scaling_and_invalid_panel_counts() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let q = vec(py, &[1., 0., 0., 0.]);
        let v = vec(py, &[0., 0., 7500.]);
        let n = PyList::new(py, [vec(py, &[0., 0., 1.])]);
        let a = vec(py, &[2.]);
        let cd = vec(py, &[2.2]);
        let f = projected_area_drag_force(
            py,
            v.as_ref(py),
            q.as_ref(py),
            1e-11,
            n.as_ref(),
            a.as_ref(py),
            cd.as_ref(py),
        )
        .unwrap();
        let actual: Vec<f64> = f
            .as_ref(py)
            .call_method0("tolist")
            .unwrap()
            .extract()
            .unwrap();
        assert!(actual[2] < 0.);
        let twice = projected_area_drag_force(
            py,
            v.as_ref(py),
            q.as_ref(py),
            2e-11,
            n.as_ref(),
            a.as_ref(py),
            cd.as_ref(py),
        )
        .unwrap();
        let scaled: Vec<f64> = twice
            .as_ref(py)
            .call_method0("tolist")
            .unwrap()
            .extract()
            .unwrap();
        assert!((scaled[2] - 2. * actual[2]).abs() < 1e-14);
        let zero = projected_area_drag_force(
            py,
            v.as_ref(py),
            q.as_ref(py),
            0.,
            n.as_ref(),
            a.as_ref(py),
            cd.as_ref(py),
        )
        .unwrap();
        let no_force: Vec<f64> = zero
            .as_ref(py)
            .call_method0("tolist")
            .unwrap()
            .extract()
            .unwrap();
        assert!(no_force.iter().all(|x| *x == 0.));
        let t = projected_area_drag_torque(
            py,
            v.as_ref(py),
            q.as_ref(py),
            1.,
            n.as_ref(),
            PyList::new(py, [vec(py, &[1., 0., 0.])]).as_ref(),
            a.as_ref(py),
            cd.as_ref(py),
        )
        .unwrap();
        assert!(t
            .as_ref(py)
            .call_method0("tolist")
            .unwrap()
            .extract::<Vec<f64>>()
            .unwrap()
            .iter()
            .all(|x| x.is_finite()));
        assert!(projected_area_drag_force(
            py,
            v.as_ref(py),
            q.as_ref(py),
            1.,
            PyList::empty(py).as_ref(),
            a.as_ref(py),
            cd.as_ref(py)
        )
        .is_err());
    });
}
