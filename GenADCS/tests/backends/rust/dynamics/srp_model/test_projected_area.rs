use super::*;
use pyo3::types::PyList;
#[test]
fn srp_force_torque_illumination_and_collision() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let p = vec![0., 0., 0.].into_py(py);
        let sun = vec![149597870700., 0., 0.].into_py(py);
        let q = vec![1., 0., 0., 0.].into_py(py);
        let n = PyList::new(py, [vec![1., 0., 0.]]);
        let a = vec![2.].into_py(py);
        let o = PyList::new(py, [vec![0., 0., 1.]]);
        let f = projected_area_srp_force(
            py,
            p.as_ref(py),
            sun.as_ref(py),
            q.as_ref(py),
            n.as_ref(),
            a.as_ref(py),
            o.as_ref(),
            1.,
        )
        .unwrap();
        let full: Vec<f64> = f
            .as_ref(py)
            .call_method0("tolist")
            .unwrap()
            .extract()
            .unwrap();
        assert!(full.iter().any(|v| v.abs() > 0.));
        let half = projected_area_srp_force(
            py,
            p.as_ref(py),
            sun.as_ref(py),
            q.as_ref(py),
            n.as_ref(),
            a.as_ref(py),
            o.as_ref(),
            0.5,
        )
        .unwrap();
        let h: Vec<f64> = half
            .as_ref(py)
            .call_method0("tolist")
            .unwrap()
            .extract()
            .unwrap();
        assert!((h[0] - 0.5 * full[0]).abs() < 1e-14);
        let arms = PyList::new(py, [vec![0., 1., 0.]]);
        assert!(projected_area_srp_torque(
            py,
            p.as_ref(py),
            sun.as_ref(py),
            q.as_ref(py),
            n.as_ref(),
            arms.as_ref(),
            a.as_ref(py),
            o.as_ref(),
            1.
        )
        .is_ok());
        assert!(projected_area_srp_force(
            py,
            p.as_ref(py),
            p.as_ref(py),
            q.as_ref(py),
            n.as_ref(),
            a.as_ref(py),
            o.as_ref(),
            1.
        )
        .is_err());
    });
}
