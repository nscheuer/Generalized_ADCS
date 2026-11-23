use super::*;
use pyo3::types::PyList;
#[test]
fn gravity_gradient_torque_and_bad_inputs() { pyo3::prepare_freethreaded_python(); Python::with_gil(|py| {
    let p=vec![7e6,7e6,0.].into_py(py);let q=vec![1.,0.,0.,0.].into_py(py);let i=PyList::new(py,[vec![2.,0.,0.],vec![0.,3.,0.],vec![0.,0.,4.]]);
    let t=static_gravity_gradient_torque(py,p.as_ref(py),q.as_ref(py),i.as_ref(),3.986e14).unwrap();let a:Vec<f64>=t.as_ref(py).call_method0("tolist").unwrap().extract().unwrap();assert!(a.iter().any(|v|v.abs()>0.));
    let zero=vec![0.,0.,0.].into_py(py);assert!(static_gravity_gradient_torque(py,zero.as_ref(py),q.as_ref(py),i.as_ref(),1.).is_err());
    let bad=vec![1.,2.].into_py(py);assert!(static_gravity_gradient_torque(py,p.as_ref(py),q.as_ref(py),bad.as_ref(py),1.).is_err());
});}
