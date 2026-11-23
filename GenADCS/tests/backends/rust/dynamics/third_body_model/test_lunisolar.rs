use super::*;
#[test]
fn lunisolar_acceleration_is_plausible_and_rejects_collisions() { pyo3::prepare_freethreaded_python(); Python::with_gil(|py| {
    let p=vec![7e6,0.,0.].into_py(py);let s=vec![1.5e11,0.,0.].into_py(py);let m=vec![0.,3.8e8,0.].into_py(py);
    let out=lunisolar_acceleration(py,p.as_ref(py),s.as_ref(py),m.as_ref(py),1.3271244e20,4.9048695e12).unwrap();let a:Vec<f64>=out.as_ref(py).call_method0("tolist").unwrap().extract().unwrap();let norm=a.iter().map(|x|x*x).sum::<f64>().sqrt();assert!(norm>1e-8&&norm<1e-5);
    let z=vec![0.,0.,0.].into_py(py);assert!(lunisolar_acceleration(py,z.as_ref(py),z.as_ref(py),m.as_ref(py),1.,1.).is_err());assert!(lunisolar_acceleration(py,z.as_ref(py),s.as_ref(py),z.as_ref(py),1.,1.).is_err());
});}
