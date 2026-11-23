#[pyclass(extends=MagneticFieldModel)]
struct IGRF14 { coefficients: Coefficients, epochs_jd: Vec<f64> }

#[pymethods]
impl IGRF14 {
    #[new]
    fn new() -> PyResult<(Self, MagneticFieldModel)> {
        let coefficients=parse_shc(include_str!("../../../../../kernels/environment/magnetic_field_model/igrf14/IGRF14.shc"),1,13)?;
        let epochs_jd=coefficients.epochs.iter().map(|year|calendar_jd(*year as i32,1,1)).collect();
        Ok((Self{coefficients,epochs_jd},MagneticFieldModel))
    }
    fn field_inertial_tesla(&self,py:Python<'_>,position_inertial_m:&PyAny,epoch_jd:f64)->PyResult<PyObject> {
        if epoch_jd<self.epochs_jd[0] || epoch_jd>self.epochs_jd[self.epochs_jd.len()-1] { return Err(PyValueError::new_err("IGRF-14 coefficients cover 1900-01-01 to 2030-01-01")); }
        let i=self.epochs_jd.partition_point(|x|*x<=epoch_jd).saturating_sub(1).min(self.epochs_jd.len()-2);
        let w=(epoch_jd-self.epochs_jd[i])/(self.epochs_jd[i+1]-self.epochs_jd[i]);
        let (mut g,mut h)=padded(&self.coefficients,i,13);let(g1,h1)=padded(&self.coefficients,i+1,13);
        for k in 0..g.len(){g[k]+=w*(g1[k]-g[k]);h[k]+=w*(h1[k]-h[k]);}
        Ok(py_vec3(py,evaluate(vec3_from_py(py,position_inertial_m)?,epoch_jd,&g,&h,13)?))
    }
    fn field(&self,py:Python<'_>,position_inertial_m:&PyAny,epoch_jd:f64)->PyResult<PyObject>{self.field_inertial_tesla(py,position_inertial_m,epoch_jd)}
}
