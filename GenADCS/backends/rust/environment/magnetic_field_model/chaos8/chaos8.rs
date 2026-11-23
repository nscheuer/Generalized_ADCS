#[pyclass(extends=MagneticFieldModel)]
struct CHAOS8 { core: Coefficients, static_field: Coefficients, include_static: bool }

#[pymethods]
impl CHAOS8 {
    #[new]
    #[pyo3(signature=(include_static=true))]
    fn new(include_static:bool)->PyResult<(Self,MagneticFieldModel)> {
        let core=parse_shc(include_str!("../../../../../kernels/environment/magnetic_field_model/chaos8/CHAOS-8.7_core.shc"),1,20)?;
        let static_field=parse_shc(include_str!("../../../../../kernels/environment/magnetic_field_model/chaos8/CHAOS-8.7_static.shc"),21,185)?;
        Ok((Self{core,static_field,include_static},MagneticFieldModel))
    }
    #[getter] fn include_static(&self)->bool{self.include_static}
    #[setter] fn set_include_static(&self,_value:bool)->PyResult<()>{Err(pyo3::exceptions::PyAttributeError::new_err("include_static is read-only after construction"))}
    fn field_inertial_tesla(&self,py:Python<'_>,position_inertial_m:&PyAny,epoch_jd:f64)->PyResult<PyObject>{
        let year=2000.+(epoch_jd-2451544.5)/365.25;
        if year<self.core.epochs[0]||year>self.core.epochs[self.core.epochs.len()-1]{return Err(PyValueError::new_err("Epoch outside CHAOS-8.7 core coefficient range"));}
        let upper=self.core.epochs.partition_point(|x|*x<=year);let mut start=upper.saturating_sub(1);start=(start/5)*5;start=start.min(self.core.epochs.len()-6);
        let nmax=if self.include_static{185}else{20};let side=nmax+1;let mut g=vec![0.;side*side];let mut h=g.clone();
        if self.include_static { let(gs,hs)=padded(&self.static_field,0,nmax);g=gs;h=hs; }
        for j in 0..6 { let mut weight=1.;for k in 0..6 {if j!=k{weight*=(year-self.core.epochs[start+k])/(self.core.epochs[start+j]-self.core.epochs[start+k]);}}
            let(gj,hj)=padded(&self.core,start+j,nmax);for n in 1..=20{for m in 0..=n{let ix=n*side+m;g[ix]+=weight*gj[ix];h[ix]+=weight*hj[ix];}}
        }
        Ok(py_vec3(py,evaluate(vec3_from_py(py,position_inertial_m)?,epoch_jd,&g,&h,nmax)?))
    }
    fn field(&self,py:Python<'_>,position_inertial_m:&PyAny,epoch_jd:f64)->PyResult<PyObject>{self.field_inertial_tesla(py,position_inertial_m,epoch_jd)}
}
