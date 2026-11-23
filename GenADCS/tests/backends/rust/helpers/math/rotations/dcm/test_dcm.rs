use super::*;
fn near(a:M,b:M){assert!((a-b).norm()<1e-10,"{a:?} != {b:?}");}
#[test] fn construction_rotation_inverse_and_composition(){Python::with_gil(|py|{
 let axis=vec![0.,0.,2.].into_py(py); let a=DCM::from_axis_angle(py,axis.as_ref(py),std::f64::consts::FRAC_PI_2).unwrap();
 let rv=vec![0.,0.,std::f64::consts::FRAC_PI_2].into_py(py); let b=DCM::from_rotation_vector(py,rv.as_ref(py)).unwrap(); near(a.m,b.m);
 near(DCM::identity().m,M::identity()); assert!(a.norm()<1e-12); assert!(a.orthogonality_error()<1e-12);
 let x=vec![1.,0.,0.].into_py(py); let y=a.rotate(py,x.as_ref(py)).unwrap(); assert!((fixed::<3>(py,y.as_ref(py),"v").unwrap()[1]-1.).abs()<1e-12);
 let back=a.body_to_inertial(py,y.as_ref(py)).unwrap(); assert!((fixed::<3>(py,back.as_ref(py),"v").unwrap()[0]-1.).abs()<1e-12);
 near(a.inverse().m*a.m,M::identity()); near(a.conjugate().m,a.inverse().m); near(a.multiply(py,b.into_py(py).as_ref(py)).unwrap().m,a.m*a.m);
 near(a.relative_to(py,DCM::identity().into_py(py).as_ref(py)).unwrap().m,a.m);
 near(DCM::from_quaternion(py,a.to_quaternion(py).unwrap().into_py(py).as_ref(py)).unwrap().m,a.m);
 });}
#[test] fn normalization_and_validation(){Python::with_gil(|py|{
 let mut drift=M::identity();drift[(0,0)]+=1e-4;drift[(1,1)]-=1e-4;
 let repaired=normalize_dcm(py,pymat(py,&drift).as_ref(py)).unwrap(); let m=mat3(py,repaired.as_ref(py),"m").unwrap(); assert!((m.transpose()*m-M::identity()).norm()<1e-12);assert!((m.determinant()-1.).abs()<1e-12);
 let zero=M::zeros();assert!(normalize_dcm(py,pymat(py,&zero).as_ref(py)).is_err());
 let mut parallel=M::zeros();parallel.set_column(0,&nalgebra::Vector3::new(1.,0.,0.));parallel.set_column(1,&nalgebra::Vector3::new(2.,0.,0.));assert!(normalize_dcm(py,pymat(py,&parallel).as_ref(py)).is_err());
 let mut reflected=M::identity();reflected[(0,0)]=-1.;assert!(DCM::new(py,Some(pymat(py,&reflected).as_ref(py))).is_err());
 let bad=vec![0.,0.,0.].into_py(py);assert!(axis_angle_to_dcm(py,bad.as_ref(py),1.).is_err());
 });}
