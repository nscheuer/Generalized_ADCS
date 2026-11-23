use super::*;
fn near(a: &[f64], b: &[f64]) { assert!(a.iter().zip(b).all(|(x,y)| (x-y).abs()<1e-10), "{a:?} != {b:?}"); }
fn same(a: [f64;4], b: [f64;4]) { assert!(dot(&a,&b).abs()>1.0-1e-10); }
#[test] fn identity_conventions_and_algebra() { Python::with_gil(|py| {
 let id=Quaternion::identity(); near(&id.q,&[1.,0.,0.,0.]); assert!((id.norm()-1.).abs()<1e-12);
 let j=Quaternion::new(py,Some(vec![0.,0.,-1.,0.].into_py(py).as_ref(py)),"JPL").unwrap(); near(&j.q,&[0.,0.,0.,1.]);
 near(&qconj([1.,2.,3.,4.]),&[1.,-2.,-3.,-4.]); near(&qinv([2.,0.,0.,0.]).unwrap(),&[0.5,0.,0.,0.]);
 near(&qmul([1.,0.,0.,0.],[0.,1.,0.,0.]),&[0.,1.,0.,0.]);
 assert!(qnormalize([0.;4]).is_err()); assert!(qinv([0.;4]).is_err());
 assert!(Quaternion::new(py,None,"invalid").is_err());
 let bad=vec![1.,f64::NAN,0.,0.].into_py(py); assert!(Quaternion::new(py,Some(bad.as_ref(py)),"Hamilton").is_err());
 }); }
#[test] fn rotation_construction_roundtrips_and_invalid_inputs() { Python::with_gil(|py| {
 let axis=vec![0.,0.,2.].into_py(py); let a=Quaternion::from_axis_angle(py,axis.as_ref(py),std::f64::consts::FRAC_PI_2).unwrap();
 let rv=vec![0.,0.,std::f64::consts::FRAC_PI_2].into_py(py); let b=Quaternion::from_rotation_vector(py,rv.as_ref(py)).unwrap(); same(a.q,b.q);
 let x=vec![1.,0.,0.].into_py(py); let y=a.rotate(py,x.as_ref(py)).unwrap(); near(&fixed::<3>(py,y.as_ref(py),"y").unwrap(),&[0.,1.,0.]);
 let back=a.body_to_inertial(py,y.as_ref(py)).unwrap(); near(&fixed::<3>(py,back.as_ref(py),"back").unwrap(),&[1.,0.,0.]);
 let inv=a.inverse().unwrap(); same(qmul(inv.q,a.q),[1.,0.,0.,0.]); same(a.relative_to(py,Quaternion::identity().into_py(py).as_ref(py)).unwrap().q,a.q);
 let m=a.to_dcm(py).unwrap(); same(Quaternion::from_dcm(py,m.as_ref(py)).unwrap().q,a.q);
 assert!(Quaternion::from_axis_angle(py,vec![0.,0.,0.].into_py(py).as_ref(py),1.).is_err());
 assert!(Quaternion::from_dcm(py,vec![vec![-1.,0.,0.],vec![0.,1.,0.],vec![0.,0.,1.]].into_py(py).as_ref(py)).is_err());
 }); }
#[test] fn interpolation_shortest_path_extrapolation_and_small_angles() { Python::with_gil(|py| {
 let id=vec![1.,0.,0.,0.].into_py(py); let axis=vec![0.,0.,1.].into_py(py);
 let end=_axis_angle_to_quaternion(py,axis.as_ref(py),std::f64::consts::FRAC_PI_2).unwrap();
 for t in [-0.5,0.,0.25,0.5,0.75,1.,1.5] { let actual=slerp(py,id.as_ref(py),end.as_ref(py),t).unwrap(); let expected=_axis_angle_to_quaternion(py,axis.as_ref(py),t*std::f64::consts::FRAC_PI_2).unwrap(); same(fixed::<4>(py,actual.as_ref(py),"q").unwrap(),fixed::<4>(py,expected.as_ref(py),"q").unwrap()); }
 let neg=vec![-1.,0.,0.,0.].into_py(py); let mid=slerp(py,id.as_ref(py),neg.as_ref(py),0.5).unwrap(); near(&fixed::<4>(py,mid.as_ref(py),"q").unwrap(),&[1.,0.,0.,0.]);
 let near_end=vec![1.,1e-8,0.,0.].into_py(py); let mid=slerp(py,id.as_ref(py),near_end.as_ref(py),0.5).unwrap(); near(&fixed::<4>(py,mid.as_ref(py),"q").unwrap(),&[1.,0.5e-8,0.,0.]);
 let zero=vec![0.;4].into_py(py); assert!(slerp(py,id.as_ref(py),zero.as_ref(py),0.5).is_err());
 for vector in [[0.,0.,0.],[1e-14,0.,0.],[0.2,-0.3,0.4]] { let input=vector.to_vec().into_py(py); let q=rotation_vector_to_quaternion(py,input.as_ref(py)).unwrap(); let out=quaternion_to_rotation_vector(py,q.as_ref(py)).unwrap(); near(&fixed::<3>(py,out.as_ref(py),"rv").unwrap(),&vector); }
 }); }
