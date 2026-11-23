use super::*;
#[test] fn kernel_roundtrip_antipodal_and_singularity(){Python::with_gil(|py|{
 for p in [[0.,0.,0.],[0.1,-0.2,0.3],[1.,0.,0.]]{let arg=p.to_vec().into_py(py);let q=rodrigues_to_quaternion(py,arg.as_ref(py)).unwrap();let out=quaternion_to_rodrigues(py,q.as_ref(py)).unwrap();let r=fixed::<3>(py,out.as_ref(py),"r").unwrap();for i in 0..3{assert!((r[i]-p[i]).abs()<1e-12);}}
 let half=vec![0.,1.,0.,0.].into_py(py);assert!(quaternion_to_rodrigues(py,half.as_ref(py)).is_err());
 let q=vec![0.8,0.6,0.,0.].into_py(py);let neg=vec![-0.8,-0.6,0.,0.].into_py(py);let a=quaternion_to_rodrigues(py,q.as_ref(py)).unwrap();let b=quaternion_to_rodrigues(py,neg.as_ref(py)).unwrap();assert_eq!(fixed::<3>(py,a.as_ref(py),"a").unwrap(),fixed::<3>(py,b.as_ref(py),"b").unwrap());
 });}
#[test] fn class_constructors_rotation_and_validation(){Python::with_gil(|py|{
 let id=Rodrigues::identity();assert_eq!(id.p,[0.;3]);let axis=vec![0.,0.,1.].into_py(py);
 let p=Rodrigues::from_axis_angle(py,axis.as_ref(py),std::f64::consts::FRAC_PI_2).unwrap();
 let x=vec![1.,0.,0.].into_py(py);let y=p.rotate(py,x.as_ref(py)).unwrap();assert!((fixed::<3>(py,y.as_ref(py),"y").unwrap()[1]-1.).abs()<1e-12);
 let back=p.body_to_inertial(py,y.as_ref(py)).unwrap();assert!((fixed::<3>(py,back.as_ref(py),"x").unwrap()[0]-1.).abs()<1e-12);
 let q=p.to_quaternion(py).unwrap();let from_q=Rodrigues::from_quaternion(py,q.into_py(py).as_ref(py)).unwrap();assert!((from_q.p[2]-p.p[2]).abs()<1e-12);
 let m=p.to_dcm(py).unwrap();let from_m=Rodrigues::from_dcm(py,pymat(py,&m.m).as_ref(py)).unwrap();assert!((from_m.p[2]-p.p[2]).abs()<1e-12);
 let inverse=p.inverse(py).unwrap();let result=p.multiply(py,inverse.into_py(py).as_ref(py)).unwrap();assert!(result.norm()<1e-12);
 assert!(Rodrigues::from_axis_angle(py,axis.as_ref(py),std::f64::consts::PI).is_err());
 let bad=vec![f64::NAN,0.,0.].into_py(py);assert!(Rodrigues::new(py,Some(bad.as_ref(py))).is_err());
 });}
