const RE_KM: f64 = 6371.2;
const PI: f64 = std::f64::consts::PI;

#[pyfunction]
fn _schmidt_table(py: Python<'_>, nmax: usize) -> PyResult<PyObject> {
    let side = nmax + 1;
    let mut values = vec![vec![0.0; side]; side];
    values[0][0] = 1.0;
    for n in 1..=nmax {
        values[n][0] = values[n - 1][0] * (2.0 * n as f64 - 1.0) / n as f64;
        for m in 1..=n {
            values[n][m] = values[n][m - 1]
                * (((n - m + 1) as f64 * if m == 1 { 2.0 } else { 1.0 })
                    / (n + m) as f64)
                    .sqrt();
        }
    }
    Ok(py.import("numpy")?.call_method1("asarray", (values,))?.into())
}

#[derive(Clone)]
struct Coefficients { epochs: Vec<f64>, nmax: usize, g: Vec<f64>, h: Vec<f64> }

impl Coefficients {
    fn idx(&self, epoch: usize, n: usize, m: usize) -> usize {
        (epoch * (self.nmax + 1) + n) * (self.nmax + 1) + m
    }
}

fn parse_shc(text: &str, expected_min: usize, expected_max: usize) -> PyResult<Coefficients> {
    let mut lines = text.lines().map(str::trim).filter(|s| !s.is_empty() && !s.starts_with('#'));
    let header: Vec<usize> = lines.next().ok_or_else(|| PyValueError::new_err("empty SHC file"))?
        .split_whitespace().take(5).map(|x| x.parse().map_err(|_| PyValueError::new_err("invalid SHC header"))).collect::<PyResult<_>>()?;
    if header.len() != 5 || header[0] != expected_min || header[1] != expected_max {
        return Err(PyValueError::new_err("unexpected SHC harmonic degree range"));
    }
    let (nmin, nmax, count) = (header[0], header[1], header[2]);
    let epochs: Vec<f64> = lines.next().ok_or_else(|| PyValueError::new_err("missing SHC epochs"))?
        .split_whitespace().map(|x| x.parse().map_err(|_| PyValueError::new_err("invalid SHC epoch"))).collect::<PyResult<_>>()?;
    if epochs.len() != count || epochs.windows(2).any(|w| w[1] <= w[0]) {
        return Err(PyValueError::new_err("invalid SHC epochs"));
    }
    let side = nmax + 1;
    let mut g = vec![0.; count * side * side];
    let mut h = vec![0.; count * side * side];
    let mut seen_g = vec![false; side * side];
    let mut seen_h = vec![false; side * side];
    let mut row_count = 0usize;
    for line in lines {
        let row: Vec<f64> = line.split_whitespace().map(|x| x.parse().map_err(|_| PyValueError::new_err("invalid SHC coefficient"))).collect::<PyResult<_>>()?;
        if row.len() != count + 2 || row[0].fract() != 0. || row[1].fract() != 0. { return Err(PyValueError::new_err("malformed SHC coefficient row")); }
        let n = row[0] as usize;
        let signed_m = row[1] as isize;
        let m = signed_m.unsigned_abs();
        if n < nmin || n > nmax || m > n { return Err(PyValueError::new_err("invalid SHC coefficient")); }
        let seen = if signed_m >= 0 { &mut seen_g } else { &mut seen_h };
        if seen[n * side + m] { return Err(PyValueError::new_err("duplicate SHC coefficient")); }
        seen[n * side + m] = true;
        row_count += 1;
        for (i, value) in row[2..].iter().enumerate() {
            let index = (i * side + n) * side + m;
            if signed_m >= 0 { g[index] = *value; } else { h[index] = *value; }
        }
    }
    let expected_rows = (nmax + 1).pow(2) - nmin.pow(2);
    if row_count != expected_rows { return Err(PyValueError::new_err("incomplete SHC coefficient set")); }
    for n in nmin..=nmax { for m in 0..=n {
        let i=n*side+m;
        if !seen_g[i] || (m>0 && !seen_h[i]) { return Err(PyValueError::new_err("incomplete SHC coefficient set")); }
    } }
    let mut schmidt = vec![0.; side * side];
    schmidt[0] = 1.;
    for n in 1..=nmax {
        schmidt[n * side] = schmidt[(n - 1) * side] * (2. * n as f64 - 1.) / n as f64;
        for m in 1..=n {
            schmidt[n * side + m] = schmidt[n * side + m - 1] * (((n - m + 1) as f64 * if m == 1 { 2. } else { 1. }) / (n + m) as f64).sqrt();
        }
    }
    for epoch in 0..count { for n in nmin..=nmax { for m in 0..=n {
        let ix = (epoch * side + n) * side + m;
        g[ix] *= schmidt[n * side + m]; h[ix] *= schmidt[n * side + m];
    } } }
    Ok(Coefficients { epochs, nmax, g, h })
}

fn calendar_jd(year: i32, month: i32, day: i32) -> f64 {
    let (mut y, mut m) = (year, month); if m <= 2 { y -= 1; m += 12; }
    let a = y.div_euclid(100); let b = 2 - a + a.div_euclid(4);
    (365.25 * (y + 4716) as f64).floor() + (30.6001 * (m + 1) as f64).floor() + day as f64 + b as f64 - 1524.5
}

fn tt_jd(utc: f64) -> PyResult<f64> {
    // UTC effective dates and TAI-UTC seconds, from 1972 through the present.
    const LEAPS: &[(f64, f64)] = &[
        (2441317.5,10.),(2441499.5,11.),(2441683.5,12.),(2442048.5,13.),(2442413.5,14.),(2442778.5,15.),
        (2443144.5,16.),(2443509.5,17.),(2443874.5,18.),(2444239.5,19.),(2444786.5,20.),(2445151.5,21.),
        (2445516.5,22.),(2446247.5,23.),(2447161.5,24.),(2447892.5,25.),(2448257.5,26.),
        (2448804.5,27.),(2449169.5,28.),(2449534.5,29.),(2450083.5,30.),(2450630.5,31.),
        (2451179.5,32.),(2453736.5,33.),(2454832.5,34.),(2456109.5,35.),(2457204.5,36.),(2457754.5,37.),
    ];
    if utc < LEAPS[0].0 { return Err(PyValueError::new_err("UTC/TT conversion is supported from 1972-01-01")); }
    let mut offset = 10.;
    for &(jd, seconds) in LEAPS { if utc >= jd { offset = seconds; } }
    Ok(utc + (offset + 32.184) / 86400.)
}

type Mat = [[f64; 3]; 3];
fn mm(a: Mat,b: Mat)->Mat { let mut o=[[0.;3];3]; for i in 0..3 { for j in 0..3 { for k in 0..3 { o[i][j]+=a[i][k]*b[k][j]; } } } o }
fn r1(a:f64)->Mat { let(c,s)=(a.cos(),a.sin()); [[1.,0.,0.],[0.,c,-s],[0.,s,c]] }
fn r2(a:f64)->Mat { let(c,s)=(a.cos(),a.sin()); [[c,0.,s],[0.,1.,0.],[-s,0.,c]] }
fn r3(a:f64)->Mat { let(c,s)=(a.cos(),a.sin()); [[c,-s,0.],[s,c,0.],[0.,0.,1.]] }
fn precession(jd:f64)->Mat {
    let t=(jd-2451545.)/36525.; let q=PI/(180.*3600.);
    let zeta=(2306.2181*t+0.30188*t*t+0.017998*t*t*t)*q;
    let z=(2306.2181*t+1.09468*t*t+0.018203*t*t*t)*q;
    let th=(2004.3109*t-0.42665*t*t-0.041833*t*t*t)*q;
    mm(mm(r3(-z),r2(th)),r3(-zeta))
}
fn gcrf_teme(jd:f64)->Mat {
    let t=(jd-2451545.)/36525.; let om=(125.04452-1934.136261*t).to_radians();
    let ms=(280.4665+36000.7698*t).to_radians(); let mn=(218.3165+481267.8813*t).to_radians();
    let q=PI/(180.*3600.);
    let dp=(-17.20*om.sin()-1.32*(2.*ms).sin()-0.23*(2.*mn).sin()+0.21*(2.*om).sin())*q;
    let de=(9.20*om.cos()+0.57*(2.*ms).cos()+0.10*(2.*mn).cos()-0.09*(2.*om).cos())*q;
    let eps=(23.43929111-0.013004167*t-1.64e-7*t*t+5.04e-7*t*t*t).to_radians();
    let nut=mm(mm(r1(-(eps+de)),r3(-dp)),r1(eps));
    mm(mm(r3(-dp*eps.cos()),nut),precession(jd))
}
fn gcrf_ecef(utc:f64,tt:f64)->Mat {
    let t=(utc-2451545.)/36525.; let sec=67310.54841+((876600.*3600.)+8640184.812866)*t+0.093104*t*t-6.2e-6*t*t*t;
    let angle=(sec/86400.-(sec/86400.).floor())*2.*PI;
    mm(r3(-angle),gcrf_teme(tt))
}

fn evaluate(position:[f64;3], jd:f64, g:&[f64], h:&[f64], nmax:usize)->PyResult<[f64;3]> {
    let tt=if jd>=2441317.5 { tt_jd(jd)? } else { jd };
    let rot=gcrf_ecef(jd,tt);
    let p=[(0..3).map(|j|rot[0][j]*position[j]).sum::<f64>(),(0..3).map(|j|rot[1][j]*position[j]).sum::<f64>(),(0..3).map(|j|rot[2][j]*position[j]).sum::<f64>()];
    let radius=p.iter().map(|x|x*x).sum::<f64>().sqrt();
    if radius<=0. { return Err(PyValueError::new_err("position must have positive radius")); }
    let theta=(p[2]/radius).clamp(-1.,1.).acos(); let phi=p[1].atan2(p[0]);
    let safe_theta=if theta==0. {1e-7} else if theta==PI {PI-1e-7} else {theta};
    let (st,ct)=(theta.sin(),theta.cos()); let (sp,cp)=(phi.sin(),phi.cos());
    let side=nmax+1; let mut p1=vec![0.;nmax+2]; let mut p2=p1.clone(); let mut pc=p1.clone();
    let mut d1=p1.clone(); let mut d2=p1.clone(); let mut dc=p1.clone();
    let mut cm=vec![0.;nmax+1];let mut sm=cm.clone();cm[0]=1.;
    for m in 1..=nmax { cm[m]=cm[m-1]*cp-sm[m-1]*sp; sm[m]=sm[m-1]*cp+cm[m-1]*sp; }
    p1[0]=1.; let ar=RE_KM/(radius/1000.);let mut arn=ar*ar;let(mut br,mut bt,mut bp)=(0.,0.,0.);
    for n in 1..=nmax { arn*=ar;
        for m in 0..=n {
            if n==m { pc[n]=safe_theta.sin()*p1[m-1]; dc[n]=safe_theta.sin()*d1[m-1]+safe_theta.cos()*p1[n-1]; }
            else if n==1 { pc[m]=safe_theta.cos()*p1[m];dc[m]=safe_theta.cos()*d1[m]-safe_theta.sin()*p1[m]; }
            else { let k=(((n-1)*(n-1)) as f64-(m*m) as f64)/((2*n-1)*(2*n-3)) as f64;pc[m]=safe_theta.cos()*p1[m]-k*p2[m];dc[m]=safe_theta.cos()*d1[m]-safe_theta.sin()*p1[m]-k*d2[m]; }
        }
        pc[n+1]=0.;dc[n+1]=0.;
        for m in 0..=n { let ix=n*side+m;let cc=g[ix]*cm[m]+h[ix]*sm[m];br+=arn*(n+1) as f64*cc*pc[m];bt-=arn*cc*dc[m];bp-=arn*m as f64*(h[ix]*cm[m]-g[ix]*sm[m])*pc[m]; }
        std::mem::swap(&mut p2,&mut p1);std::mem::swap(&mut p1,&mut pc);std::mem::swap(&mut d2,&mut d1);std::mem::swap(&mut d1,&mut dc);
    }
    bp/=safe_theta.sin();
    let fixed=[(br*st*cp+bt*ct*cp-bp*sp)*1e-9,(br*st*sp+bt*ct*sp+bp*cp)*1e-9,(br*ct-bt*st)*1e-9];
    Ok([(0..3).map(|i|rot[i][0]*fixed[i]).sum(),(0..3).map(|i|rot[i][1]*fixed[i]).sum(),(0..3).map(|i|rot[i][2]*fixed[i]).sum()])
}

fn padded(source:&Coefficients, epoch:usize, nmax:usize)->(Vec<f64>,Vec<f64>) {
    let side=nmax+1;let mut g=vec![0.;side*side];let mut h=vec![0.;side*side];
    for n in 0..=source.nmax {for m in 0..=n {let a=source.idx(epoch,n,m);let b=n*side+m;g[b]=source.g[a];h[b]=source.h[a];}}
    (g,h)
}

#[pyfunction]
fn read_shc(py:Python<'_>,path:&str,nmin:usize,nmax:usize)->PyResult<PyObject>{
    let contents=std::fs::read_to_string(path).map_err(|e|PyValueError::new_err(format!("could not read SHC file: {e}")))?;
    let c=parse_shc(&contents,nmin,nmax)?;
    let side=c.nmax+1;let count=c.epochs.len();
    let mut gs=Vec::with_capacity(count);let mut hs=Vec::with_capacity(count);
    for epoch in 0..count {let start=epoch*side*side;gs.push(c.g[start..start+side*side].to_vec());hs.push(c.h[start..start+side*side].to_vec());}
    let np=py.import("numpy")?;
    let times=np.call_method1("asarray",(c.epochs,))?;
    let g=np.call_method1("asarray",(gs,))?;
    let h=np.call_method1("asarray",(hs,))?;
    Ok((times,g,h).into_py(py))
}

fn vec3_from_py(py:Python<'_>, value:&PyAny)->PyResult<[f64;3]> {
    let vals:Vec<f64>=py.import("numpy")?.call_method1("asarray",(value,))?.call_method1("astype",("float64",))?.call_method0("tolist")?.extract()?;
    vals.try_into().map_err(|_|PyValueError::new_err("position must have shape (3,)"))
}
fn py_vec3(py:Python<'_>,v:[f64;3])->PyObject { v.to_pyarray(py).into_py(py) }
