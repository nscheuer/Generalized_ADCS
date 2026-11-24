#[pyfunction]
#[pyo3(signature=(year,month,day,hour=0,minute=0,second=0.0))]
fn calendar_to_jd(year: i32, month: i32, day: i32, hour: i32, minute: i32, second: f64) -> f64{
    calendar_to_jd_native(year, month, day, hour, minute, second)
}

#[pyfunction]
fn jd_to_calendar(jd: f64) -> (i32, i32, i32, i32, i32, f64){
    jd_to_calendar_native(jd)
}

#[pyfunction]
fn utc_jd_to_tai_jd(jd: f64) -> PyResult<f64>{
    utc_jd_to_tai_jd_native(jd).map_err(PyValueError::new_err)
}

#[pyfunction]
fn tai_jd_to_utc_jd(jd: f64) -> PyResult<f64>{
    tai_jd_to_utc_jd_native(jd).map_err(PyValueError::new_err)
}

#[pyfunction]
fn tai_jd_to_tt_jd(jd: f64) -> f64{
    tai_jd_to_tt_jd_native(jd)
}

#[pyfunction]
fn tt_jd_to_tai_jd(jd: f64) -> f64{
    tt_jd_to_tai_jd_native(jd)
}

#[pyfunction]
fn utc_jd_to_tt_jd(jd: f64) -> PyResult<f64>{
    utc_jd_to_tt_jd_native(jd).map_err(PyValueError::new_err)
}

#[pyfunction]
fn tt_jd_to_utc_jd(jd: f64) -> PyResult<f64>{
    tt_jd_to_utc_jd_native(jd).map_err(PyValueError::new_err)
}

#[pyfunction]
fn utc_jd_to_ut1_jd(jd: f64, dut1_seconds: f64) -> f64{
    utc_jd_to_ut1_jd_native(jd, dut1_seconds)
}

#[pyfunction]
fn ut1_jd_to_utc_jd(jd: f64, dut1_seconds: f64) -> f64{
    ut1_jd_to_utc_jd_native(jd, dut1_seconds)
}

#[pyfunction]
fn tt_jd_to_tdb_jd(jd: f64) -> f64{
    tt_jd_to_tdb_jd_native(jd)
}

#[pyfunction]
fn tdb_jd_to_tt_jd(jd: f64) -> f64{
    tdb_jd_to_tt_jd_native(jd)
}

#[pyfunction]
fn utc_jd_to_tdb_jd(jd: f64) -> PyResult<f64>{
    utc_jd_to_tdb_jd_native(jd).map_err(PyValueError::new_err)
}

#[pyfunction]
fn tdb_jd_to_utc_jd(jd: f64) -> PyResult<f64>{
    tdb_jd_to_utc_jd_native(jd).map_err(PyValueError::new_err)
}

#[pyfunction]
fn tai_jd_to_ut1_jd(jd: f64, d: f64) -> PyResult<f64>{
    tai_jd_to_ut1_jd_native(jd, d).map_err(PyValueError::new_err)
}

#[pyfunction]
fn ut1_jd_to_tai_jd(jd: f64, d: f64) -> PyResult<f64>{
    ut1_jd_to_tai_jd_native(jd, d).map_err(PyValueError::new_err)
}

#[pyfunction]
fn tt_jd_to_ut1_jd(jd: f64, d: f64) -> PyResult<f64>{
    tt_jd_to_ut1_jd_native(jd, d).map_err(PyValueError::new_err)
}

#[pyfunction]
fn ut1_jd_to_tt_jd(jd: f64, d: f64) -> PyResult<f64>{
    ut1_jd_to_tt_jd_native(jd, d).map_err(PyValueError::new_err)
}

#[pyfunction]
fn tai_jd_to_tdb_jd(jd: f64) -> PyResult<f64>{
    tai_jd_to_tdb_jd_native(jd).map_err(PyValueError::new_err)
}

#[pyfunction]
fn tdb_jd_to_tai_jd(jd: f64) -> f64{
    tdb_jd_to_tai_jd_native(jd)
}

#[pyfunction]
fn ut1_jd_to_tdb_jd(jd: f64, d: f64) -> PyResult<f64>{
    ut1_jd_to_tdb_jd_native(jd, d).map_err(PyValueError::new_err)
}

#[pyfunction]
fn tdb_jd_to_ut1_jd(jd: f64, d: f64) -> PyResult<f64>{
    tdb_jd_to_ut1_jd_native(jd, d).map_err(PyValueError::new_err)
}

#[pyfunction]
fn jd_to_utc_calendar_fields(jd: f64) -> (i32, i32, i32, f64){
    jd_to_utc_calendar_fields_native(jd)
}

#[pyfunction]
fn day_of_year(year: i32, month: i32, day: i32) -> i32{
    day_of_year_native(year, month, day)
}

