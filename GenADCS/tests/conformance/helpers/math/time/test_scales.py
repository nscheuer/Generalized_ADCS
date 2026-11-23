import pytest

SEC=86400.

def test_tai_tt_fixed_offset(time_api):
    tai=2459000.123456;tt=time_api.tai_jd_to_tt_jd(tai)
    assert (tt-tai)*SEC==pytest.approx(32.184,abs=2e-5)
    assert time_api.tt_jd_to_tai_jd(tt)==pytest.approx(tai,abs=2e-10)

def test_utc_tai_leap_step_contract(time_api):
    epochs=[2441317.5,2441499.5,2441683.5,2442048.5,2442413.5,2442778.5,2443144.5,2443509.5,2443874.5,2444239.5,2444786.5,2445151.5,2445516.5,2446247.5,2447161.5,2447892.5,2448257.5,2448804.5,2449169.5,2449534.5,2450083.5,2450630.5,2451179.5,2453736.5,2454832.5,2456109.5,2457204.5,2457754.5]
    for index,jd in enumerate(epochs):
        tai=time_api.utc_jd_to_tai_jd(jd)
        assert (tai-jd)*SEC==pytest.approx(10.+index,abs=2e-5)
        if index: assert (tai-time_api.utc_jd_to_tai_jd(jd-1e-6))*SEC==pytest.approx(1.0864,abs=2e-4)
    with pytest.raises(ValueError):time_api.utc_jd_to_tai_jd(2441317.5-1e-5)

def test_ut1_dut1_forward_inverse_and_all_scale_paths(time_api):
    utc=time_api.calendar_to_jd(2022,9,17,3,4,5.);dut1=-.137
    ut1=time_api.utc_jd_to_ut1_jd(utc,dut1)
    assert (ut1-utc)*SEC==pytest.approx(dut1,abs=2e-5)
    tai=time_api.utc_jd_to_tai_jd(utc);tt=time_api.utc_jd_to_tt_jd(utc);tdb=time_api.utc_jd_to_tdb_jd(utc)
    assert time_api.tai_jd_to_utc_jd(tai)==pytest.approx(utc,abs=2e-10)
    assert time_api.tt_jd_to_utc_jd(tt)==pytest.approx(utc,abs=2e-10)
    assert time_api.tdb_jd_to_utc_jd(tdb)==pytest.approx(utc,abs=3e-10)
    assert time_api.ut1_jd_to_utc_jd(ut1,dut1)==pytest.approx(utc,abs=2e-10)
    assert time_api.ut1_jd_to_tai_jd(ut1,dut1)==pytest.approx(tai,abs=2e-10)
    assert time_api.ut1_jd_to_tt_jd(ut1,dut1)==pytest.approx(tt,abs=3e-10)
    assert time_api.ut1_jd_to_tdb_jd(ut1,dut1)==pytest.approx(tdb,abs=4e-10)
    assert time_api.tdb_jd_to_ut1_jd(tdb,dut1)==pytest.approx(ut1,abs=4e-10)

def test_tdb_correction_stays_small_and_varies_smoothly(time_api):
    vals=[(time_api.tt_jd_to_tdb_jd(2451545.+i*365.25/99)- (2451545.+i*365.25/99))*SEC for i in range(100)]
    assert max(map(abs,vals))<.002;assert max(vals)-min(vals)>.002
    assert max(abs(a-b) for a,b in zip(vals,vals[1:]))<.001
