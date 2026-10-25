extends RefCounted
## Timeline lanes computed from the log, incrementally (a budget of rows per
## frame, and new rows as a live stream grows):
## - eclipse: sunlit / penumbra / umbra spans from shadow_factor;
## - passes: for each ground site of scene.json "timeline", when the spacecraft
##   is above the site's minimum elevation (same rule as
##   ADCS.viz.timeline.ground_passes: geodetic elevation, linear rise/set).

enum Light { SUNLIT, PENUMBRA, UMBRA }

const ROWS_PER_CHUNK := 500
## CPU time one update() may use [us]: a long log fills in over several frames without a hitch.
var budget_us := 2000
const SUNLIT_ABOVE := 0.999
const UMBRA_BELOW := 0.001

var store: RefCounted
## Closed spans [[t0, t1, Light], ...]; plus the open span, see eclipse_spans().
var _ecl: Array = []
var _ecl_open := [NAN, -1]  # [t0, kind]
var _ecl_k := 0
var sites: Array = []  ## scene "timeline" sites: {name, color, itrs_m, up_itrs, min_elev_deg}
var _site_state: Array = []  ## per site: {passes, k, in, rise, e_prev, e_max, t_max, partial}
var done := false  ## all rows so far processed


func setup(data_store: RefCounted, timeline_desc: Dictionary) -> void:
	store = data_store
	sites = timeline_desc.get("sites", [])
	_site_state.clear()
	for s: Dictionary in sites:
		_site_state.append({"passes": [], "k": 0, "in": false, "rise": 0.0, "e_prev": 0.0, "e_max": -1e9,
			"t_max": 0.0, "partial": false})


static func sky_of(sf: float) -> int:
	return Light.SUNLIT if sf >= SUNLIT_ABOVE else (Light.UMBRA if sf <= UMBRA_BELOW else Light.PENUMBRA)


## Process new rows in chunks for up to budget_us. Returns true if anything changed.
func update() -> bool:
	if store == null or store.n_rows == 0:
		return false
	var t0 := Time.get_ticks_usec()
	var changed := false
	while Time.get_ticks_usec() - t0 < budget_us:
		var step := _update_eclipse()
		for i in sites.size():
			step = _update_site(i) or step
		changed = changed or step
		if not step:
			break
	done = _ecl_k >= store.n_rows and _site_state.all(func(s: Dictionary) -> bool: return s["k"] >= store.n_rows)
	return changed


func _update_eclipse() -> bool:
	if not store.has("shadow_factor"):
		return false
	var sf: PackedFloat64Array = store.cols["shadow_factor"]
	var n: int = mini(store.n_rows, _ecl_k + ROWS_PER_CHUNK)
	if _ecl_k >= n:
		return false
	for k in range(_ecl_k, n):
		var kind := sky_of(sf[k])
		if _ecl_open[1] < 0:
			_ecl_open = [store.t(k), kind]
		elif kind != _ecl_open[1]:
			# Boundary: where shadow_factor crosses the threshold between the two states.
			var prev: int = _ecl_open[1]
			var thr := 0.5
			if (prev == Light.SUNLIT) != (kind == Light.SUNLIT) and (prev == Light.PENUMBRA or kind == Light.PENUMBRA):
				thr = SUNLIT_ABOVE
			elif prev == Light.PENUMBRA or kind == Light.PENUMBRA:
				thr = UMBRA_BELOW
			var a := sf[k - 1]
			var b := sf[k]
			var f := clampf((thr - a) / (b - a), 0.0, 1.0) if b != a else 0.5
			var tb: float = store.t(k - 1) + f * (store.t(k) - store.t(k - 1))
			_ecl.append([_ecl_open[0], tb, prev])
			_ecl_open = [tb, kind]
	_ecl_k = n
	return true


## Elevation of the spacecraft above the site at row k minus the site's threshold [deg].
func _elev(k: int, site: Dictionary) -> float:
	var r: PackedFloat64Array = store.cols["r_sc"]
	var q: PackedFloat64Array = store.cols["q_itrs_gcrs"]
	var x := r[3 * k] * 1000.0
	var y := r[3 * k + 1] * 1000.0
	var z := r[3 * k + 2] * 1000.0
	var qn := sqrt(q[4 * k] ** 2 + q[4 * k + 1] ** 2 + q[4 * k + 2] ** 2 + q[4 * k + 3] ** 2)
	var w := q[4 * k] / qn
	var a := q[4 * k + 1] / qn
	var b := q[4 * k + 2] / qn
	var c := q[4 * k + 3] / qn
	# ITRS position: R(q) r, R from the scalar-first Hamilton quaternion q_itrs_gcrs.
	var xi := (1 - 2 * (b * b + c * c)) * x + 2 * (a * b - w * c) * y + 2 * (a * c + w * b) * z
	var yi := 2 * (a * b + w * c) * x + (1 - 2 * (a * a + c * c)) * y + 2 * (b * c - w * a) * z
	var zi := 2 * (a * c - w * b) * x + 2 * (b * c + w * a) * y + (1 - 2 * (a * a + b * b)) * z
	var p: Array = site["itrs_m"]
	var u: Array = site["up_itrs"]
	var lx := xi - float(p[0])
	var ly := yi - float(p[1])
	var lz := zi - float(p[2])
	var s := (lx * float(u[0]) + ly * float(u[1]) + lz * float(u[2])) / sqrt(lx * lx + ly * ly + lz * lz)
	return rad_to_deg(asin(clampf(s, -1.0, 1.0))) - float(site.get("min_elev_deg", 10.0))


func _update_site(i: int) -> bool:
	if not store.has("r_sc") or not store.has("q_itrs_gcrs"):
		return false
	var st: Dictionary = _site_state[i]
	var site: Dictionary = sites[i]
	var n: int = mini(store.n_rows, int(st["k"]) + ROWS_PER_CHUNK)
	var k0: int = st["k"]
	if k0 >= n:
		return false
	var offset := float(site.get("min_elev_deg", 10.0))
	for k in range(k0, n):
		var e := _elev(k, site)
		var t: float = store.t(k)
		if e >= 0.0:
			if not st["in"]:
				st["in"] = true
				st["partial"] = k == 0
				st["rise"] = t if k == 0 else store.t(k - 1) + (t - store.t(k - 1)) * float(st["e_prev"]) / (float(st["e_prev"]) - e)
				st["e_max"] = e
				st["t_max"] = t
			elif e > float(st["e_max"]):
				st["e_max"] = e
				st["t_max"] = t
		elif st["in"]:
			var t_set: float = store.t(k - 1) + (t - store.t(k - 1)) * float(st["e_prev"]) / (float(st["e_prev"]) - e)
			(st["passes"] as Array).append([st["rise"], t_set, st["t_max"], float(st["e_max"]) + offset, st["partial"]])
			st["in"] = false
		st["e_prev"] = e
	st["k"] = n
	return true


## Eclipse spans so far, the last one ending at the last processed row.
func eclipse_spans() -> Array:
	if _ecl_open[1] < 0:
		return _ecl
	return _ecl + [[_ecl_open[0], store.t(_ecl_k - 1), _ecl_open[1]]]


## Passes of site i so far: [[t_rise, t_set, t_max, max_elev_deg, partial], ...]; an
## ongoing pass ends at the last processed row (partial).
func passes(i: int) -> Array:
	var st: Dictionary = _site_state[i]
	var out: Array = (st["passes"] as Array).duplicate()
	if st["in"]:
		out.append([st["rise"], store.t(int(st["k"]) - 1), st["t_max"], float(st["e_max"]) + float(sites[i].get("min_elev_deg", 10.0)), true])
	return out


## Times where something starts or ends (eclipse changes, pass rise/set), sorted.
func event_times() -> PackedFloat64Array:
	var out := PackedFloat64Array()
	for s: Array in _ecl:
		out.append(s[1])
	for i in sites.size():
		for p: Array in passes(i):
			out.append(p[0])
			out.append(p[1])
	out.sort()
	return out


## Sky state and the passes in progress at time t (for the hover card).
func at(t: float) -> Dictionary:
	var sky := -1
	for s: Array in eclipse_spans():
		if s[0] <= t and t <= s[1]:
			sky = s[2]
			break
	var over := []
	for i in sites.size():
		for p: Array in passes(i):
			if p[0] <= t and t <= p[1]:
				over.append(String(sites[i]["name"]))
	return {"sky": sky, "passes": over}
