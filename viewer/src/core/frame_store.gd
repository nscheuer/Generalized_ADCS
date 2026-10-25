extends RefCounted
## Rows of one run as flat float64 columns, with time lookup and interpolation.
##
## Positions in the log are km; sample() returns metres.

const D := preload("res://src/core/dmath.gd")

var names: PackedStringArray
var widths: Dictionary = {}
var cols: Dictionary = {}  # name -> PackedFloat64Array (row-major, n * width)
var n_rows := 0


func _init(channel_names: PackedStringArray, channel_widths: PackedInt32Array) -> void:
	names = channel_names
	for i in names.size():
		widths[names[i]] = channel_widths[i]
		cols[names[i]] = PackedFloat64Array()


## Packed arrays are values (copy-on-write) in GDScript: append to a local and
## store it back, or the dictionary entry never changes.
func append_block(block_cols: Dictionary, rows: int) -> void:
	for name: String in names:
		var col: PackedFloat64Array = cols[name]
		col.append_array(block_cols[name])
		cols[name] = col
	n_rows += rows


func append_row(values: PackedFloat64Array) -> void:
	var off := 0
	for name: String in names:
		var w: int = widths[name]
		var col: PackedFloat64Array = cols[name]
		col.append_array(values.slice(off, off + w))
		cols[name] = col
		off += w
	n_rows += 1


func has(name: String) -> bool:
	return cols.has(name)


func get_row(name: String, k: int) -> PackedFloat64Array:
	var w: int = widths[name]
	return (cols[name] as PackedFloat64Array).slice(k * w, (k + 1) * w)


func t(k: int) -> float:
	return (cols["t_s"] as PackedFloat64Array)[k]


func t_min() -> float:
	return t(0) if n_rows > 0 else 0.0


func t_max() -> float:
	return t(n_rows - 1) if n_rows > 0 else 0.0


## Index k and fraction s with t(k) <= ts <= t(k + 1), clamped to the data.
func locate(ts: float) -> Array:
	var tcol: PackedFloat64Array = cols["t_s"]
	var k := clampi(tcol.bsearch(ts, true) - 1, 0, maxi(n_rows - 2, 0))
	if n_rows < 2:
		return [0, 0.0]
	var h := tcol[k + 1] - tcol[k]
	return [k, clampf((ts - tcol[k]) / h, 0.0, 1.0)]


## Linear interpolation of any channel.
func value(name: String, ts: float) -> PackedFloat64Array:
	var ks: Array = locate(ts)
	var k: int = ks[0]
	var s: float = ks[1]
	var a := get_row(name, k)
	if n_rows < 2:
		return a
	var b := get_row(name, k + 1)
	var out := PackedFloat64Array()
	out.resize(a.size())
	for i in a.size():
		out[i] = a[i] + s * (b[i] - a[i])
	return out


## Interpolated state at time `ts` (seconds since epoch): Hermite position,
## slerped attitude and Earth orientation, linear Sun/Moon. Metres.
func sample(ts: float) -> Dictionary:
	var ks: Array = locate(ts)
	var k: int = ks[0]
	var s: float = ks[1]
	var k1 := mini(k + 1, n_rows - 1)
	var h := t(k1) - t(k)
	var r := D.hermite(get_row("r_sc", k), get_row("v_sc", k), get_row("r_sc", k1), get_row("v_sc", k1), h, s) \
		if h > 0.0 else get_row("r_sc", k)
	var out := {
		"t": ts,
		"k": k,
		"r_sc": D.scale(r, 1000.0),
		"v_sc": D.scale(value("v_sc", ts), 1000.0),
		"q_gcrs_body": D.slerp(get_row("q_gcrs_body", k), get_row("q_gcrs_body", k1), s),
		"w_body": value("w_body", ts),
		"q_itrs_gcrs": D.slerp(get_row("q_itrs_gcrs", k), get_row("q_itrs_gcrs", k1), s),
		"r_sun": D.scale(value("r_sun", ts), 1000.0),
		"shadow_factor": value("shadow_factor", ts)[0],
	}
	if has("r_moon"):
		out["r_moon"] = D.scale(value("r_moon", ts), 1000.0)
	for opt: String in ["wheel_speed", "h_wheel", "u_cmd", "pointing_err"]:
		if has(opt):
			out[opt] = value(opt, ts)
	return out
