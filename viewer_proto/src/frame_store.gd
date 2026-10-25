extends RefCounted
## Received rows of one run, as flat float64 columns, with interpolation.

const D := preload("res://src/dmath.gd")

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


## Interpolated state at time `ts` (seconds since epoch), clamped to the data.
## Positions are km in the log; returned in metres.
func sample(ts: float) -> Dictionary:
	var tcol: PackedFloat64Array = cols["t_s"]
	var k := tcol.bsearch(ts, true) - 1
	k = clampi(k, 0, n_rows - 2)
	var h := tcol[k + 1] - tcol[k]
	var s := clampf((ts - tcol[k]) / h, 0.0, 1.0)
	var r := D.hermite(get_row("r_sc", k), get_row("v_sc", k), get_row("r_sc", k + 1), get_row("v_sc", k + 1), h, s)
	var out := {
		"r_sc": D.scale(r, 1000.0),
		"q_gcrs_body": D.slerp(get_row("q_gcrs_body", k), get_row("q_gcrs_body", k + 1), s),
		"q_itrs_gcrs": D.slerp(get_row("q_itrs_gcrs", k), get_row("q_itrs_gcrs", k + 1), s),
		"r_sun": D.scale(_lerp3("r_sun", k, s), 1000.0),
		"shadow_factor": lerpf(get_row("shadow_factor", k)[0], get_row("shadow_factor", k + 1)[0], s),
	}
	if has("r_moon"):
		out["r_moon"] = D.scale(_lerp3("r_moon", k, s), 1000.0)
	return out


func _lerp3(name: String, k: int, s: float) -> PackedFloat64Array:
	var a := get_row(name, k)
	var b := get_row(name, k + 1)
	return PackedFloat64Array([a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1]), a[2] + s * (b[2] - a[2])])
