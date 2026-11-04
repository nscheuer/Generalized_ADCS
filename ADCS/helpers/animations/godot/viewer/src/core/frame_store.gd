extends RefCounted
## Rows of one run as flat float64 columns, with time lookup and interpolation.
##
## Positions in the log are km; sample() returns metres.

const D := preload("res://src/core/dmath.gd")
const CHUNK_ROWS := 1024

var names: PackedStringArray
var widths: Dictionary = {}
## Completed chunks are immutable packed arrays; the mutable Array tail avoids
## copy-on-write of all prior rows on each streamed append.
var chunks: Dictionary = {}  # name -> Array[PackedFloat64Array]
var tails: Dictionary = {}  # name -> Array (float values for the current chunk)
var n_rows := 0


func _init(channel_names: PackedStringArray, channel_widths: PackedInt32Array) -> void:
	names = channel_names
	for i in names.size():
		widths[names[i]] = channel_widths[i]
		chunks[names[i]] = []
		tails[names[i]] = []


## Bulk input follows the same chunked path as live rows, avoiding copies of
## previously stored history.
func append_block(block_cols: Dictionary, rows: int) -> void:
	for row in range(rows):
		var values := PackedFloat64Array()
		for name: String in names:
			var w: int = widths[name]
			var start := row * w
			var col: PackedFloat64Array = block_cols[name]
			for j in w:
				values.append(col[start + j])
		append_row(values)


func append_row(values: PackedFloat64Array) -> void:
	var off := 0
	for name: String in names:
		var w: int = widths[name]
		var tail: Array = tails[name]
		for j in w:
			tail.append(values[off + j])
		if tail.size() == CHUNK_ROWS * w:
			(chunks[name] as Array).append(PackedFloat64Array(tail))
			tails[name] = []
		off += w
	n_rows += 1


func has(name: String) -> bool:
	return widths.has(name)


func get_row(name: String, k: int) -> PackedFloat64Array:
	var w: int = widths[name]
	var chunk_index := int(k / CHUNK_ROWS)
	var offset := (k % CHUNK_ROWS) * w
	if chunk_index < (chunks[name] as Array).size():
		var col: PackedFloat64Array = chunks[name][chunk_index]
		return col.slice(offset, offset + w)
	var tail: Array = tails[name]
	var out := PackedFloat64Array()
	for j in w:
		out.append(tail[offset + j])
	return out


func t(k: int) -> float:
	return get_row("t_s", k)[0]


func t_min() -> float:
	return t(0) if n_rows > 0 else 0.0


func t_max() -> float:
	return t(n_rows - 1) if n_rows > 0 else 0.0


## Index k and fraction s with t(k) <= ts <= t(k + 1), clamped to the data.
func locate(ts: float) -> Array:
	if n_rows < 2:
		return [0, 0.0]
	var lo := 0
	var hi := n_rows
	while lo < hi:
		var mid := int((lo + hi) / 2)
		if t(mid) <= ts:
			lo = mid + 1
		else:
			hi = mid
	var k := clampi(lo - 1, 0, maxi(n_rows - 2, 0))
	var h := t(k + 1) - t(k)
	return [k, clampf((ts - t(k)) / h, 0.0, 1.0)]


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
	if has("q_sun_gcrs"):
		out["q_sun_gcrs"] = D.slerp(get_row("q_sun_gcrs", k), get_row("q_sun_gcrs", k1), s)
	if has("q_moon_gcrs"):
		out["q_moon_gcrs"] = D.slerp(get_row("q_moon_gcrs", k), get_row("q_moon_gcrs", k1), s)
	for body: String in ["mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"]:
		if has("r_" + body):
			out["r_" + body] = D.scale(value("r_" + body, ts), 1000.0)
			out["q_" + body + "_gcrs"] = D.slerp(get_row("q_" + body + "_gcrs", k), get_row("q_" + body + "_gcrs", k1), s)
	for opt: String in ["wheel_speed", "h_wheel", "u_cmd", "pointing_err"]:
		if has(opt):
			out[opt] = value(opt, ts)
	return out


## After the log ends, keep returning the exact final frame. The time cursor
## continues, but the rendered simulation state does not advance.
func sample_extended(ts: float) -> Dictionary:
	var end_t := t_max()
	var frame := sample(minf(ts, end_t))
	frame["after_simulation"] = ts > end_t
	return frame
