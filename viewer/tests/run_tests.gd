extends SceneTree
## Headless unit tests for the viewer core.
##
##   godot --headless --path viewer --script res://tests/run_tests.gd -- [--fixture DIR]
##
## --fixture: an adcs-vizlog written by testing/test_viz/test_viewer_godot.py
## (a 7000 km circular orbit with known values). Prints "ALL PASSED" and exits 0
## on success.

const D := preload("res://src/core/dmath.gd")
const FrameStore := preload("res://src/core/frame_store.gd")
const Clock := preload("res://src/core/viz_clock.gd")
const Origin := preload("res://src/core/origin.gd")
const Protocol := preload("res://src/io/protocol.gd")
const FileSource := preload("res://src/io/file_source.gd")
const World := preload("res://src/scene/world.gd")

var failures := PackedStringArray()
var checks := 0


func _init() -> void:
	var args := {}
	var argv := OS.get_cmdline_user_args()
	for i in argv.size():
		if argv[i] == "--fixture" and i + 1 < argv.size():
			args["fixture"] = argv[i + 1]
	for t: Callable in [test_rot_mat, test_look_rot, test_slerp, test_hermite, test_frame_store, test_clock,
			test_origin, test_protocol_frames, test_moon_frame]:
		t.call()
	if args.has("fixture"):
		test_file_source(args["fixture"])
	if failures.is_empty():
		print("ALL PASSED (%d checks)" % checks)
		quit(0)
	else:
		for f in failures:
			printerr("FAIL: ", f)
		quit(1)


func expect(cond: bool, what: String) -> void:
	checks += 1
	if not cond:
		failures.append(what)


func near(a: float, b: float, tol: float, what: String) -> void:
	expect(absf(a - b) <= tol, "%s: %s vs %s (tol %s)" % [what, a, b, tol])


# ------------------------------------------------------------------ dmath
func test_rot_mat() -> void:
	var m := D.rot_mat(PackedFloat64Array([1, 0, 0, 0]))
	for i in 9:
		near(m[i], 1.0 if i in [0, 4, 8] else 0.0, 1e-15, "identity quaternion [%d]" % i)
	# 90 deg about +z maps x -> y (active, scalar-first), as ADCS.helpers.math_helpers.rot_mat.
	var q := PackedFloat64Array([cos(PI / 4), 0, 0, sin(PI / 4)])
	var v := D.mat_vec(D.rot_mat(q), D.v3(1, 0, 0))
	near(v[0], 0.0, 1e-15, "rotz90 x")
	near(v[1], 1.0, 1e-15, "rotz90 y")
	# GCRS -> Godot mapping is a proper rotation: z up becomes Godot +Y.
	expect(D.to_godot(D.v3(0, 0, 1)).is_equal_approx(Vector3.UP), "GCRS +z is Godot up")
	expect(is_equal_approx(D.basis_from(D.rot_mat(q)).determinant(), 1.0), "basis_from keeps handedness")


func test_look_rot() -> void:
	var r := D.look_rot(D.v3(1, 2, 3), D.v3(0, 0, 1))
	var back := PackedFloat64Array([r[2], r[5], r[8]])
	var f := D.unit(D.v3(1, 2, 3))
	near(D.dot(back, f), -1.0, 1e-12, "camera looks along -Z")
	near(D.dot(PackedFloat64Array([r[0], r[3], r[6]]), PackedFloat64Array([r[1], r[4], r[7]])), 0.0, 1e-12,
		"orthogonal axes")


func test_slerp() -> void:
	var q0 := PackedFloat64Array([1, 0, 0, 0])
	var q1 := PackedFloat64Array([cos(PI / 4), 0, 0, sin(PI / 4)])
	var h := D.slerp(q0, q1, 0.5)
	near(h[0], cos(PI / 8), 1e-12, "slerp halfway w")
	var neg := PackedFloat64Array([-q1[0], -q1[1], -q1[2], -q1[3]])
	var h2 := D.slerp(q0, neg, 0.5)
	near(absf(h2[0]), cos(PI / 8), 1e-12, "slerp takes the short arc with -q")


func test_hermite() -> void:
	var a := 7000.0e3
	var n := sqrt(3.986004418e14 / pow(a, 3))
	var h := 30.0
	var r0 := D.v3(a, 0, 0)
	var v0 := D.v3(0, a * n, 0)
	var r1 := D.v3(a * cos(n * h), a * sin(n * h), 0)
	var v1 := D.v3(-a * n * sin(n * h), a * n * cos(n * h), 0)
	var mid := D.hermite(r0, v0, r1, v1, h, 0.5)
	var exact := D.v3(a * cos(n * h / 2), a * sin(n * h / 2), 0)
	expect(D.norm(D.sub(mid, exact)) < 0.05, "Hermite within 5 cm at 30 s spacing (%f m)" % D.norm(D.sub(mid, exact)))


# ------------------------------------------------------------------ frame store, clock, origin
func test_frame_store() -> void:
	var fs := FrameStore.new(PackedStringArray(["t_s", "x"]), PackedInt32Array([1, 2]))
	fs.append_block({"t_s": PackedFloat64Array([0.0, 1.0]), "x": PackedFloat64Array([0, 0, 10, 20])}, 2)
	fs.append_row(PackedFloat64Array([2.0, 20, 40]))
	expect(fs.n_rows == 3, "rows appended (packed arrays written back)")
	var v := fs.value("x", 1.5)
	near(v[0], 15.0, 1e-12, "linear interpolation")
	near(fs.value("x", 99.0)[1], 40.0, 1e-12, "clamped past the end")
	var ks: Array = fs.locate(1.0)
	expect(ks[0] == 0 and is_equal_approx(ks[1], 1.0) or ks[0] == 1 and is_equal_approx(ks[1], 0.0), "locate at a sample")


func test_clock() -> void:
	var c := Clock.new()
	c.set_range(0.0, 10.0)
	c.rate = 2.0
	c.advance(1.0)
	near(c.t, 2.0, 1e-12, "advance x rate")
	c.advance(100.0)
	expect(c.t == 10.0 and c.paused, "stops at the end")
	c.toggle_pause()
	expect(c.t == 0.0 and not c.paused, "play again from the start")
	c.seek(-5.0)
	expect(c.t == 0.0, "seek clamps")
	c.faster()
	expect(c.rate > 2.0, "faster")


func test_origin() -> void:
	var o := Origin.new()
	expect(o.begin_frame(D.v3(0, 0, 0), 1.0, 0.5), "craft 0.5 m away is near")
	near(o.near, 0.1, 1e-12, "near = 0.2 x surface distance")
	near(o.far / o.near, Origin.FRUSTUM_RATIO, 1e-6, "far/near ratio")
	expect(not o.begin_frame(D.v3(0, 0, 0), 1.0e6, 1.0), "craft 1000 km away is in scaled space")
	var last := 0.0
	for d: float in [1.0e3, 1.0e4, 2.0e4, 1.0e6, 7.0e6, 4.0e8, 1.0e10]:
		var drawn := o.far_scale(d) * d
		expect(drawn > last and drawn < o.far, "far_scale is monotonic and inside the frustum at %s m" % d)
		last = drawn


func test_moon_frame() -> void:
	var r := World.moon_rotation(D.v3(3.8e8, 1.0e7, -2.0e7))
	var x := PackedFloat64Array([r[0], r[3], r[6]])
	near(D.dot(x, D.unit(D.v3(-3.8e8, -1.0e7, 2.0e7))), 1.0, 1e-12, "Moon +X faces Earth")


# ------------------------------------------------------------------ protocol
func _frame(kind: int, payload: PackedByteArray, seq: int) -> PackedByteArray:
	var h := StreamPeerBuffer.new()
	h.big_endian = false
	h.put_data("ADVZ".to_ascii_buffer())
	h.put_u16(1)
	h.put_u16(kind)
	h.put_u32(payload.size())
	h.put_u64(seq)
	var out := h.data_array
	out.append_array(payload)
	return out


func test_protocol_frames() -> void:
	var c := Protocol.new()
	var hello := {"protocol": 1, "mode": "playback", "scene": {},
		"manifest": {"channels": [{"name": "t_s", "width": 1}, {"name": "r", "width": 3}], "runs": [{"id": "0"}]}}
	var buf := _frame(Protocol.Msg.HELLO, JSON.stringify(hello).to_utf8_buffer(), 0)
	var row := StreamPeerBuffer.new()
	row.big_endian = false
	row.put_u16(1)
	row.put_data("0".to_utf8_buffer())
	for x: float in [5.0, 1.0, 2.0, 3.0]:
		row.put_double(x)
	buf.append_array(_frame(Protocol.Msg.ROW, row.data_array, 1))
	var asset := StreamPeerBuffer.new()
	asset.big_endian = false
	asset.put_u16(5)
	asset.put_data("a.glb".to_utf8_buffer())
	var meta := JSON.stringify({"kind": "model_lod"}).to_utf8_buffer()
	asset.put_u32(meta.size())
	asset.put_data(meta)
	asset.put_data(PackedByteArray([1, 2, 3]))
	buf.append_array(_frame(Protocol.Msg.ASSET, asset.data_array, 2))
	buf.append_array(_frame(Protocol.Msg.END, JSON.stringify({}).to_utf8_buffer(), 3))
	# Deliver everything in one read, half of the last frame first (partial frame handling).
	c._buf = buf.slice(0, buf.size() - 3)
	var msgs := c.poll_messages()
	expect(msgs.size() == 3, "partial final frame waits (got %d)" % msgs.size())
	expect(c.names == PackedStringArray(["t_s", "r"]), "HELLO sets the channel layout")
	var decoded: Array = c.decode_row(msgs[1][2])
	expect(decoded[0] == "0" and (decoded[1] as PackedFloat64Array) == PackedFloat64Array([5, 1, 2, 3]), "ROW decodes")
	expect(c.assets.has("a.glb") and (c.assets["a.glb"]["data"] as PackedByteArray) == PackedByteArray([1, 2, 3]),
		"ASSET stored")
	c._buf.append_array(buf.slice(buf.size() - 3))
	c.poll_messages()
	expect(c.session_ended, "END {} ends the session")


# ------------------------------------------------------------------ file source
func test_file_source(path: String) -> void:
	var fs := FileSource.new()
	expect(fs.open(path), "fixture vizlog opens: %s" % fs.error_message)
	if fs.store == null:
		return
	expect(fs.store.n_rows == 121, "fixture rows (%d)" % fs.store.n_rows)
	# The fixture is a 7000 km circular orbit: |r| at any (interpolated) time is 7000 km.
	for ts: float in [0.0, 37.3, 119.9]:
		var f := fs.store.sample(ts)
		near(D.norm(f["r_sc"]), 7.0e6, 0.05, "Hermite radius at t=%s" % ts)
	var q: PackedFloat64Array = fs.store.sample(60.0)["q_itrs_gcrs"]
	near(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3], 1.0, 1e-12, "unit Earth orientation")
