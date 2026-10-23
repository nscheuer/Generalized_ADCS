extends RefCounted
## End-to-end checks run inside the real viewer (`--check NAME --config FILE --out DIR`).
## Each writes raw measurements to DIR/result.json; tools/viz_validation/run_viewer_checks.py
## judges them against values computed independently in Python.

const D := preload("res://src/core/dmath.gd")
const CameraRig := preload("res://src/camera/camera_rig.gd")
const R_EARTH := 6378137.0

var app  # app.gd (untyped: avoids a cyclic preload)
var _lin := PackedFloat32Array()


func _init(a: Variant) -> void:
	app = a
	for i in 256:
		_lin.append(Color(i / 255.0, 0, 0).srgb_to_linear().r)


static func finish(a: Node, result: Dictionary) -> void:
	var out_dir: String = a.opts.get("out", OS.get_user_data_dir())
	DirAccess.make_dir_recursive_absolute(out_dir)
	result["adapter"] = RenderingServer.get_video_adapter_name()
	result["godot"] = Engine.get_version_info()["string"]
	var f := FileAccess.open(out_dir.path_join("result.json"), FileAccess.WRITE)
	f.store_string(JSON.stringify(result))
	f.close()
	if result.has("error"):
		printerr("check error: ", result["error"])
	a.get_tree().quit(1 if result.has("error") else 0)


func _cfg() -> Dictionary:
	if not app.opts.has("config"):
		return {}
	var c: Variant = JSON.parse_string(FileAccess.get_file_as_string(app.opts["config"]))
	return c if c is Dictionary else {}


func _out(name: String) -> String:
	return String(app.opts.get("out", OS.get_user_data_dir())).path_join(name)


func _image() -> Image:
	var img: Image = app.world.viewport.get_texture().get_image()
	img.convert(Image.FORMAT_RGBA8)
	return img


func _frames(n: int) -> void:
	for i in n:
		await RenderingServer.frame_post_draw


func run(name: String) -> void:
	app._log("check %s: starting" % name)
	await app.get_tree().process_frame
	app._log("check %s: first frame" % name)
	var cfg := _cfg()
	var result: Dictionary
	match name:
		"assets": result = await assets_check()
		"shots": result = await shots(cfg)
		"jitter": result = await jitter(cfg)
		"terminator": result = await terminator()
		"landmarks": result = await landmarks(cfg)
		"scrub": result = await scrub(cfg)
		"capture": result = await capture(cfg)
		"light": result = await light_probe()
		"live": result = await live(cfg)
		"pose": result = await pose(cfg)
		"perf": result = await perf(cfg)
		"timeline": result = await timeline(cfg)
		"layers": result = await layers(cfg)
		_: result = {"error": "unknown check " + name}
	result["check"] = name
	result["source"] = "stream" if app.opts.has("connect") else "file"
	result["n_rows"] = app.source.store.n_rows
	result["model"] = app.world.craft.source
	result["model_lods"] = app.world.craft.lods.size()
	finish(app, result)


# ------------------------------------------------------------------ assets
func assets_check() -> Dictionary:
	await _frames(3)
	return {"loaded": app.loaded_assets, "load_ms": app.assets.load_ms, "craft_radius": app.world.craft.radius,
		"wheels": app.world.craft.wheel_nodes.size()}


# ------------------------------------------------------------------ screenshots
## shots: [{name, t, mode: orbit|earth, frame, yaw_deg, pitch_deg, dist}]
func shots(cfg: Dictionary) -> Dictionary:
	var out := []
	for s: Dictionary in cfg.get("shots", []):
		var rig: CameraRig = app.rig
		rig.mode = CameraRig.Mode.EARTH if s.get("mode", "orbit") == "earth" else CameraRig.Mode.ORBIT
		rig.frame_kind = s.get("frame", "lvlh")
		rig.yaw = deg_to_rad(float(s.get("yaw_deg", -140.0)))
		rig.pitch = deg_to_rad(float(s.get("pitch_deg", 18.0)))
		if rig.mode == CameraRig.Mode.EARTH:
			rig.earth_dist = float(s.get("dist", 4.0 * R_EARTH))
		else:
			rig.dist = float(s.get("dist", 4.0 * app.world.craft.radius))
		var t: float = s.get("t", app.clock.t_min)
		app.world.set_exposure(String(s.get("exposure_mode", "artistic")), float(s.get("ev", 0.0)))
		if s.get("sunward", false):
			# Angles relative to the Sun direction (inertial frame), so the lit side faces the camera.
			var f: Dictionary = app.source.store.sample(t)
			var c: PackedFloat64Array = f["r_sc"] if rig.mode == CameraRig.Mode.ORBIT else D.v3(0, 0, 0)
			var sd := D.unit(D.sub(f["r_sun"], c))
			rig.frame_kind = "inertial"
			rig.yaw = atan2(sd[1], sd[0]) + deg_to_rad(float(s.get("yaw_deg", 0.0)))
			rig.pitch = clampf(asin(sd[2]) + deg_to_rad(float(s.get("pitch_deg", 0.0))), -1.5, 1.5)
		app.render_at(t)
		await _frames(int(s.get("settle_frames", 3)))
		var img := _image()
		var file := "%s.png" % s.get("name", "shot_%d" % out.size())
		img.save_png(_out(file))
		out.append({"name": file, "t": t, "info": app.world.info.duplicate()})
	return {"shots": out}


# ------------------------------------------------------------------ jitter
## Camera locked to the body frame 1.5 craft radii out: the spacecraft must not
## move in the image while the frame time advances (7.5 km/s in GCRS).
func jitter(cfg: Dictionary) -> Dictionary:
	var frames: int = cfg.get("frames", 300)
	var fps: float = cfg.get("fps", 60.0)
	app.world.origin.absolute = bool(cfg.get("absolute", false))
	# Geometry only: constant flat lighting and no wheel animation, so any
	# pixel change comes from position error, not from the Sun moving in the
	# body frame while the controller slews.
	app.world.dynamic_lighting = false
	app.world.env.ambient_light_color = Color(0.8, 0.8, 0.8)
	app.world.overlays.set_enabled(false)  # vectors change with the data; the model alone must not move
	var rig: CameraRig = app.rig
	rig.mode = CameraRig.Mode.ORBIT
	rig.frame_kind = "body"
	rig.yaw = deg_to_rad(30.0)
	rig.pitch = deg_to_rad(20.0)
	rig.dist = 2.0 * app.world.craft.radius
	var t0: float = app.clock.t_min
	var ref: PackedByteArray
	var w := 0
	var h := 0
	var diffs := PackedFloat64Array()
	for i in frames + 10:
		app.render_at(t0 + maxf(0.0, (i - 10) / fps))
		await RenderingServer.frame_post_draw
		if i < 10:
			continue
		var img := _image()
		if ref.is_empty():
			ref = img.get_data()
			w = img.get_width()
			h = img.get_height()
			img.save_png(_out("jitter_first.png"))
			continue
		diffs.append(_roi_max_diff(ref, img.get_data(), w, h))
	return {"frames": frames, "absolute": app.world.origin.absolute, "max_diff": diffs}


## Mean absolute difference (0-255) over the central 20% x 20% of the image.
static func _roi_max_diff(a: PackedByteArray, b: PackedByteArray, w: int, h: int) -> float:
	var x0 := int(w * 0.4)
	var x1 := int(w * 0.6)
	var y0 := int(h * 0.4)
	var y1 := int(h * 0.6)
	var total := 0.0
	var n := 0
	for y in range(y0, y1):
		for x in range(x0, x1):
			var i := (y * w + x) * 4
			total += absf(a[i] - b[i]) + absf(a[i + 1] - b[i + 1]) + absf(a[i + 2] - b[i + 2])
			n += 3
	return total / n


# ------------------------------------------------------------------ terminator
func terminator() -> Dictionary:
	var f: Dictionary = app.render_at(app.clock.t_min)
	var r := D.rot_mat(f["q_itrs_gcrs"])
	var s_itrs := D.unit(D.mat_vec(r, f["r_sun"]))
	var subsolar := {"lon_deg": rad_to_deg(atan2(s_itrs[1], s_itrs[0])), "lat_deg": rad_to_deg(asin(s_itrs[2]))}
	var mat: ShaderMaterial = app.world.earth_mat
	for p: String in ["has_day", "has_night", "has_clouds", "has_normals", "has_ocean", "has_atmosphere"]:
		mat.set_shader_parameter(p, false)
	app.world.shell.visible = false  # geometry check: no haze or twilight reddening
	mat.set_shader_parameter("albedo", Vector3.ONE)
	app.world.craft.visible = false
	app.world.moon.visible = false
	app.world.env.tonemap_mode = Environment.TONE_MAPPER_LINEAR
	app.world.env.glow_enabled = false
	var sun_dir := D.unit(f["r_sun"])
	var view := D.unit(D.cross(sun_dir, D.v3(0, 0, 1)))
	var dist := 4.0 * R_EARTH
	var cam_rot := D.look_rot(D.scale(view, -1.0), D.v3(0, 0, 1))
	var cam := D.scale(view, dist)
	for i in 4:
		app.world.update(f, cam, cam_rot)
		app.world.moon.visible = false
		await RenderingServer.frame_post_draw
	var img := _image()
	img.save_png(_out("terminator.png"))
	var right := PackedFloat64Array([cam_rot[0], cam_rot[3], cam_rot[6]])
	var lit_right := D.dot(sun_dir, right) > 0.0
	var w := img.get_width()
	var h := img.get_height()
	var data := img.get_data()
	var offsets := PackedFloat64Array()
	for y in range(h / 2 - 20, h / 2 + 21):
		var xs := PackedFloat64Array()
		var ls := PackedFloat64Array()
		for k in range(-60, 61):
			var x := w / 2 + k
			var lin := _lin[data[(y * w + x) * 4]]
			if lin > 0.003 and lin < 0.2 and ((k > 0) == lit_right):
				xs.append(x + 0.5)
				ls.append(lin)
		if xs.size() >= 4:
			offsets.append(_zero_crossing(xs, ls) - w / 2.0)
	var focal: float = app.world.focal_px()
	return {"subsolar": subsolar, "offsets_px": offsets, "disk_radius_px": focal * tan(asin(R_EARTH / dist))}


static func _zero_crossing(xs: PackedFloat64Array, ls: PackedFloat64Array) -> float:
	var n := float(xs.size())
	var mx := 0.0
	var ml := 0.0
	for i in xs.size():
		mx += xs[i] / n
		ml += ls[i] / n
	var sxl := 0.0
	var sxx := 0.0
	for i in xs.size():
		sxl += (xs[i] - mx) * (ls[i] - ml)
		sxx += (xs[i] - mx) * (xs[i] - mx)
	return mx - ml / (sxl / sxx)


# ------------------------------------------------------------------ landmarks (through the cube maps)
## The day map comes from an asset cache built by Python from a texture with
## red dots at known places. The camera looks down at each place using GCRS
## directions computed in Python; the dot must be at the image centre.
func landmarks(cfg: Dictionary) -> Dictionary:
	var mat: ShaderMaterial = app.world.earth_mat
	mat.set_shader_parameter("ambient", 1.0)
	for p: String in ["has_night", "has_clouds", "has_normals", "has_ocean", "has_atmosphere"]:
		mat.set_shader_parameter(p, false)
	app.world.shell.visible = false
	app.world.craft.visible = false
	var f: Dictionary = app.render_at(app.clock.t_min)
	var dist := 3.0 * R_EARTH
	var out := []
	for lm: Dictionary in cfg.get("points", []):
		var dir := PackedFloat64Array(lm["dir_gcrs"])
		var up := D.v3(0, 0, 1) if absf(dir[2]) < 0.95 else D.v3(1, 0, 0)
		for i in 3:
			app.world.update(f, D.scale(dir, dist), D.look_rot(D.scale(dir, -1.0), up))
			await RenderingServer.frame_post_draw
		var img := _image()
		img.save_png(_out("landmark_%s.png" % lm["name"]))
		var c := _red_centroid(img)
		out.append({"name": lm["name"], "cx": c.x, "cy": c.y, "center": [img.get_width() / 2.0, img.get_height() / 2.0]})
	var focal: float = app.world.focal_px()
	return {"points": out, "focal_px": focal, "altitude_m": dist - R_EARTH, "has_day": app.loaded_assets.has("earth_day")}


func _red_centroid(img: Image) -> Vector2:
	var data := img.get_data()
	var w := img.get_width()
	var h := img.get_height()
	var sw := 0.0
	var sx := 0.0
	var sy := 0.0
	for y in range(h / 2 - 150, h / 2 + 150):
		for x in range(w / 2 - 150, w / 2 + 150):
			var i := (y * w + x) * 4
			var wgt := _lin[data[i]] - maxf(_lin[data[i + 1]], _lin[data[i + 2]])
			if wgt > 0.05:
				sw += wgt
				sx += wgt * (x + 0.5)
				sy += wgt * (y + 0.5)
	return Vector2(-1, -1) if sw <= 0.0 else Vector2(sx / sw, sy / sw)


# ------------------------------------------------------------------ scrubbing
## Jump to random times: CPU time to sample + pose, and wall time until the
## frame is on screen.
func scrub(cfg: Dictionary) -> Dictionary:
	var n: int = cfg.get("jumps", 200)
	var rng := RandomNumberGenerator.new()
	rng.seed = 7
	var cpu := PackedFloat64Array()
	var shown := PackedFloat64Array()
	await _frames(5)
	for i in n:
		var t := rng.randf_range(app.clock.t_min, app.clock.t_max)
		var t0 := Time.get_ticks_usec()
		app.render_at(t)
		var t1 := Time.get_ticks_usec()
		await RenderingServer.frame_post_draw
		var t2 := Time.get_ticks_usec()
		cpu.append((t1 - t0) / 1000.0)
		shown.append((t2 - t0) / 1000.0)
	return {"cpu_ms": cpu, "displayed_ms": shown, "duration_s": app.clock.t_max - app.clock.t_min}


# ------------------------------------------------------------------ deterministic capture
func capture(cfg: Dictionary) -> Dictionary:
	var frames: int = cfg.get("frames", 120)
	var fps: float = cfg.get("fps", 60.0)
	for i in frames:
		app.render_at(app.clock.t_min + i / fps * float(cfg.get("time_scale", 10.0)))
		await RenderingServer.frame_post_draw
	return {"frames": frames}


# ------------------------------------------------------------------ lighting diagnostics
## Spacecraft seen from the Sun's direction under lighting variants; centre brightness of each.
func light_probe() -> Dictionary:
	var rig: CameraRig = app.rig
	rig.mode = CameraRig.Mode.ORBIT
	var f: Dictionary = app.source.store.sample(app.clock.t_min)
	var sd := D.unit(D.sub(f["r_sun"], f["r_sc"]))
	rig.frame_kind = "inertial"
	rig.yaw = atan2(sd[1], sd[0])
	rig.pitch = asin(sd[2])
	rig.dist = 3.0 * app.world.craft.radius
	var out := {}
	var light: DirectionalLight3D = app.world.light
	for variant: String in ["normal", "no_shadows", "energy_x10", "ambient_bright"]:
		app.world.env.ambient_light_color = Color(0.03, 0.035, 0.045)
		app.render_at(app.clock.t_min)
		match variant:
			"no_shadows":
				light.shadow_enabled = false
			"energy_x10":
				light.shadow_enabled = false
				light.light_energy = 30.0
			"ambient_bright":
				app.world.env.ambient_light_color = Color(1, 1, 1)
		await _frames(3)
		var img := _image()
		img.save_png(_out("light_%s.png" % variant))
		var c := img.get_pixel(img.get_width() / 2, img.get_height() / 2)
		out[variant] = {"center": [c.r, c.g, c.b], "light_energy": light.light_energy, "light_visible": light.visible,
			"light_forward": -light.global_transform.basis.z, "shadow": light.shadow_enabled}
	var to_sun := D.to_godot(sd)
	return {"variants": out, "to_sun_godot": [to_sun.x, to_sun.y, to_sun.z], "shadow_factor": f["shadow_factor"]}


# ------------------------------------------------------------------ live session
## Follow a live session to its end, rendering every frame like the interactive viewer.
func live(cfg: Dictionary) -> Dictionary:
	var timeout_ms := int(float(cfg.get("timeout_s", 120.0)) * 1000.0)
	var deadline := Time.get_ticks_msec() + timeout_ms
	var frames := 0
	var lags := PackedFloat64Array()
	var first_rows: int = app.source.store.n_rows
	var last := Time.get_ticks_usec()
	while not app.source.finished and Time.get_ticks_msec() < deadline:
		var now := Time.get_ticks_usec()
		app._process((now - last) / 1.0e6)  # poll, follow the newest row, render, HUD
		last = now
		lags.append(app.clock.t_max - app.clock.t)
		frames += 1
		await RenderingServer.frame_post_draw
	app.source.poll()
	app.clock.set_range(app.source.store.t_min(), app.source.store.t_max())
	return {"live": app.source.live, "finished": app.source.finished, "rows": app.source.store.n_rows,
		"rows_at_start": first_rows, "frames": frames, "follow_lag_s": lags, "t_max": app.clock.t_max}


# ------------------------------------------------------------------ explicit camera poses
## shots: [{name, t, cam_pos (GCRS m), look_at (GCRS m), up (GCRS), fov_deg,
##          hide: ["craft", "model", "earth", "moon", "stars", "sun"], overlays: [ids] | absent (all), labels}]
## Renders exactly the requested view (e.g. from DSCOVR for an EPIC comparison).
## "model" hides the spacecraft's geometry but keeps its overlays.
func pose(cfg: Dictionary) -> Dictionary:
	var out := []
	var ov: RefCounted = app.world.overlays
	var w: RefCounted = app.world
	var parts: Array[Node3D] = [w.earth, w.shell, w.stars, w.sun]
	for c in w.craft.get_children():
		if c is Node3D and not ov.nodes.has(c):
			parts.append(c)
	var shown := parts.map(func(n: Node3D) -> bool: return n.visible)
	for s: Dictionary in cfg.get("shots", []):
		for i in parts.size():
			parts[i].visible = shown[i]
		var hide: Array = s.get("hide", [])
		app.world.craft.visible = not hide.has("craft")
		var keep: Array = s.get("overlays", [])
		for n: Node3D in ov.nodes:
			n.visible = keep.has(String(n.name)) if s.has("overlays") else bool(n.desc.get("visible", true))
		ov.show_labels = bool(s.get("labels", true))
		app.world.fov_deg = float(s.get("fov_deg", app.world.fov_deg))
		# Optional display overrides, e.g. linear tone mapping for photometric comparisons.
		var tm: String = s.get("tonemap", "agx")
		app.world.env.tonemap_mode = Environment.TONE_MAPPER_LINEAR if tm == "linear" else Environment.TONE_MAPPER_AGX
		app.world.env.glow_enabled = tm != "linear"
		app.world.set_exposure(String(s.get("exposure_mode", "artistic")), float(s.get("ev", 0.0)))
		var t: float = s.get("t", app.clock.t_min)
		var f: Dictionary = app.source.store.sample(t)
		var cam := PackedFloat64Array(s["cam_pos"])
		var target := PackedFloat64Array(s.get("look_at", [0.0, 0.0, 0.0]))
		var rot := D.look_rot(D.sub(target, cam), PackedFloat64Array(s.get("up", [0.0, 0.0, 1.0])))
		for i in int(s.get("settle_frames", 3)):
			app.world.update(f, cam, rot)
			app.world.moon.visible = app.world.moon.visible and not hide.has("moon")
			_hide_parts(hide)
			await RenderingServer.frame_post_draw
		var img := _image()
		var file := "%s.png" % s.get("name", "pose_%d" % out.size())
		img.save_png(_out(file))
		out.append({"name": file, "t": t, "fov_deg": app.world.fov_deg, "info": app.world.info.duplicate(),
			"overlays": ov.summary(), "hinges": w.craft.hinge_report()})
	return {"shots": out}


func _hide_parts(hide: Array) -> void:
	var w: RefCounted = app.world
	if hide.has("earth"):
		w.earth.visible = false
		w.shell.visible = false
	if hide.has("stars"):
		w.stars.visible = false
	if hide.has("sun"):
		w.sun.visible = false
	if hide.has("model"):
		for c in w.craft.get_children():
			if c is Node3D and not w.overlays.nodes.has(c):
				c.visible = false


# ------------------------------------------------------------------ performance
## Frame times with the full scene: orbit the spacecraft while zooming from
## metres to Earth-scale distances, vsync off.
func perf(cfg: Dictionary) -> Dictionary:
	var frames: int = cfg.get("frames", 600)
	DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
	for feature: String in cfg.get("disable", []):
		match feature:
			"atmosphere":
				app.world.earth_mat.set_shader_parameter("has_atmosphere", false)
				app.world.shell.visible = false
			"msaa":
				app.world.viewport.msaa_3d = Viewport.MSAA_DISABLED
			"glow":
				app.world.env.glow_enabled = false
			"clouds":
				app.world.earth_mat.set_shader_parameter("has_clouds", false)
			"overlays":
				app.world.overlays.set_enabled(false)
			"textures":
				for p: String in ["has_day", "has_night", "has_clouds", "has_normals", "has_ocean"]:
					app.world.earth_mat.set_shader_parameter(p, false)
	if cfg.get("fxaa", false):
		app.world.viewport.screen_space_aa = Viewport.SCREEN_SPACE_AA_FXAA
	for k: String in ["pixel_steps", "limb_band_lo", "limb_band_hi"]:
		if cfg.has(k):
			app.world.earth_mat.set_shader_parameter(k, cfg[k])
	if cfg.has("shell_steps"):
		app.world.shell_mat.set_shader_parameter("atm_steps", int(cfg["shell_steps"]))
	if cfg.has("atm_steps"):
		app.world.earth_mat.set_shader_parameter("atm_steps", int(cfg["atm_steps"]))
		app.world.shell_mat.set_shader_parameter("atm_steps", int(cfg["atm_steps"]))
	# Stress: extra labelled arrows on the spacecraft (overlay cost scaling).
	for i in int(cfg.get("extra_arrows", 0)):
		var n: Node3D = preload("res://src/overlays/registry.gd").create({"type": "arrow", "id": "stress%d" % i,
			"source": "@sun", "scale": "fixed_px", "px": 40.0 + i, "label": "arrow %d" % i}, app.source.scene)
		app.world.craft.add_child(n)
		app.world.overlays.nodes.append(n)
		app.world.overlays.labels.append(n.label)
	var rid: RID = app.world.viewport.get_viewport_rid()
	RenderingServer.viewport_set_measure_render_time(rid, true)
	var ov_us := PackedFloat64Array()
	var layout_us := PackedFloat64Array()
	var rig: CameraRig = app.rig
	rig.mode = CameraRig.Mode.ORBIT
	rig.frame_kind = "inertial"
	var gpu := PackedFloat64Array()
	var wall := PackedFloat64Array()
	var last := Time.get_ticks_usec()
	for i in frames + 30:
		var u := float(i) / frames
		rig.dist = exp(log(1.0) + (log(3.0e7) - log(1.0)) * (0.5 - 0.5 * cos(TAU * u)))
		rig.yaw = TAU * u * 1.5
		rig.pitch = 0.3
		app.render_at(app.clock.t_min + u * (app.clock.t_max - app.clock.t_min))
		await RenderingServer.frame_post_draw
		var now := Time.get_ticks_usec()
		if i >= 30:
			gpu.append(RenderingServer.viewport_get_measured_render_time_gpu(rid))
			ov_us.append(app.world.overlays.cost_us)
			layout_us.append(app.world.overlays.layout_us)
			wall.append((now - last) / 1000.0)
		last = now
	return {"gpu_ms": gpu, "frame_ms": wall, "overlay_cpu_us": ov_us, "label_layout_us": layout_us, "n_overlays": app.world.overlays.nodes.size(),
		"res": [app.world.viewport.size.x, app.world.viewport.size.y],
		"atmosphere": app.loaded_assets.has("atmosphere")}



# ------------------------------------------------------------------ timeline
## One interactive frame: pose the world, refresh the HUD, draw.
func _tick() -> void:
	app.render_at(app.clock.t)
	app.hud.refresh(app.clock, app.last_frame, app._status_lines())
	await RenderingServer.frame_post_draw


func _mouse(pos: Vector2, button: int, pressed: bool, shift := false, double := false) -> void:
	var e := InputEventMouseButton.new()
	e.position = pos
	e.global_position = pos
	e.button_index = button as MouseButton
	e.pressed = pressed
	e.shift_pressed = shift
	e.double_click = double
	app.get_viewport().push_input(e)


func _motion(pos: Vector2, held: bool) -> void:
	var e := InputEventMouseMotion.new()
	e.position = pos
	e.global_position = pos
	e.button_mask = MOUSE_BUTTON_MASK_LEFT if held else 0
	app.get_viewport().push_input(e)


func _key(code: Key, shift := false) -> void:
	for down: bool in [true, false]:
		var e := InputEventKey.new()
		e.keycode = code
		e.physical_keycode = code
		e.pressed = down
		e.shift_pressed = shift
		app.get_viewport().push_input(e)


func _root_image() -> Image:
	var img: Image = app.get_viewport().get_texture().get_image()
	img.convert(Image.FORMAT_RGBA8)
	return img


func _tl_state(tl: Control) -> Dictionary:
	var tr: Rect2 = tl._rects["track"]
	var lanes_px := []
	for row: Dictionary in tl._lane_rows:
		var r: Rect2 = row["rect"]
		lanes_px.append({"label": row["label"], "kind": row["kind"], "y0": tl.global_position.y + r.position.y,
			"y1": tl.global_position.y + r.end.y})
	return {"t": app.clock.t, "view": [tl.view0, tl.view1], "fit": tl.fit,
		"track_x0": tl.global_position.x + tr.position.x, "track_w": tr.size.x, "ui": tl.ui, "lanes_px": lanes_px}


## cfg: {clicks: [fractions], zoom_notches, drag: [f0, f1], type: [texts], add_bookmark_at, shots: bool,
##       perf_frames}. Everything is reported; run_viewer_checks.py judges it.
func timeline(cfg: Dictionary) -> Dictionary:
	var tl: Control = app.hud.timeline
	var clock: RefCounted = app.clock
	DisplayServer.window_set_size(app.world.viewport.size)
	app.world.resize_to(Vector2(app.world.viewport.size))
	clock.paused = true
	clock.t = float(cfg.get("t0", clock.t_min))
	var t_lanes := Time.get_ticks_usec()
	while not tl.lanes.done:
		tl.lanes.update()
	var lanes_ms := (Time.get_ticks_usec() - t_lanes) / 1000.0
	for i in 4:
		await _tick()
	var out := {"lanes_full_ms": lanes_ms, "rows": app.source.store.n_rows}
	var y_lane: float = tl.global_position.y + (tl._lane_rows[0]["rect"] as Rect2).get_center().y
	# Clicks on the Sun lane at fractions of the track, from the whole-run view.
	var clicks := []
	for f: float in cfg.get("clicks", []):
		tl.fit_view()
		await _tick()
		var st := _tl_state(tl)
		var x: float = st["track_x0"] + f * st["track_w"]
		_mouse(Vector2(x, y_lane), MOUSE_BUTTON_LEFT, true)
		_mouse(Vector2(x, y_lane), MOUSE_BUTTON_LEFT, false)
		await _tick()
		var img := _root_image()
		var playhead := []  # columns of accent pixels on the ruler row above the lanes
		var yr := int(st["lanes_px"][0]["y0"]) - 6
		for px in range(maxi(0, int(x) - 12), mini(img.get_width(), int(x) + 13)):
			var c := img.get_pixel(px, yr)
			if c.b > 0.8 and c.r < 0.6 and c.g > 0.45:
				playhead.append(px)
		clicks.append({"frac": f, "x": x, "state": st, "t_after": clock.t, "paused_after": clock.paused,
			"playhead_px": playhead})
	out["clicks"] = clicks
	# Wheel zoom about the middle of the track.
	tl.fit_view()
	await _tick()
	var st0 := _tl_state(tl)
	var xm: float = st0["track_x0"] + 0.5 * st0["track_w"]
	for i in int(cfg.get("zoom_notches", 3)):
		_mouse(Vector2(xm, y_lane), MOUSE_BUTTON_WHEEL_UP, true)
	await _tick()
	out["zoom"] = {"before": st0, "after": _tl_state(tl), "x": xm}
	# Drag-scrub between two fractions of the zoomed window.
	var dr: Array = cfg.get("drag", [0.2, 0.7])
	var st1 := _tl_state(tl)
	var xa: float = st1["track_x0"] + float(dr[0]) * st1["track_w"]
	var xb: float = st1["track_x0"] + float(dr[1]) * st1["track_w"]
	_mouse(Vector2(xa, y_lane), MOUSE_BUTTON_LEFT, true)
	for i in 10:
		_motion(Vector2(lerpf(xa, xb, (i + 1) / 10.0), y_lane), true)
		await _tick()
	_mouse(Vector2(xb, y_lane), MOUSE_BUTTON_LEFT, false)
	await _tick()
	out["drag"] = {"state": st1, "x_end": xb, "t_after": clock.t}
	# Keyboard: next sample, next event; typed times.
	var keys := {}
	clock.t = float(cfg.get("t_key", clock.t_min + 1.0))
	_key(KEY_RIGHT)
	keys["right"] = clock.t
	_key(KEY_RIGHT)
	keys["right2"] = clock.t
	_key(KEY_LEFT)
	keys["left"] = clock.t
	clock.t = clock.t_min
	_key(KEY_RIGHT, true)
	keys["next_event"] = clock.t
	out["keys"] = keys
	var typed := []
	for text: String in cfg.get("type", []):
		clock.t = clock.t_min
		tl._on_time_submitted(text)
		typed.append({"text": text, "t": clock.t})
	out["typed"] = typed
	# Bookmarks: the log's, plus one added with B (persisted for the next run).
	if cfg.has("add_bookmark_at"):
		clock.t = float(cfg["add_bookmark_at"])
		_key(KEY_B)
		await _tick()
		tl.list_popup.hide()
	out["bookmarks"] = tl.bookmarks.items.duplicate(true)
	out["bookmark_file"] = tl.bookmarks.path
	if cfg.get("remove_user_bookmarks", false):
		for b: Dictionary in tl.bookmarks.items.duplicate():
			tl.bookmarks.remove(b["id"])
	# Lanes as computed.
	out["eclipse"] = tl.lanes.eclipse_spans()
	var passes := []
	for i in tl.lanes.sites.size():
		passes.append({"site": tl.lanes.sites[i]["name"], "passes": tl.lanes.passes(i)})
	out["passes"] = passes
	# Screenshots: whole run with the state card (paused), then zoomed on a pass with the hover card.
	if cfg.get("shots", false):
		tl.fit_view()
		clock.t = float(cfg.get("t_shot", clock.t_min))
		for i in 3:
			await _tick()
		_root_image().save_png(_out("timeline_fit.png"))
		out["fit_state"] = _tl_state(tl)
		var card: Control = app.hud.card
		out["card"] = card.last.duplicate()
		out["card_visible"] = card.visible
		if not passes.is_empty() and not (passes[-1]["passes"] as Array).is_empty():
			var p: Array = passes[-1]["passes"][0]
			var span: float = (float(p[1]) - float(p[0])) * 1.25
			tl.set_view(float(p[2]) - span / 2.0, float(p[2]) + span / 2.0)
			await _tick()
			var stz := _tl_state(tl)
			var xh: float = tl.global_position.x + tl.t_to_x(float(p[2]))
			_motion(Vector2(xh, stz["lanes_px"][-1]["y0"] + 4.0), false)
			for i in 3:
				await _tick()
			_root_image().save_png(_out("timeline_zoomed.png"))
			out["zoomed_state"] = _tl_state(tl)
		tl.open_list()
		for i in 3:
			await _tick()
		_root_image().save_png(_out("timeline_bookmarks.png"))
		tl.list_popup.hide()
		tl.open_time_entry()
		tl.time_edit.text = "T+00:12:30"
		for i in 3:
			await _tick()
		_root_image().save_png(_out("timeline_type.png"))
		tl.time_edit.visible = false
	# Draw and update cost while playing.
	var nf := int(cfg.get("perf_frames", 0))
	if nf > 0:
		var draw := PackedFloat64Array()
		var upd := PackedFloat64Array()
		var static0: int = tl.static_draws
		var static_frame := []
		clock.paused = false
		clock.rate = (clock.t_max - clock.t_min) / (nf / 60.0)
		clock.t = clock.t_min
		for i in nf:
			clock.advance(1.0 / 60.0)
			var t0 := Time.get_ticks_usec()
			app.hud.timeline.refresh()
			upd.append(Time.get_ticks_usec() - t0)
			var before: int = tl.static_draws
			await _tick()
			draw.append(tl.draw_us)
			static_frame.append(tl.static_draws > before)
		out["draw_us"] = draw
		out["static_frame"] = static_frame
		out["static_draws"] = tl.static_draws - static0
		out["refresh_us"] = upd
	return out


# ------------------------------------------------------------------ overlay group toggles
## Click each chip and press the keys; report the group state and node visibility after each step,
## and screenshots of the spacecraft with the vectors shown and hidden.
func layers(_cfg: Dictionary) -> Dictionary:
	var ov: RefCounted = app.world.overlays
	var chips: Control = app.hud.layers
	DisplayServer.window_set_size(app.world.viewport.size)
	app.world.resize_to(Vector2(app.world.viewport.size))
	app.rig.fit_to_craft(app.world.craft.radius)
	app.clock.paused = true
	app.clock.t = app.clock.t_min + 0.5 * (app.clock.t_max - app.clock.t_min)
	for i in 4:
		await _tick()
	var out := {"groups": ov.groups(), "initial": ov.shown.duplicate(), "steps": [], "chips": []}
	for c: Dictionary in chips._chips:
		var r: Rect2 = c["rect"]
		out["chips"].append({"group": c["group"], "rect": [chips.global_position.x + r.position.x,
			chips.global_position.y + r.position.y, r.size.x, r.size.y]})
	var record := func(action: String) -> void:
		var vis := {}
		for g: String in ov.shown:
			vis[g] = ov.nodes.filter(func(n: Node3D) -> bool: return ov.group_of(n) == g).all(
				func(n: Node3D) -> bool: return n.visible == bool(ov.shown[g]))
			vis[g] = bool(ov.shown[g]) if vis[g] else not bool(ov.shown[g])
		out["steps"].append({"action": action, "shown": ov.shown.duplicate(), "visible": vis})
	var vec_rect := Rect2()
	for c: Dictionary in chips._chips:
		if c["group"] == "vectors":
			vec_rect = c["rect"]
	var centre := chips.global_position + vec_rect.get_center()
	_root_image().save_png(_out("vectors_on.png"))
	for i in 2:
		_mouse(centre, MOUSE_BUTTON_LEFT, true)
		_mouse(centre, MOUSE_BUTTON_LEFT, false)
		await _tick()
		record.call("click vectors")
		if i == 0:
			await _tick()
			_root_image().save_png(_out("vectors_off.png"))
	for k: Array in [[KEY_A, "key A"], [KEY_G, "key G"], [KEY_O, "key O"], [KEY_O, "key O"], [KEY_V, "key V"]]:
		_key(k[0])
		await _tick()
		record.call(k[1])
	await _tick()
	_root_image().save_png(_out("chips.png"))
	return out
