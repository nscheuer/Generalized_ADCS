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
## shots: [{name, t, cam_pos (GCRS m), look_at (GCRS m), up (GCRS), fov_deg, hide: ["craft", "moon"]}]
## Renders exactly the requested view (e.g. from DSCOVR for an EPIC comparison).
func pose(cfg: Dictionary) -> Dictionary:
	var out := []
	for s: Dictionary in cfg.get("shots", []):
		var hide: Array = s.get("hide", [])
		app.world.craft.visible = not hide.has("craft")
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
			await RenderingServer.frame_post_draw
		var img := _image()
		var file := "%s.png" % s.get("name", "pose_%d" % out.size())
		img.save_png(_out(file))
		out.append({"name": file, "t": t, "fov_deg": app.world.fov_deg, "info": app.world.info.duplicate()})
	return {"shots": out}


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
	var rid: RID = app.world.viewport.get_viewport_rid()
	RenderingServer.viewport_set_measure_render_time(rid, true)
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
			wall.append((now - last) / 1000.0)
		last = now
	return {"gpu_ms": gpu, "frame_ms": wall, "res": [app.world.viewport.size.x, app.world.viewport.size.y],
		"atmosphere": app.loaded_assets.has("atmosphere")}

