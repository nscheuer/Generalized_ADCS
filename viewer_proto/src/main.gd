extends Node
## Prototype entry point.
##
##   <godot> --path viewer_proto -- --connect HOST:PORT --token T --test NAME
##           [--config cfg.json] [--out DIR] [--res WxH]
##
## Receives a session from ADCS.viz.VizServer, runs one test, writes
## DIR/result.json and quits. Driven by tools/viz_validation/run_phase0.py.

const VizStream := preload("res://src/viz_stream.gd")
const FrameStore := preload("res://src/frame_store.gd")
const Scene := preload("res://src/scene.gd")
const Tests := preload("res://src/tests.gd")

var opts: Dictionary = {}
var config: Dictionary = {}
var stream: VizStream
var store: FrameStore
var scene: Scene
var out_dir := ""


func _ready() -> void:
	opts = _parse_args(OS.get_cmdline_user_args())
	out_dir = opts.get("out", OS.get_user_data_dir())
	DirAccess.make_dir_recursive_absolute(out_dir)
	if opts.has("config"):
		config = JSON.parse_string(FileAccess.get_file_as_string(opts["config"]))
	var test_name: String = opts.get("test", "view")

	stream = VizStream.new()
	var hp := String(opts.get("connect", "127.0.0.1:0")).rsplit(":", true, 1)
	if not stream.connect_to(hp[0], int(hp[1]), opts.get("token", "")):
		_finish({"error": stream.error_message})
		return

	var tests := Tests.new(self)
	var result: Dictionary
	if test_name.begins_with("transport"):
		result = tests.transport()
	else:
		if not _receive_all(60.0):
			_finish({"error": "session did not complete: " + stream.error_message})
			return
		var res := Vector2i(1280, 720)
		if opts.has("res"):
			var wh := String(opts["res"]).split("x")
			res = Vector2i(int(wh[0]), int(wh[1]))
		scene = Scene.new(self, res)
		var cam_cfg: Dictionary = config.get("camera", {})
		scene.camera.near = cam_cfg.get("near", scene.camera.near)
		scene.camera.far = cam_cfg.get("far", scene.camera.far)
		await get_tree().process_frame
		match test_name:
			"jitter": result = await tests.jitter()
			"zoom": result = await tests.zoom()
			"terminator": result = await tests.terminator()
			"landmarks": result = await tests.landmarks()
			"perf": result = await tests.perf()
			"capture": result = await tests.capture()
			"hdr": result = await tests.hdr()
			_: result = {"error": "unknown test " + test_name}
	result["test"] = test_name
	result["adapter"] = RenderingServer.get_video_adapter_name()
	result["api"] = RenderingServer.get_video_adapter_api_version()
	result["godot"] = Engine.get_version_info()["string"]
	_finish(result)


## Collect a playback session (BLOCKs of run 0 until END {}).
func _receive_all(timeout_s: float) -> bool:
	var deadline := Time.get_ticks_msec() + int(timeout_s * 1000.0)
	while not stream.session_ended and Time.get_ticks_msec() < deadline:
		for m: Array in stream.poll_messages():
			if m[0] == VizStream.Msg.BLOCK:
				var b: Array = stream.decode_block(m[2])
				if store == null:
					store = FrameStore.new(stream.names, stream.widths)
				if b[0] == store_run_id():
					store.append_block(b[3], b[2])
		OS.delay_msec(1)
	return stream.session_ended and store != null and store.n_rows >= 2


func store_run_id() -> String:
	var runs: Array = stream.hello["manifest"]["runs"]
	return str(runs[0]["id"]) if runs.size() > 0 else "0"


func save_image(img: Image, name: String) -> void:
	img.save_png(out_dir.path_join(name))


func _finish(result: Dictionary) -> void:
	var f := FileAccess.open(out_dir.path_join("result.json"), FileAccess.WRITE)
	f.store_string(JSON.stringify(result))
	f.close()
	if result.has("error"):
		printerr("prototype error: ", result["error"])
	get_tree().quit(1 if result.has("error") else 0)


static func _parse_args(argv: PackedStringArray) -> Dictionary:
	var out := {}
	var i := 0
	while i < argv.size():
		var a := argv[i]
		if a.begins_with("--"):
			var key := a.substr(2)
			if i + 1 < argv.size() and not argv[i + 1].begins_with("--"):
				out[key] = argv[i + 1]
				i += 1
			else:
				out[key] = true
		i += 1
	return out
