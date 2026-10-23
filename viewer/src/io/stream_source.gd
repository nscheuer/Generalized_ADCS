extends RefCounted
## Data source: a session from ADCS.viz.VizServer (playback or live).

const Protocol := preload("res://src/io/protocol.gd")
const FrameStore := preload("res://src/core/frame_store.gd")

var client := Protocol.new()
var manifest: Dictionary = {}
var scene: Dictionary = {}
var assets: Dictionary = {}
var store: FrameStore
var run_id := ""
var live := false
var finished := false
var error_message := ""


func open(host: String, port: int, token: String, wanted_run: String = "") -> bool:
	if not client.connect_to(host, port, token):
		error_message = client.error_message
		return false
	manifest = client.hello["manifest"]
	scene = client.hello.get("scene", {})
	live = String(client.hello.get("mode", "")).begins_with("live")
	var runs: Array = manifest.get("runs", [])
	run_id = wanted_run if wanted_run != "" else (str(runs[0]["id"]) if runs.size() > 0 else "0")
	store = FrameStore.new(client.names, client.widths)
	assets = client.assets
	return true


## Consume what has arrived. Returns true if new rows were added.
func poll() -> bool:
	var before := store.n_rows
	for m: Array in client.poll_messages():
		match m[0]:
			Protocol.Msg.BLOCK:
				var b: Array = client.decode_block(m[2])
				if b[0] == run_id:
					store.append_block(b[3], b[2])
			Protocol.Msg.ROW:
				var r: Array = client.decode_row(m[2])
				if r[0] == run_id:
					store.append_row(r[1])
			Protocol.Msg.ERROR:
				error_message = (m[2] as PackedByteArray).get_string_from_utf8()
	finished = client.session_ended
	assets = client.assets
	return store.n_rows != before


## Block until the whole playback session (or timeout) has arrived.
func receive_all(timeout_s: float) -> bool:
	var deadline := Time.get_ticks_msec() + int(timeout_s * 1000.0)
	while not finished and Time.get_ticks_msec() < deadline:
		poll()
		OS.delay_msec(1)
	return finished and store.n_rows >= 2
