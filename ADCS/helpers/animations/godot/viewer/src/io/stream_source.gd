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
var ephemeris_state: Dictionary = {}
var _ephemeris_pending := false
var _ephemeris_wanted := NAN
var _ephemeris_failed_from := INF


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
			Protocol.Msg.EPHEMERIS_STATE:
				var state: Variant = JSON.parse_string((m[2] as PackedByteArray).get_string_from_utf8())
				_ephemeris_pending = false
				if state is Dictionary:
					if state.has("error"):
						error_message = String(state["error"])
						_ephemeris_failed_from = minf(_ephemeris_failed_from, float(state["t_s"]))
					else:
						ephemeris_state = state
				if not is_nan(_ephemeris_wanted):
					request_ephemeris(_ephemeris_wanted)
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


## Request DE421 state outside the logged interval. One request stays in flight;
## a timeline jump replaces the next requested time instead of queuing work.
func request_ephemeris(t_s: float) -> void:
	if live or t_s <= store.t_max() or t_s >= _ephemeris_failed_from:
		return
	_ephemeris_wanted = t_s
	if _ephemeris_pending:
		return
	if not ephemeris_state.is_empty() and absf(float(ephemeris_state["t_s"]) - t_s) < 1.0:
		_ephemeris_wanted = NAN
		return
	_ephemeris_pending = true
	_ephemeris_wanted = NAN
	client.send(Protocol.Msg.EPHEMERIS_QUERY, JSON.stringify({"t_s": t_s}).to_utf8_buffer())


func apply_ephemeris(frame: Dictionary, t_s: float) -> void:
	if t_s <= store.t_max() or ephemeris_state.is_empty():
		return
	if absf(float(ephemeris_state["t_s"]) - t_s) > 300.0:
		return
	for key: String in ephemeris_state:
		if key != "t_s":
			frame[key] = PackedFloat64Array(ephemeris_state[key])


## Block until the whole playback session (or timeout) has arrived.
func receive_all(timeout_s: float) -> bool:
	var deadline := Time.get_ticks_msec() + int(timeout_s * 1000.0)
	while not finished and Time.get_ticks_msec() < deadline:
		poll()
		OS.delay_msec(1)
	return finished and store.n_rows >= 2
