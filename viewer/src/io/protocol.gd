extends RefCounted
## Client for the ADCS visualization stream, protocol v1.
## GDScript port of ADCS/viz/client.py; see docs/source/viz/protocol_v1.rst.
## Started from the Phase 0 prototype (viewer_proto/src/viz_stream.gd).

const MAGIC := "ADVZ"
const VERSION := 1
const HEADER_SIZE := 20
enum Msg { HELLO = 1, BLOCK = 2, ROW = 3, END = 4, ACK = 5, CONNECT = 6, ERROR = 7, ASSET = 8 }

var tcp := StreamPeerTCP.new()
var _buf := PackedByteArray()
## Messages decoded together with HELLO, handed out by the next poll_messages().
var _pending: Array = []
var hello: Dictionary = {}
var names: PackedStringArray = PackedStringArray()
var widths: PackedInt32Array = PackedInt32Array()
var lockstep := false
var session_ended := false
var error_message := ""
## Files sent as ASSET messages: name -> {"meta": Dictionary, "data": PackedByteArray}.
var assets: Dictionary = {}


func connect_to(host: String, port: int, token: String, timeout_s: float = 15.0) -> bool:
	if tcp.connect_to_host(host, port) != OK:
		error_message = "connect_to_host failed"
		return false
	var deadline := Time.get_ticks_msec() + int(timeout_s * 1000.0)
	while true:
		tcp.poll()
		var status := tcp.get_status()
		if status == StreamPeerTCP.STATUS_CONNECTED:
			break
		if status == StreamPeerTCP.STATUS_ERROR or Time.get_ticks_msec() > deadline:
			error_message = "could not connect to %s:%d" % [host, port]
			return false
		OS.delay_msec(1)
	tcp.set_no_delay(true)
	var connect_msg := {"token": token, "client": "adcs-viewer", "protocol": VERSION}
	send(Msg.CONNECT, JSON.stringify(connect_msg).to_utf8_buffer())
	# Wait for HELLO (or ERROR).
	while Time.get_ticks_msec() < deadline:
		var msgs := poll_messages()
		for i in msgs.size():
			var m: Array = msgs[i]
			if m[0] == Msg.HELLO:
				# Anything that arrived in the same read (first BLOCK/ROW) must not be lost.
				_pending = msgs.slice(i + 1)
				return true
			if m[0] == Msg.ERROR:
				error_message = str(JSON.parse_string((m[2] as PackedByteArray).get_string_from_utf8()))
				return false
		OS.delay_msec(1)
	error_message = "no HELLO received"
	return false


func send(kind: int, payload: PackedByteArray, seq: int = 0) -> void:
	var h := StreamPeerBuffer.new()
	h.big_endian = false
	h.put_data(MAGIC.to_ascii_buffer())
	h.put_u16(VERSION)
	h.put_u16(kind)
	h.put_u32(payload.size())
	h.put_u64(seq)
	var out := h.data_array
	out.append_array(payload)
	tcp.put_data(out)


## Read what is available and return complete messages as [kind, seq, payload].
## HELLO is consumed here (fills names/widths); ROW is acknowledged in lockstep.
func poll_messages() -> Array:
	tcp.poll()
	var n := tcp.get_available_bytes()
	if n > 0:
		var got: Array = tcp.get_data(n)
		if got[0] == OK:
			_buf.append_array(got[1])
	var out: Array = _pending
	_pending = []
	var off := 0
	while _buf.size() - off >= HEADER_SIZE:
		if _buf.slice(off, off + 4).get_string_from_ascii() != MAGIC:
			error_message = "bad magic"
			session_ended = true
			break
		var kind := _buf.decode_u16(off + 6)
		var length := _buf.decode_u32(off + 8)
		var seq := _buf.decode_u64(off + 12)
		if _buf.size() - off < HEADER_SIZE + length:
			break
		var payload := _buf.slice(off + HEADER_SIZE, off + HEADER_SIZE + length)
		off += HEADER_SIZE + length
		if kind == Msg.HELLO:
			_on_hello(payload)
		elif kind == Msg.ROW and lockstep:
			send(Msg.ACK, PackedByteArray(), seq)
		elif kind == Msg.ASSET:
			_on_asset(payload)
		elif kind == Msg.END:
			var info: Variant = JSON.parse_string(payload.get_string_from_utf8())
			if info is Dictionary and not (info as Dictionary).has("run_id"):
				session_ended = true
		out.append([kind, seq, payload])
	if off > 0:
		_buf = _buf.slice(off)
	return out


func _on_asset(payload: PackedByteArray) -> void:
	var rid: Array = decode_run_id(payload)
	var off: int = rid[1]
	var m := payload.decode_u32(off)
	var meta: Variant = JSON.parse_string(payload.slice(off + 4, off + 4 + m).get_string_from_utf8())
	assets[rid[0]] = {"meta": meta if meta is Dictionary else {}, "data": payload.slice(off + 4 + m)}


func _on_hello(payload: PackedByteArray) -> void:
	hello = JSON.parse_string(payload.get_string_from_utf8())
	lockstep = hello.get("mode", "") == "live-lockstep"
	names.clear()
	widths.clear()
	for c: Dictionary in hello["manifest"]["channels"]:
		names.append(c["name"])
		widths.append(int(c["width"]))


static func decode_run_id(payload: PackedByteArray) -> Array:
	var n := payload.decode_u16(0)
	return [payload.slice(2, 2 + n).get_string_from_utf8(), 2 + n]


## BLOCK payload -> [run_id, row_start, n_rows, {channel: PackedFloat64Array (n*width)}]
func decode_block(payload: PackedByteArray) -> Array:
	var rid: Array = decode_run_id(payload)
	var off: int = rid[1]
	var row_start := payload.decode_u32(off)
	var n_rows := payload.decode_u32(off + 4)
	off += 8
	var cols := {}
	for i in names.size():
		var count := n_rows * widths[i]
		cols[names[i]] = payload.slice(off, off + 8 * count).to_float64_array()
		off += 8 * count
	return [rid[0], row_start, n_rows, cols]


## ROW payload -> [run_id, PackedFloat64Array of all channels]
func decode_row(payload: PackedByteArray) -> Array:
	var rid: Array = decode_run_id(payload)
	return [rid[0], payload.slice(rid[1]).to_float64_array()]
