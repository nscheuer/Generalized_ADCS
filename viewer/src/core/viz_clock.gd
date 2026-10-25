extends RefCounted
## Simulation time shown by the viewer, in seconds since the log epoch (TT).
##
## Interactive: advances with wall time x rate. Deterministic (capture): set
## explicitly from a frame index, never from the wall clock.

const RATES: Array[float] = [-1000.0, -100.0, -10.0, -1.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0,
	120.0, 300.0, 600.0, 1800.0, 3600.0, 10000.0]

var t := 0.0
var rate := 1.0
var paused := false
var t_min := 0.0
var t_max := 0.0
## Live sessions: stay this far behind the newest row to hide network jitter.
var follow_live := false
var live_lag_s := 0.15


func set_range(lo: float, hi: float) -> void:
	t_min = lo
	t_max = hi


func advance(wall_dt: float) -> void:
	if follow_live:
		t = maxf(t_min, t_max - live_lag_s)
		return
	if paused:
		return
	t = clampf(t + wall_dt * rate, t_min, t_max)
	if t >= t_max and rate > 0.0:
		paused = true


func seek(ts: float) -> void:
	follow_live = false
	t = clampf(ts, t_min, t_max)


func faster() -> void:
	_step_rate(1)


func slower() -> void:
	_step_rate(-1)


func _step_rate(d: int) -> void:
	var i := RATES.find(rate)
	if i < 0:
		i = RATES.find(1.0)
	rate = RATES[clampi(i + d, 0, RATES.size() - 1)]


func toggle_pause() -> void:
	if paused and t >= t_max and rate > 0.0:
		t = t_min  # replay from the start
	paused = not paused
	follow_live = false
