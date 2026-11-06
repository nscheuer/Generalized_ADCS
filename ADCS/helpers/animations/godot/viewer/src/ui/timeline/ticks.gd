extends RefCounted
## Ruler ticks on round UTC times: the major step is the smallest whose labels
## fit; minor ticks subdivide it.

const TF := preload("res://src/ui/timeline/time_fmt.gd")

## [major, minor] steps [s].
const STEPS: Array = [[0.1, 0.02], [0.2, 0.05], [0.5, 0.1], [1.0, 0.2], [2.0, 0.5], [5.0, 1.0], [10.0, 2.0],
	[15.0, 5.0], [30.0, 5.0], [60.0, 10.0], [120.0, 30.0], [300.0, 60.0], [600.0, 120.0], [900.0, 300.0],
	[1800.0, 300.0], [3600.0, 600.0], [7200.0, 1800.0], [10800.0, 3600.0], [21600.0, 3600.0],
	[43200.0, 10800.0], [86400.0, 21600.0], [172800.0, 43200.0], [604800.0, 86400.0]]


## Label width estimate [px] for a step (seconds shown below one minute).
static func label_px(major: float, char_px: float) -> float:
	return char_px * (8.0 if major < 60.0 else (5.0 if major < 86400.0 else 6.0)) + 24.0


## Index into STEPS for a visible span [s] over width [px].
static func choose(span_s: float, width_px: float, char_px: float = 7.0) -> int:
	var px_per_s := width_px / maxf(span_s, 1e-9)
	for i in STEPS.size():
		if STEPS[i][0] * px_per_s >= label_px(STEPS[i][0], char_px):
			return i
	return STEPS.size() - 1


## Ticks between t0 and t1 (seconds since the epoch), aligned to multiples of
## the step in UTC: [{t, major, label}]. Day boundaries are labelled with the date.
static func make(t0: float, t1: float, epoch: float, width_px: float, char_px: float = 7.0) -> Array:
	var out := []
	var e := 0.0 if is_nan(epoch) else epoch
	var step: Array = STEPS[choose(t1 - t0, width_px, char_px)]
	var major: float = step[0]
	var minor: float = step[1]
	var target := (t1 - t0) * label_px(major, char_px) / maxf(width_px, 1.0)
	if target < STEPS[0][0] or target > STEPS[STEPS.size() - 1][0]:
		var magnitude := pow(10.0, floorf(log(target) / log(10.0)))
		var scaled := target / magnitude
		major = (1.0 if scaled <= 1.0 else (2.0 if scaled <= 2.0 else (5.0 if scaled <= 5.0 else 10.0))) * magnitude
		minor = major / 5.0
	if major < 0.1:
		# UTC seconds cannot retain submicrosecond precision at Unix epoch scale.
		var first := int(ceilf(t0 / minor))
		var last := int(floorf(t1 / minor))
		if last - first > 4000:
			return out
		var decimals := clampi(int(ceilf(-log(major) / log(10.0))) + 1, 1, 9)
		for k in range(first, last + 1):
			var value := k * minor
			var on_major := k % 5 == 0
			out.append({"t": value, "major": on_major, "label": (("%." + str(decimals) + "f s") % value) if on_major else ""})
		return out
	var k0 := int(ceilf((e + t0) / minor - 1e-9))
	var k1 := int(floorf((e + t1) / minor + 1e-9))
	if k1 - k0 > 4000:  # never draw absurd numbers of ticks
		return out
	for k in range(k0, k1 + 1):
		var u := k * minor
		var is_major := absf(fmod(u + major * 0.5, major) - major * 0.5) < minor * 1e-3
		var tick := {"t": u - e, "major": is_major, "label": ""}
		if is_major:
			if is_nan(epoch):
				tick["label"] = TF.elapsed(u - e).trim_prefix("T+")
			elif major >= 86400.0 or absf(fmod(u, 86400.0)) < minor * 1e-3:
				tick["label"] = TF.date_label(u)
			else:
				tick["label"] = TF.clock_label(u, major < 60.0)
		out.append(tick)
	return out
