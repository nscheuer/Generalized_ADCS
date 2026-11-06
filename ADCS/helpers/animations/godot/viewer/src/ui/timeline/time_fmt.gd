extends RefCounted
## Times shown and typed in the timeline. The log's time is TT seconds since
## its epoch; UTC = epoch (UTC) + t, ignoring a leap second inside the run.

const MONTHS := ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


## Unix time [s] of the log epoch from the manifest's ISO UTC string, or NAN.
static func epoch_unix(manifest: Dictionary) -> float:
	var iso: Variant = manifest.get("epoch", {}).get("iso_utc")
	if not iso is String or String(iso).is_empty():
		return NAN
	return parse_iso(String(iso))


## "2026-03-14T12:41:07.250Z" / "2026-03-14 12:41:07" / "2026-03-14" -> unix seconds, or NAN.
static func parse_iso(text: String) -> float:
	var s := text.strip_edges().replace("UTC", "").strip_edges().trim_suffix("Z").replace("T", " ")
	var re := RegEx.create_from_string("^(\\d{4})-(\\d{1,2})-(\\d{1,2})(?:\\s+(\\d{1,2}):(\\d{2})(?::(\\d{2}(?:\\.\\d+)?))?)?$")
	var m := re.search(s)
	if m == null:
		return NAN
	var d := {"year": int(m.get_string(1)), "month": int(m.get_string(2)), "day": int(m.get_string(3)),
		"hour": int(m.get_string(4)) if m.get_string(4) != "" else 0,
		"minute": int(m.get_string(5)) if m.get_string(5) != "" else 0, "second": 0}
	var sec := float(m.get_string(6)) if m.get_string(6) != "" else 0.0
	if d["month"] < 1 or d["month"] > 12 or d["day"] < 1 or d["day"] > 31 or d["hour"] > 23 or d["minute"] > 59 or sec >= 61.0:
		return NAN
	return float(Time.get_unix_time_from_datetime_dict(d)) + sec


## "2026-03-14 12:41:07.2" (decimals: digits after the seconds).
static func utc(unix: float, decimals: int = 0) -> String:
	if is_nan(unix):
		return "--"
	var scale := pow(10.0, decimals)
	var u := roundf(unix * scale) / scale
	var whole := floorf(u)
	var s := Time.get_datetime_string_from_unix_time(int(whole), true)
	if decimals > 0:
		s += ("." + "%0" + str(decimals) + "d") % (int(roundf((u - whole) * scale)) % int(scale))
	return s


## "12:41:07" / "12:41" / "Mar 14" for tick labels.
static func clock_label(unix: float, with_seconds: bool) -> String:
	var d := Time.get_datetime_dict_from_unix_time(int(floorf(unix + 1e-6)))
	return "%02d:%02d:%02d" % [d["hour"], d["minute"], d["second"]] if with_seconds else "%02d:%02d" % [d["hour"], d["minute"]]


static func date_label(unix: float) -> String:
	var d := Time.get_datetime_dict_from_unix_time(int(floorf(unix + 1e-6)))
	return "%s %d" % [MONTHS[int(d["month"]) - 1], d["day"]]


## Elapsed time since the start of the run: "T+00:41:07", "T+2d 03:00:00", "T−00:00:05".
static func elapsed(t: float, decimals: int = 0) -> String:
	var sign := "+" if t >= 0.0 else "−"
	var a := absf(t)
	var scale := pow(10.0, decimals)
	a = roundf(a * scale) / scale
	var days := int(a / 86400.0)
	var rem := a - days * 86400.0
	var h := int(rem / 3600.0)
	var m := int((rem - h * 3600.0) / 60.0)
	var s := rem - h * 3600.0 - m * 60.0
	var sec := (("%0" + str(3 + decimals) + "." + str(decimals) + "f") % s) if decimals > 0 else "%02d" % int(s)
	var hms := "%02d:%02d:%s" % [h, m, sec]
	return "T%s%s%s" % [sign, ("%dd " % days) if days > 0 else "", hms]


## Seconds from "1h2m3s", "62m", "90s", "3723", "1:02:03", "02:03" (mm:ss), or NAN.
static func parse_duration(text: String) -> float:
	var s := text.strip_edges().to_lower().replace(" ", "")
	if s.is_empty():
		return NAN
	if s.is_valid_float():
		return float(s)
	if ":" in s:
		var parts := s.split(":")
		var total := 0.0
		for p in parts:
			if not p.is_valid_float():
				return NAN
			total = total * 60.0 + float(p)
		return total
	var re := RegEx.create_from_string("^(?:(\\d+(?:\\.\\d+)?)d)?(?:(\\d+(?:\\.\\d+)?)h)?(?:(\\d+(?:\\.\\d+)?)m)?(?:(\\d+(?:\\.\\d+)?)s)?$")
	var m := re.search(s)
	if m == null or m.get_string(0).is_empty():
		return NAN
	var mult := [86400.0, 3600.0, 60.0, 1.0]
	var out := 0.0
	for i in 4:
		if m.get_string(i + 1) != "":
			out += float(m.get_string(i + 1)) * mult[i]
	return out


## A typed time -> seconds since the epoch, or NAN. Accepts a full UTC date-time,
## a UTC clock time ("12:41", "12:41:07"; on the date shown at `t_ref`), or an
## elapsed time ("T+41:07", "+2460", "T+1h2m"). A clock time on its own is
## read as UTC; prefix "+" or "T+" for elapsed time.
static func parse(text: String, epoch: float, t_ref: float) -> float:
	var s := text.strip_edges()
	var up := s.to_upper()
	if up.begins_with("T+") or up.begins_with("T-") or up.begins_with("T−") or s.begins_with("+") or s.begins_with("-"):
		var neg := "-" in s.substr(0, 2) or "−" in s.substr(0, 2)
		var body := s.substr(2 if up.begins_with("T") else 1)
		var d := parse_duration(body)
		return -d if neg else d
	if is_nan(epoch):
		return parse_duration(s)
	var full := parse_iso(s)
	if not is_nan(full):
		return full - epoch
	var re := RegEx.create_from_string("^(\\d{1,2}):(\\d{2})(?::(\\d{2}(?:\\.\\d+)?))?(?:\\s*(?:UTC|Z))?$")
	var m := re.search(up)
	if m != null:
		var day := floorf((epoch + t_ref) / 86400.0) * 86400.0
		var sec := float(m.get_string(3)) if m.get_string(3) != "" else 0.0
		if int(m.get_string(1)) > 23 or int(m.get_string(2)) > 59:
			return NAN
		return day + int(m.get_string(1)) * 3600.0 + int(m.get_string(2)) * 60.0 + sec - epoch
	return parse_duration(s)
