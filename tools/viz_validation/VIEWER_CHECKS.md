# Viewer end-to-end checks

Generated 2026-10-22 21:29 UTC by `tools/viz_validation/run_viewer_checks.py`.

| Check | Result | Details | Criterion |
|---|---|---|---|
| assets | PASS | {"loaded": {"earth_clouds": true, "earth_day": true, "earth_night": true, "earth_normals": true, "earth_ocean_mask": true, "moon_color": true, "moon_normals": true, "stars": 9050}, "load_ms": {"earth_clouds": 312, "earth_day": 2503, "earth_night": 81, "earth_normals": 500, "earth_ocean_mask": 73, "m | all 8 asset products, model (3 LODs, 3 wheels x 3 LODs) over the stream |
| file_source | PASS | {"n_rows": 600, "model": "model"} | vizlog read from disk, model found via scene.json paths |
| shots | recorded | {} | screenshots for visual review |
| jitter | PASS | {"roi_mean_abs_diff": {"p50": 0.0, "p99": 0.0, "max": 0.0}} | spacecraft static in a body-locked view: mean ROI diff < 0.5 / 255 |
| jitter_control | PASS | {"roi_mean_abs_diff": {"p50": 2.35225875289352, "p99": 23.265428964118907, "max": 99.6171694155093}} | control (absolute float32) must visibly move: mean ROI diff > 5 |
| terminator | PASS | {"numeric_err_deg": 1.8478994056063678e-13, "image_err_deg": 0.010101325680908189, "image_offset_px": -0.039562916347765806} | sub-solar point < 0.005 deg; image terminator < 0.1 deg |
| landmarks | PASS | {"points": [{"name": "null_island", "err_px": 0.0, "err_km": 0.0}, {"name": "greenwich", "err_px": 0.0895502073796947, "err_km": 1.3143536906870006}, {"name": "cape_horn", "err_px": 0.10321030484236732, "err_km": 1.5148467999779898}, {"name": "honolulu", "err_px": 0.07043082162866154, "err_km": 1.03 | red dot at screen centre within 2 px, texture through the asset pipeline's cube maps |
| scrub | PASS | {"cpu_ms": {"p50": 0.234, "p99": 0.3692099999999998, "max": 0.403}, "displayed_ms": {"p50": 8.469999999999999, "p99": 11.164839999999996, "max": 15.205}, "rows": 86401} | jump anywhere in a 24 h log (86 401 rows) and show it: p99 < 50 ms |
| capture | PASS | {"frames": 91} | two --write-movie runs: bit-identical frames |
| live | PASS | {"rows": 2400, "expected_rows": 2400, "frames_while_live": 1060, "sim_seconds": 15.8, "follow_lag_s": {"p50": 0.150000000000091, "p99": 0.150000000000091, "max": 0.150000000000091}} | viewer follows a running simulation (broadcast) and ends with every row |
