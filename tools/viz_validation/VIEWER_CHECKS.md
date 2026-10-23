# Viewer end-to-end checks

Generated 2026-10-22 23:27 UTC by `tools/viz_validation/run_viewer_checks.py`.

| Check | Result | Details | Criterion |
|---|---|---|---|
| assets | PASS | {"loaded": {"atmosphere": true, "earth_clouds": true, "earth_day": true, "earth_night": true, "earth_normals": true, "earth_ocean_mask": true, "moon_color": true, "moon_normals": true, "stars": 9050}, "load_ms": {"earth_clouds": 365, "earth_day": 2724, "earth_night": 88, "earth_normals": 534, "earth | all 8 asset products, model (3 LODs, 3 wheels x 3 LODs) over the stream |
| file_source | PASS | {"n_rows": 600, "model": "model"} | vizlog read from disk, model found via scene.json paths |
| shots | recorded | {} | screenshots for visual review |
| jitter | PASS | {"roi_mean_abs_diff": {"p50": 0.0, "p99": 0.0, "max": 0.0}} | spacecraft static in a body-locked view: mean ROI diff < 0.5 / 255 |
| jitter_control | PASS | {"roi_mean_abs_diff": {"p50": 2.20189525462963, "p99": 46.03755588107543, "max": 99.8126898871528}} | control (absolute float32) must visibly move: mean ROI diff > 5 |
| terminator | PASS | {"numeric_err_deg": 1.8478994056063678e-13, "image_err_deg": 0.010428019079495745, "image_offset_px": -0.040842445788551514} | sub-solar point < 0.005 deg; image terminator < 0.1 deg |
| landmarks | PASS | {"points": [{"name": "null_island", "err_px": 0.0, "err_km": 0.0}, {"name": "greenwich", "err_px": 0.10935201147822077, "err_km": 1.6049903632388072}, {"name": "cape_horn", "err_px": 0.09319111013539488, "err_km": 1.3677922489484757}, {"name": "honolulu", "err_px": 0.06607277670467512, "err_km": 0.9 | red dot at screen centre within 2 px, texture through the asset pipeline's cube maps |
| scrub | PASS | {"cpu_ms": {"p50": 0.397, "p99": 0.46502, "max": 0.47}, "displayed_ms": {"p50": 11.182500000000001, "p99": 14.563869999999998, "max": 15.634}, "rows": 86401} | jump anywhere in a 24 h log (86 401 rows) and show it: p99 < 50 ms |
| capture | PASS | {"frames": 91} | two --write-movie runs: bit-identical frames |
| live | PASS | {"rows": 2400, "expected_rows": 2400, "frames_while_live": 1125, "sim_seconds": 17.0, "follow_lag_s": {"p50": 0.150000000000091, "p99": 0.150000000000091, "max": 0.150000000000091}} | viewer follows a running simulation (broadcast) and ends with every row |
| moon | PASS | {"lit_fraction": 0.29771375868055555, "image": "/mnt/c/Users/goldm/AppData/Local/generalized_adcs/viewer_checks/moon/moon.png"} | Moon rendered and lit (gibbous phase fills the frame partly) |
| exposure | PASS | {"max_sky_px_next_to_earth_0ev": 0.0, "star_px_physical_0ev": 0, "star_px_physical_+16ev": 346, "star_px_artistic": 1487} | physical: no stars beside a sunlit Earth or in deep space at 0 EV; > 20 star pixels at +16 EV |
| perf_1080p | recorded | {"gpu_ms": {"p50": 5.28499984741211, "p99": 12.173150329589884, "max": 13.076000213623}, "frame_ms": {"p50": 6.050000000000001, "p99": 13.759159999999998, "max": 14.354}, "atmosphere": true, "res": [1920, 1080], "quality": "balanced"} | frame time, quality 'balanced', atmosphere + clouds + model (recorded; RTX 3060 budget: GPU p99 < 8.3 ms at 1440p) |
| perf_1440p | recorded | {"gpu_ms": {"p50": 9.092000007629395, "p99": 23.167509822845428, "max": 24.5289993286133}, "frame_ms": {"p50": 10.056999999999999, "p99": 24.310809999999996, "max": 25.498}, "atmosphere": true, "res": [2560, 1440], "quality": "balanced"} | frame time, quality 'balanced', atmosphere + clouds + model (recorded; RTX 3060 budget: GPU p99 < 8.3 ms at 1440p) |
| perf_1440p_high | recorded | {"gpu_ms": {"p50": 8.50500011444092, "p99": 26.129929714202873, "max": 27.443000793457}, "frame_ms": {"p50": 9.1495, "p99": 27.118479999999998, "max": 28.611}, "atmosphere": true, "res": [2560, 1440], "quality": "high"} | frame time, quality 'high', atmosphere + clouds + model (recorded; RTX 3060 budget: GPU p99 < 8.3 ms at 1440p) |
