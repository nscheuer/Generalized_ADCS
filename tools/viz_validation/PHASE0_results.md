# Phase 0-V prototype results

Generated 2026-10-22 19:17 UTC by `tools/viz_validation/run_phase0.py`.

- GPU: **Intel(R) Arc(TM) 140V GPU (16GB)** (1.4.348), Godot 4.7.2-stable (official)
- Host: Linux 6.6.87.2-microsoft-standard-WSL2 (WSL2)
- Viewer process: Windows (native, launched from WSL)

| Test | Result | Key numbers | Criterion |
|---|---|---|---|
| jitter | PASS | rms_px=0, max_px=0, lost_frames=0 | RMS < 0.1 px, max < 0.25 px, never lost |
| jitter_control | PASS | rms_px=0.6684, max_px=1.179, lost_frames=564 | control must show > 2 px (or lose the fiducial) |
| zoom | PASS | precision 8/8, occlusion 4/4, single frustum works up to far/near = 1e+07 | depth order correct at 1 m..4e8 m within a 1e6 frustum (sep 1e-5 x D) and across the scaled-space split |
| terminator | PASS | numeric_err_deg=1.848e-13, image_err_deg=0.02838, image_offset_px=-0.1265, basis_err_deg=8.538e-07 | numeric < 0.005 deg; image < 0.1 deg; float32 basis < 0.005 deg |
| landmarks | PASS | null_island 0.00 px, greenwich 0.03 px, cape_horn 0.06 px, honolulu 0.17 px | landmark at screen centre within 2 px |
| hdr | PASS | albedo_5e4 0.497/0.500, albedo_3e4 0.301/0.300, albedo_2e4 0.202/0.200 | values > 1 survive to tonemapping (within 10%) |
| capture | PASS | frames_run0=121, identical=True | two --write-movie runs produce bit-identical frames |
| transport_broadcast | PASS | received=10000, gaps=0, dropped=0 | 10000 rows of 512 B at 1000 Hz: all received, no gaps, none dropped |
| transport_lockstep | PASS | received=2000, rtt_p50_ms=1.745, rtt_p99_ms=5.435, one_way_p99_ms_est=2.717, rtt_max_ms=65.75 | all rows delivered; one-way latency p99 < 5 ms, estimated as round trip / 2 |
| perf_1080p | recorded | gpu_p50_ms=2.42, gpu_p99_ms=3.215, frame_p50_ms=2.987 | RTX 3060 target: GPU p50 < 4 ms, p99 < 8.3 ms (recorded only on other GPUs) |
| perf_1440p | recorded | gpu_p50_ms=4.452, gpu_p99_ms=5.985, frame_p50_ms=4.892 | RTX 3060 target: GPU p50 < 4 ms, p99 < 8.3 ms (recorded only on other GPUs) |

Raw measurements: `tools/viz_validation/results/latest.json`.
