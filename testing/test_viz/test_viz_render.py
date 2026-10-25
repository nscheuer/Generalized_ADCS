"""Offline video rendering (ADCS.viz.render): camera keys, sample schedule, linear-light averaging, encoding."""

import json
import re
import subprocess

import numpy as np
import pytest

from ADCS.viz import render as rd
from ADCS.viz.render import CameraKey, RenderJob


def test_camera_holds_before_and_after_the_keys():
    keys = [CameraKey(at=1.0, dist_m=2.0), CameraKey(at=3.0, dist_m=8.0)]
    assert rd.camera_at(keys, 0.0)["dist_m"] == 2.0
    assert rd.camera_at(keys, 9.0)["dist_m"] == 8.0
    assert rd.camera_at(keys, 3.0)["dist_m"] == pytest.approx(8.0)


def test_camera_eases_distance_logarithmically_and_yaw_the_short_way():
    keys = [CameraKey(at=0.0, dist_m=1.0, yaw_deg=170.0, pitch_deg=0.0, fov_deg=40.0),
            CameraKey(at=2.0, dist_m=1.0e6, yaw_deg=-170.0, pitch_deg=30.0, fov_deg=60.0)]
    mid = rd.camera_at(keys, 1.0)  # smoothstep(0.5) = 0.5
    assert mid["dist_m"] == pytest.approx(1.0e3)  # geometric mean
    assert (mid["yaw_deg"] - 180.0) % 360.0 == pytest.approx(0.0, abs=1e-9)  # through 180, not through 0
    assert mid["pitch_deg"] == pytest.approx(15.0) and mid["fov_deg"] == pytest.approx(50.0)
    q = rd.camera_at(keys, 0.5)  # smoothstep(0.25) = 0.15625: eases in
    assert q["pitch_deg"] == pytest.approx(30.0 * 0.15625)
    lin = [keys[0], CameraKey(at=2.0, dist_m=1.0e6, yaw_deg=-170.0, pitch_deg=30.0, fov_deg=60.0, ease="linear")]
    assert rd.camera_at(lin, 0.5)["pitch_deg"] == pytest.approx(7.5)


def test_a_change_of_mode_or_frame_is_a_cut():
    keys = [CameraKey(at=0.0, mode="orbit", dist_m=2.0), CameraKey(at=4.0, mode="earth", dist_m=2.0e7)]
    assert rd.camera_at(keys, 3.99)["mode"] == "orbit" and rd.camera_at(keys, 3.99)["dist_m"] == 2.0
    assert rd.camera_at(keys, 4.0)["mode"] == "earth" and rd.camera_at(keys, 4.0)["dist_m"] == 2.0e7
    keys = [CameraKey(at=0.0, frame="lvlh"), CameraKey(at=4.0, frame="body", dist_m=9.0)]
    assert rd.camera_at(keys, 2.0)["frame"] == "lvlh" and rd.camera_at(keys, 2.0)["dist_m"] == 3.0


def test_samples_without_motion_blur():
    job = RenderJob(duration_s=2.0, fps=5.0, t_start=100.0, t_end=300.0)
    s = rd.samples(job, 0.0, 1000.0)
    assert len(s) == 10 and {x["layer"] for x in s} == {"all"}
    np.testing.assert_allclose([x["u"] for x in s], np.arange(10) * 2.0 / 9)
    np.testing.assert_allclose([x["t"] for x in s], np.linspace(100.0, 300.0, 10))
    assert s[0]["t"] == 100.0 and s[-1]["t"] == 300.0  # the frames span both ends of the window
    keys = [CameraKey(at=0.0, dist_m=1.0), CameraKey(at=2.0, dist_m=50.0)]
    last = rd.samples(RenderJob(duration_s=2.0, fps=5.0, keys=keys), 0.0, 10.0)[-1]
    assert last["cam"]["dist_m"] == pytest.approx(50.0)  # the camera reaches the last key


def test_samples_with_motion_blur_are_centred_and_add_the_sharp_layer():
    job = RenderJob(duration_s=1.5, fps=4.0, motion_blur=4, shutter=0.5)  # 6 frames, spacing 0.3 s
    s = rd.samples(job, 0.0, 100.0)
    assert len(s) == 6 * (4 + 2)
    frame2 = [x for x in s if x["i"] == 2]
    scene = [x for x in frame2 if x["j"] < 4]
    assert all(x["layer"] == "scene" for x in scene)
    np.testing.assert_allclose([x["u"] for x in scene], 0.6 + 0.15 * (np.array([0.5, 1.5, 2.5, 3.5]) / 4 - 0.5))
    assert np.mean([x["u"] for x in scene]) == pytest.approx(0.6)  # centred on the frame time
    assert [(x["j"], x["layer"]) for x in frame2[4:]] == [(4, "all"), (5, "scene")]
    assert all(x["u"] == pytest.approx(0.6) for x in frame2[4:])
    assert max(x["u"] for x in s) == 1.5  # clamped at the end
    assert min(x["u"] for x in s) == 0.0  # clamped at the start


def test_samples_reject_times_outside_the_log():
    with pytest.raises(ValueError, match="inside the log"):
        rd.samples(RenderJob(t_start=-5.0), 0.0, 100.0)
    with pytest.raises(ValueError, match="increase"):
        rd.samples(RenderJob(t_start=50.0, t_end=40.0), 0.0, 100.0)


@pytest.mark.parametrize("kw, match", [
    ({"width": 641}, "even"), ({"supersample": 5}, "supersample"), ({"width": 8000, "supersample": 3}, "16384"),
    ({"motion_blur": 0}, "motion_blur"), ({"shutter": 1.5}, "shutter"), ({"ui": True, "supersample": 2}, "ui"), ({"ui": True, "width": 640, "height": 360}, "960x540"),
    ({"quality": "ultra"}, "quality"), ({"keys": []}, "at least one"), ({"overlays": {"stars": False}}, "group"),
    ({"keys": [CameraKey(at=2.0), CameraKey(at=1.0)]}, "time order"), ({"keys": [CameraKey(at=0, mode="free")]}, "bad"),
])
def test_job_validation(kw, match):
    with pytest.raises(ValueError, match=match):
        RenderJob(**kw).validate()


def test_srgb_round_trip_and_linear_averaging():
    v = np.arange(256, dtype=np.uint8)
    np.testing.assert_array_equal(rd.linear_to_srgb(rd.srgb_to_linear(v)), v)
    black, white = np.zeros((2, 2, 3), np.uint8), np.full((2, 2, 3), 255, np.uint8)
    mid = rd.average_linear([black, white], 1)
    assert np.all(mid == 188)  # 50 % linear light is sRGB 188, not 128
    # Box filter down by 2: one 2x2 block of black and white -> the same 50 %.
    checker = np.array([[0, 255], [255, 0]], np.uint8)[..., None].repeat(3, axis=2)
    small = rd.average_linear([checker], 2)
    assert small.shape == (1, 1, 3) and np.all(small == 188)


def test_overlays_are_kept_sharp_under_motion_blur():
    a, b = np.zeros((4, 4, 3), np.uint8), np.full((4, 4, 3), 255, np.uint8)
    on = np.zeros((4, 4, 3), np.uint8)
    on[1, 1] = (255, 0, 0)  # an overlay pixel at the frame time
    off = np.zeros((4, 4, 3), np.uint8)
    out = rd.average_linear([a, b], 1, sharp=(on, off))
    assert tuple(out[1, 1]) == (255, 0, 0)  # the overlay, not blurred
    assert np.all(out[0, 0] == 188) and np.all(out[3, 3] == 188)  # the scene, blurred


def test_load_job_and_parse_key(tmp_path):
    p = tmp_path / "job.json"
    p.write_text(json.dumps({"duration_s": 4, "fps": 24, "supersample": 2,
                             "keys": [{"at": 0, "dist_m": 1.5}, {"at": 3, "mode": "earth", "dist_m": 2e7}]}))
    job = rd.load_job(p)
    job.validate()
    assert job.n_frames == 96 and job.keys[1].mode == "earth"
    k = rd.parse_key("at=2,mode=earth,dist=2e7,yaw=-60,pitch=30,fov=35,ease=linear")
    assert (k.at, k.mode, k.dist_m, k.yaw_deg, k.pitch_deg, k.fov_deg, k.ease) == (2, "earth", 2e7, -60, 30, 35, "linear")
    with pytest.raises(ValueError, match="at="):
        rd.parse_key("dist=3")
    with pytest.raises(ValueError, match="field"):
        rd.parse_key("at=0,zoom=3")


def test_cli_builds_the_job(monkeypatch, tmp_path):
    from ADCS.viz import __main__ as cli

    seen = {}

    def fake(path, job, out, **kw):
        seen.update(job=job, out=out, kw=kw)
        return rd.RenderResult(out, tmp_path, {})

    monkeypatch.setattr(rd, "render_video", fake)
    assert cli.main(["render", "run.vizlog", "out.mp4", "--duration", "3", "--res", "1280x720", "--supersample", "2",
                     "--motion-blur", "4", "--key", "at=0,dist=1.5", "--key", "at=2,dist=1e6", "--codec", "prores",
                     "--run", "7"]) == 0
    job = seen["job"]
    assert (job.duration_s, job.width, job.height, job.supersample, job.motion_blur) == (3, 1280, 720, 2, 4)
    assert [k.dist_m for k in job.keys] == [1.5, 1e6]
    assert seen["kw"]["codec"] == "prores" and seen["kw"]["run_id"] == "7"


FFMPEG = rd.find_ffmpeg()


@pytest.mark.skipif(FFMPEG is None, reason="no ffmpeg (install imageio-ffmpeg)")
@pytest.mark.parametrize("codec", ["h264", "prores"])
def test_ffmpeg_encodes_the_frames(tmp_path, codec):
    from PIL import Image

    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(12):
        im = np.zeros((90, 160, 3), np.uint8)
        im[:, : 10 * (i + 1)] = 200
        Image.fromarray(im).save(frames / f"{i:06d}.png")
    out = tmp_path / ("v" + rd.CODECS[codec][0])
    subprocess.run(rd.ffmpeg_command(FFMPEG, frames, 24.0, out, codec, 18), check=True)
    probe = subprocess.run([FFMPEG, "-i", str(out), "-f", "null", "-"], capture_output=True, text=True, check=False).stderr
    assert int(re.findall(r"frame=\s*(\d+)", probe)[-1]) == 12
    assert re.search(r"160x90", probe) and re.search(r"24 fps", probe)
