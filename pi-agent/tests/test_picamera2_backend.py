"""Logique du backend Picamera2 contre un module ``picamera2`` factice.

Ne remplace pas un essai sur le Pi (voir docs/pi-agent.md, « Validation sur le matériel ») :
vérifie l'enchaînement des appels, les contrôles envoyés, le calcul de ``seq`` à partir
de ``SensorTimestamp``, la conversion de l'aperçu et la calibration AWB.
"""

from __future__ import annotations

import sys
import threading
import time
import types

import numpy as np
import pytest
from eggvision_shared.formats import CameraSettings


class FakeRequest:
    def __init__(self, cam: FakePicamera2, ts: int) -> None:
        self.cam, self.ts = cam, ts
        self.released = False

    def make_array(self, name: str) -> np.ndarray:
        cfg = self.cam.config[name]
        w, h = cfg["size"]
        if name == "main":
            return np.full((h, w, 3), 100, np.uint8)
        yuv = np.full((h * 3 // 2, w), 128, np.uint8)
        yuv[:h] = 90
        return yuv

    def get_metadata(self) -> dict:
        md = {"SensorTimestamp": self.ts, "ExposureTime": self.cam.controls.get("ExposureTime"), "AnalogueGain": 1.0}
        md["ColourGains"] = (2.6, 1.7) if self.cam.controls.get("AwbEnable") else self.cam.controls.get("ColourGains")
        return md

    def release(self) -> None:
        self.released = True


class FakePicamera2:
    instances: list[FakePicamera2] = []
    skip_at: int | None = None  # simule une image perdue par le capteur

    @staticmethod
    def global_camera_info() -> list[dict]:
        return [{"Model": "imx296", "Num": 0}]

    def __init__(self, num: int = 0) -> None:
        self.camera_properties = {"Model": "imx296", "PixelArraySize": (1456, 1088)}
        self.controls: dict = {}
        self.config: dict = {}
        self.started = False
        self.n = 0
        self.t0 = 1_000_000_000
        self.calls: list[str] = []
        FakePicamera2.instances.append(self)

    def create_video_configuration(self, main, lores, buffer_count, controls, display):
        assert display is None
        return {"main": dict(main), "lores": dict(lores), "buffer_count": buffer_count, "controls": dict(controls)}

    def configure(self, config) -> None:
        self.calls.append("configure")
        self.config = config
        self.controls.update(config["controls"])

    def start(self) -> None:
        self.calls.append("start")
        self.started = True

    def stop(self) -> None:
        self.calls.append("stop")
        self.started = False

    def close(self) -> None:
        self.calls.append("close")

    def set_controls(self, controls: dict) -> None:
        self.calls.append(f"set_controls:{sorted(controls)}")
        self.controls.update(controls)

    def capture_request(self) -> FakeRequest:
        if not self.started:
            time.sleep(0.01)
            raise RuntimeError("caméra arrêtée")
        time.sleep(0.005)
        period = self.controls["FrameDurationLimits"][0] * 1000
        self.n += 1
        if FakePicamera2.skip_at is not None and self.n == FakePicamera2.skip_at:
            self.n += 1  # une image sautée
        return FakeRequest(self, self.t0 + self.n * period)


@pytest.fixture()
def backend(monkeypatch):
    mod = types.ModuleType("picamera2")
    mod.Picamera2 = FakePicamera2  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "picamera2", mod)
    FakePicamera2.instances.clear()
    FakePicamera2.skip_at = 5
    from eggvision_agent.camera.picamera2_backend import Picamera2Backend, picamera2_available

    assert picamera2_available()
    b = Picamera2Backend()
    yield b
    b.stop()


def test_configuration_and_frames(backend):
    frames = []
    lock = threading.Lock()

    def on_frame(f):
        with lock:
            frames.append(f)

    backend.start(on_frame, CameraSettings(exposure_us=500, fps=30))
    cam = FakePicamera2.instances[0]
    cfg = cam.config
    assert cfg["main"] == {"size": (1456, 1088), "format": "RGB888"}
    lw, lh = cfg["lores"]["size"]
    assert cfg["lores"]["format"] == "YUV420" and lw % 64 == 0 and lh % 2 == 0
    c = cfg["controls"]
    assert c["AeEnable"] is False and c["AwbEnable"] is False
    assert c["FrameDurationLimits"] == (33333, 33333) and c["ExposureTime"] == 500
    assert c["ColourGains"] == (2.0, 1.8) and "ScalerCrop" not in c
    deadline = time.time() + 5
    while len(frames) < 8 and time.time() < deadline:
        time.sleep(0.01)
    with lock:
        seqs = [f.seq for f in frames]
        first = frames[0]
    assert seqs[:6] == [0, 1, 2, 3, 5, 6]  # l'image sautée par le capteur crée un trou dans seq
    assert first.image.shape == (1088, 1456, 3) and first.image.flags["C_CONTIGUOUS"]
    preview = backend.preview_to_bgr(first.preview)
    assert preview.shape == (lh, lw, 3)
    with pytest.raises(ValueError):
        backend.preview_to_bgr(np.zeros((10, 10), np.uint8))
    assert backend.ready and backend.info()["metadata"]["ExposureTime"] == 500


def test_live_settings_roi_and_awb(backend):
    backend.start(lambda f: None, CameraSettings())
    cam = FakePicamera2.instances[0]
    backend.apply_settings(CameraSettings(exposure_us=800, analogue_gain=2.0, fps=25))
    assert cam.controls["ExposureTime"] == 800 and cam.controls["FrameDurationLimits"] == (40000, 40000)
    assert cam.calls.count("configure") == 1  # réglages à chaud, sans reconfiguration
    # changement de ROI : reconfiguration complète, sortie à la taille de la ROI
    backend.apply_settings(CameraSettings(roi=(100, 200, 801, 601)))
    assert cam.calls.count("configure") == 2 and cam.config["main"]["size"] == (800, 600)
    assert cam.controls["ScalerCrop"] == (100, 200, 800, 600)
    assert backend.output_size(CameraSettings(roi=(100, 200, 801, 601))) == (800, 600)
    time.sleep(0.1)
    red, blue = backend.calibrate_awb(0.5, CameraSettings())
    assert (red, blue) == (2.6, 1.7)
    assert cam.controls["AwbEnable"] is False and cam.controls["ColourGains"] == (2.6, 1.7)
    backend.stop()
    assert cam.calls[-2:] == ["stop", "close"] and not backend.ready
