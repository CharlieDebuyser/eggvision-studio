"""Prise de vue : sources locales (simulée, USB factice), réglages, préréglages, aperçu, API."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest
from eggvision import state
from eggvision.capture.analysis import analyze_image, detect_eggs
from eggvision.capture.base import SourceError
from eggvision.capture.pi import normalize_agent_url
from eggvision.capture.usb import UsbCameraSource
from eggvision.db import models as m
from eggvision_shared.formats import LotManifest, read_metadata


def _wait(cond, timeout: float = 30.0, step: float = 0.05):
    deadline = time.time() + timeout
    while True:
        value = cond()
        if value:
            return value
        if time.time() > deadline:
            raise TimeoutError("condition non atteinte")
        time.sleep(step)


@pytest.fixture()
def empty_project(client, tmp_path):
    yield state.projects.create(tmp_path, "Essai capture")


def _connect_sim(client, **params):
    r = client.post("/api/capture/connect", json={"kind": "simulated", "scale": 0.25, **params})
    assert r.status_code == 200, r.text
    _wait(lambda: (client.get("/api/capture/status").json()["source"] or {}).get("actual_fps", 0) > 0)
    return r.json()


LOT = {"name": "Viables matin", "type": "viable", "farm": "Ferme du Val", "comment": "essai"}


def test_simulated_source_end_to_end(client, empty_project):
    st = _connect_sim(client)
    assert st["connected"] and st["kind"] == "simulated"
    src = client.get("/api/capture/status").json()["source"]
    assert src["camera"]["output_width"] == 368 and src["ready"]
    assert 20 < src["actual_fps"] < 40
    empty_project.update_config({"user": {"name": "Bob"}})

    r = client.post("/api/capture/lots", json={**LOT, "settings": {"exposure_us": 700}})
    assert r.status_code == 200, r.text
    live = r.json()
    lot_id = live["lot"]["lot_id"]
    assert live["active"] and live["locked"] and live["lot"]["kind"] == "simulated"
    # verrouillage pendant la capture
    assert client.put("/api/capture/settings", json={"exposure_us": 900}).status_code == 409
    assert client.post("/api/capture/awb", json={"duration_s": 0}).status_code == 409
    assert client.post("/api/capture/lots", json=LOT).status_code == 409
    assert client.post("/api/capture/disconnect").status_code == 409
    assert client.post("/api/capture/connect", json={"kind": "simulated"}).status_code == 409
    with empty_project.session() as s:
        assert s.get(m.Lot, lot_id).status == "en_cours"  # visible dès le départ
    time.sleep(1.2)
    cur = client.get("/api/capture/lots/current").json()
    assert cur["stats"]["frames_written"] > 10 and cur["stats"]["measured_fps"] == pytest.approx(30, rel=0.05)
    assert cur["stats"]["disk_free_bytes"] > 0 and cur["stats"]["queue_size"] > 0

    stop = client.post("/api/capture/lots/stop").json()
    assert not stop["active"] and stop["result"]["lot_id"] == lot_id
    assert stop["last"]["status"] == "termine" and stop["last"]["stop_reason"] == "user"
    manifest = LotManifest.load(empty_project.lot_dir(lot_id))
    assert manifest.camera.exposure_us == 700 and manifest.operator == "Bob" and manifest.source["kind"] == "simulated"
    metas = read_metadata(empty_project.lot_dir(lot_id))
    with empty_project.session() as s:
        lot = s.get(m.Lot, lot_id)
        assert lot.status == "termine" and lot.sync_status == "done" and lot.source_kind == "simulated"
        assert lot.frame_count == len(metas) == manifest.stats.frames_written > 30
        assert lot.farm == "Ferme du Val" and lot.camera["exposure_us"] == 700
        assert lot.extra["stop_reason"] == "user"
        assert s.query(m.Frame).filter(m.Frame.lot_id == lot_id).count() == len(metas)
    lots = client.get("/api/capture/lots").json()
    assert lots["total"] == 1 and lots["items"][0]["id"] == lot_id
    assert client.put("/api/capture/settings", json={"exposure_us": 900}).status_code == 200
    assert client.post("/api/capture/disconnect").json()["connected"] is False


def test_disk_threshold_auto_stop_local(client, empty_project, monkeypatch):
    _connect_sim(client)
    assert client.put("/api/capture/options", json={"min_free_gb": 9999, "auto_sync": True}).status_code == 200
    r = client.post("/api/capture/lots", json=LOT)
    assert r.status_code == 507 and "Espace disque" in r.json()["detail"]
    client.put("/api/capture/options", json={"min_free_gb": 0.001, "auto_sync": True})
    assert client.post("/api/capture/lots", json=LOT).status_code == 200
    import eggvision_shared.capture as cap

    time.sleep(0.3)
    monkeypatch.setattr(cap, "disk_free_bytes", lambda path: 10)
    live = _wait(lambda: (lambda x: x if not x["active"] else None)(client.get("/api/capture/lots/current").json()), timeout=10)
    assert live["last"]["stop_reason"] == "disk_low" and live["last"]["status"] == "termine"
    with empty_project.session() as s:
        lot = s.query(m.Lot).one()
        assert lot.extra["stop_reason"] == "disk_low" and lot.frame_count > 0


def test_settings_awb_presets(client, empty_project):
    assert client.get("/api/capture/settings").status_code == 409  # aucune source
    _connect_sim(client)
    s = client.put("/api/capture/settings", json={"exposure_us": 650, "fps": 25}).json()
    assert s["exposure_us"] == 650 and s["fps"] == 25
    assert client.put("/api/capture/settings", json={"exposure_us": 100000}).status_code == 422
    awb = client.post("/api/capture/awb", json={"duration_s": 0.05}).json()
    assert awb["settings"]["red_gain"] == awb["red_gain"] != 2.0

    assert client.put("/api/capture/presets/Matin").json()["settings"]["exposure_us"] == 650  # réglages actuels
    body = {"settings": {"exposure_us": 300, "analogue_gain": 2.5, "red_gain": 1.5, "blue_gain": 1.6, "fps": 30, "roi": [0, 0, 200, 150]}}
    assert client.put("/api/capture/presets/Soir", json=body).status_code == 200
    assert client.put("/api/capture/presets/Mauvais", json={"settings": {"analogue_gain": 100}}).status_code == 422
    presets = client.get("/api/capture/presets").json()
    assert [p["name"] for p in presets] == ["Matin", "Soir"]
    applied = client.post("/api/capture/presets/Soir/apply").json()
    assert applied["exposure_us"] == 300 and applied["roi"] == [0, 0, 200, 150]
    _wait(lambda: client.get("/api/capture/status").json()["source"]["camera"]["output_width"] == 200)
    assert client.delete("/api/capture/presets/Soir").status_code == 200
    assert client.delete("/api/capture/presets/Soir").status_code == 404
    assert client.post("/api/capture/presets/Soir/apply").status_code == 404


def test_preview_analysis_overlays(client, empty_project):
    assert client.get("/api/capture/preview.jpg").status_code == 409
    _connect_sim(client)
    r = client.get("/api/capture/preview.jpg", params={"width": 160})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR).shape[1] == 160
    a = client.get("/api/capture/preview/analysis", params={"detections": True, "bins": 32}).json()
    assert a["sharpness"] > 0 and len(a["histogram"]["r"]) == 32
    assert sum(a["histogram"]["luma"]) == pytest.approx(1.0, abs=1e-3)
    assert 0 <= a["saturated"]["any"] <= 1 and "detections" in a and "detections_frame" in a and "image" not in a
    img = client.get("/api/capture/preview/analysis", params={"detections": True, "image": True, "width": 200}).json()["image"]
    import base64

    assert img.startswith("data:image/jpeg;base64,")
    assert cv2.imdecode(np.frombuffer(base64.b64decode(img.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR).shape[1] == 200
    src = client.get("/api/capture/status").json()["source"]
    assert src["storage"]["free_bytes"] > 0  # source locale : disque du projet
    # surexposition → pixels saturés signalés
    client.put("/api/capture/settings", json={"exposure_us": 4000, "analogue_gain": 4})
    time.sleep(0.15)
    hot = client.get("/api/capture/preview/analysis").json()
    assert hot["saturation_warning"] and hot["saturated"]["any"] > a["saturated"]["any"]

    ov = client.get("/api/capture/overlays").json()
    assert len(ov["lane_lines"]) == 7 and ov["grid"]["enabled"] is False
    ov["lane_lines"] = [{"x1": 0.1, "y1": 0, "x2": 0.12, "y2": 1}, {"x1": 0.5, "y1": 0, "x2": 0.5, "y2": 1}]
    ov["grid"] = {"enabled": True, "cols": 4, "rows": 3}
    ov["show_detections"] = True
    assert client.put("/api/capture/overlays", json=ov).status_code == 200
    assert empty_project.load_config()["capture"]["overlays"]["grid"]["cols"] == 4
    again = client.get("/api/capture/overlays").json()
    assert len(again["lane_lines"]) == 2 and again["show_detections"] is True
    assert client.put("/api/capture/overlays", json={"lane_lines": [{"x1": 2, "y1": 0, "x2": 0, "y2": 1}]}).status_code == 422
    assert client.get("/api/capture/options").json() == {"min_free_gb": 2.0, "auto_sync": True}


def test_mjpeg_local_stream(tmp_path):
    """Flux MJPEG d'une source locale, à travers un vrai serveur (TestClient ne gère pas les flux infinis)."""
    import threading

    import httpx
    import uvicorn
    from eggvision.app import create_app
    from eggvision_agent.testing import free_port

    port = free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning", timeout_graceful_shutdown=1))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    _wait(lambda: server.started)
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10) as c:
            assert c.post("/api/capture/connect", json={"kind": "simulated", "scale": 0.25}).status_code == 200
            _wait(lambda: c.get("/api/capture/status").json()["source"]["actual_fps"] > 0)
            frames, buf = 0, b""
            with c.stream("GET", "/api/capture/preview.mjpeg", params={"fps": 20, "width": 100}) as r:
                assert r.headers["content-type"].startswith("multipart/x-mixed-replace")
                for chunk in r.iter_bytes():
                    buf += chunk
                    frames = buf.count(b"Content-Type: image/jpeg")
                    if frames >= 3:
                        break
            assert frames >= 3
            c.post("/api/capture/disconnect")
    finally:
        server.should_exit = True
        t.join(10)


def test_diagnostics_without_agent(client):
    diag = client.get("/api/diagnostics").json()["agent"]
    assert diag["connected"] is False and "Aucun agent" in diag["detail"]
    _connect_sim(client)
    assert "source active" in client.get("/api/diagnostics").json()["agent"]["detail"]
    assert {s["kind"] for s in client.get("/api/capture/sources").json()} == {"pi", "usb", "simulated"}
    assert client.post("/api/capture/connect", json={"kind": "inconnu"}).status_code == 400
    assert client.post("/api/capture/connect", json={"kind": "pi", "url": "127.0.0.1:1"}).status_code in (503, 504)
    assert client.get("/api/capture/agent/lots").status_code == 409  # pas d'agent


def test_analysis_unit():
    img = np.zeros((100, 200, 3), np.uint8)
    img[:, :100] = 255
    a = analyze_image(img, bins=16)
    assert a["saturated"]["any"] == pytest.approx(0.5) and a["saturation_warning"]
    assert a["histogram"]["luma"][0] == pytest.approx(0.5) and a["histogram"]["luma"][-1] == pytest.approx(0.5)
    assert a["dark_ratio"] == pytest.approx(0.5) and a["sharpness"] > 0
    flat = analyze_image(np.full((50, 50, 3), 128, np.uint8))
    assert flat["sharpness"] == 0 and not flat["saturation_warning"]
    dets, error, size = detect_eggs(img, {"image_width": 400, "image_height": 200})
    assert size == (400, 200)
    assert (dets is None and error) or isinstance(dets, list)  # module de détection facultatif


def test_normalize_agent_url():
    assert normalize_agent_url("192.168.4.1") == "http://192.168.4.1:8770"
    assert normalize_agent_url(" eggpi.local:9000/ ") == "http://eggpi.local:9000"
    assert normalize_agent_url("https://10.0.0.2:8770/api") == "https://10.0.0.2:8770"
    with pytest.raises(SourceError):
        normalize_agent_url("")


class FakeCapture:
    """Caméra UVC factice (OpenCV ``VideoCapture``)."""

    def __init__(self, device):
        self.device = device
        self.opened = True
        self.props: dict[int, float] = {}
        self.i = 0

    def isOpened(self):  # noqa: N802 — API OpenCV
        return self.opened

    def getBackendName(self):  # noqa: N802
        return "V4L2"

    def set(self, prop, value):
        self.props[prop] = value
        return prop != cv2.CAP_PROP_AUTO_WB  # ce pilote refuse de couper l'AWB

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH: 320, cv2.CAP_PROP_FRAME_HEIGHT: 240}.get(prop, self.props.get(prop, 0))

    def read(self):
        time.sleep(1 / 60)
        img = np.full((240, 320, 3), (self.i * 5) % 200, np.uint8)
        img[:, :, 2] = 150
        self.i += 1
        return True, img

    def release(self):
        self.opened = False


def test_usb_source_with_fake_camera(tmp_path):
    src = UsbCameraSource(0, capture_factory=FakeCapture)
    src.open()
    try:
        _wait(lambda: src.ready)
        cap = src._cap
        s = src.set_settings({"exposure_us": 800, "analogue_gain": 2.0})
        assert s.exposure_us == 800
        assert cap.props[cv2.CAP_PROP_EXPOSURE] == 8.0  # V4L2 : unités de 100 µs
        assert cap.props[cv2.CAP_PROP_AUTO_EXPOSURE] == 1 and cap.props[cv2.CAP_PROP_GAIN] == 10.0
        st = src.status()
        assert st["camera"]["controls"]["exposure"] and not st["camera"]["controls"]["auto_wb_off"]
        assert st["camera"]["output_width"] == 320 and st["kind"] == "usb"
        red_before = src.preview_frame()[:, :, 2].mean()
        src.set_settings({"red_gain": 1.0})  # gains couleur logiciels
        time.sleep(0.1)
        assert src.preview_frame()[:, :, 2].mean() < red_before * 0.7
        manifest = LotManifest(lot_id="lot-usb", name="USB", type="production")
        finished = []
        src.on_lot_finished = lambda lot_id, man, reason: finished.append((lot_id, man["status"], reason))
        src.start_lot(manifest, tmp_path / "lot-usb", 0)
        with pytest.raises(SourceError) as e:
            src.set_settings({"exposure_us": 900})
        assert e.value.status == 409
        time.sleep(0.4)
        out = src.stop_lot()
        assert out["manifest"]["status"] == "termine" and finished == [("lot-usb", "termine", "user")]
        metas = read_metadata(tmp_path / "lot-usb")
        assert len(metas) == out["manifest"]["stats"]["frames_written"] > 5
        assert out["manifest"]["source"]["kind"] == "usb" and out["manifest"]["width"] == 320
    finally:
        src.close()
    assert not cap.opened
