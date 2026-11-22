"""Tests de l'agent Pi en mode simulé (aucun matériel requis)."""

from __future__ import annotations

import io
import json
import shutil
import time
import zipfile
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest
from eggvision_agent.api import create_app
from eggvision_agent.service import AgentService
from eggvision_agent.testing import make_test_config, running_agent
from eggvision_shared.formats import LotManifest, LotWriter, read_metadata, sha256_file
from fastapi.testclient import TestClient


def _wait(cond, timeout: float = 30.0, step: float = 0.05) -> None:
    deadline = time.time() + timeout
    while not cond():
        if time.time() > deadline:
            raise TimeoutError("condition non atteinte")
        time.sleep(step)


@pytest.fixture()
def agent(tmp_path):
    cfg = make_test_config(tmp_path / "storage")
    service = AgentService(cfg)
    with TestClient(create_app(cfg, service)) as client:
        _wait(lambda: service.backend.ready and service.actual_fps() > 0)
        yield client, service


LOT = {"name": "Lot d'essai", "type": "declasse:felure", "farm": "Ferme du Val", "operator": "Alice", "comment": "test"}


def test_status(agent):
    client, _ = agent
    st = client.get("/api/status").json()
    assert st["agent"] == "eggvision-agent" and st["api_version"] == 1
    assert st["camera"]["backend"] == "simulated" and st["camera"]["ready"]
    assert st["camera"]["output_width"] == 368 and st["camera"]["output_height"] == 272
    assert st["storage"]["free_bytes"] > 0 and st["device"]["cpu_count"] >= 1
    assert st["capturing"] is False and st["capture"] is None
    assert 20 < st["camera"]["actual_fps"] < 40


def test_settings_validation_and_lock(agent):
    client, service = agent
    s = client.get("/api/camera/settings").json()
    assert s["exposure_us"] == 500 and s["fps"] == 30
    r = client.put("/api/camera/settings", json={"exposure_us": 800, "analogue_gain": 2.0})
    assert r.status_code == 200 and r.json()["exposure_us"] == 800 and r.json()["fps"] == 30
    assert client.put("/api/camera/settings", json={"analogue_gain": 99}).status_code == 422
    assert client.put("/api/camera/settings", json={"exposure_us": 50_000}).status_code == 422  # > période d'image
    assert client.put("/api/camera/settings", json={"roi": [10, 10, 4000, 100]}).status_code == 200  # rognée au capteur
    assert client.put("/api/camera/settings", json={"roi": None}).status_code == 200
    # les réglages survivent à un redémarrage
    assert AgentService(service.config)._load_settings().exposure_us == 800

    r = client.post("/api/lots", json={**LOT, "camera": {"exposure_us": 600, "fps": 30}})
    assert r.status_code == 200, r.text
    lot_id = r.json()["lot_id"]
    assert client.put("/api/camera/settings", json={"exposure_us": 900}).status_code == 409
    assert client.post("/api/camera/awb", json={"duration_s": 0}).status_code == 409
    assert client.post("/api/lots", json=LOT).status_code == 409  # une seule capture à la fois
    assert client.get(f"/api/lots/{lot_id}/files").status_code == 409
    time.sleep(0.3)
    stop = client.post("/api/lots/stop").json()
    assert stop["manifest"]["camera"]["exposure_us"] == 600  # réglages figés dans le lot
    assert client.put("/api/camera/settings", json={"exposure_us": 900}).status_code == 200


def test_awb_calibration(agent):
    client, _ = agent
    r = client.post("/api/camera/awb", json={"duration_s": 0.1})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["settings"]["red_gain"] == body["red_gain"] != 2.0
    assert client.get("/api/camera/settings").json()["blue_gain"] == body["blue_gain"]


def test_preview(agent):
    client, _ = agent
    r = client.get("/api/preview.jpg", params={"width": 160})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] == 160
    full = cv2.imdecode(np.frombuffer(client.get("/api/preview.jpg").content, np.uint8), cv2.IMREAD_COLOR)
    assert full.shape[1] == 200  # preview_width de la configuration de test


def test_capture_lot_conformant(agent):
    client, service = agent
    r = client.post("/api/lots", json={**LOT, "lot_id": "lot-essai-1", "source": {"pc": "poste-1"}})
    assert r.status_code == 200, r.text
    live = client.get("/api/capture").json()
    assert live["active"] and live["lot_id"] == "lot-essai-1"
    time.sleep(1.6)
    live = client.get("/api/status").json()["capture"]
    assert live["running"] and live["frames_received"] > 30 and live["name"] == LOT["name"]
    final = client.post("/api/lots/lot-essai-1/stop").json()
    m = final["manifest"]
    assert m["status"] == "termine" and m["type"] == "declasse:felure" and m["farm"] == "Ferme du Val"
    assert m["source"]["kind"] == "pi" and m["source"]["pc"] == "poste-1"
    assert m["stats"]["frames_dropped"] == 0 and m["stats"]["frames_written"] >= 45
    assert m["stats"]["measured_fps"] == pytest.approx(30.0, rel=0.02)
    lot_dir = service.lots_root / "lot-essai-1"
    metas = read_metadata(lot_dir)
    assert len(metas) == m["stats"]["frames_written"]
    assert [x.seq for x in metas] == list(range(len(metas)))
    assert all(x.encoder_count is None for x in metas)
    deltas = np.diff([x.sensor_ts_ns for x in metas])
    assert np.allclose(deltas, 1e9 / 30, rtol=0.01)
    assert cv2.imread(str(lot_dir / metas[0].file)).shape == (272, 368, 3)

    lots = client.get("/api/lots").json()
    assert lots["total"] == 1 and lots["items"][0]["status"] == "termine" and not lots["items"][0]["capturing"]
    assert client.get("/api/lots/lot-essai-1").json()["manifest"]["name"] == LOT["name"]
    files = client.get("/api/lots/lot-essai-1/files").json()
    assert files["n_files"] == len(metas) + 2 and files["total_bytes"] > 0
    entry = next(f for f in files["files"] if f["path"] == metas[3].file)
    data = client.get(f"/api/lots/lot-essai-1/files/{entry['path']}").content
    assert len(data) == entry["size"]
    assert client.get("/api/lots/lot-essai-1/files/../../agent_settings.json").status_code in (404, 400)
    assert client.get("/api/lots/inconnu/files").status_code == 404
    assert client.post("/api/lots/stop").status_code == 409
    assert client.delete("/api/lots/lot-essai-1").status_code == 200
    assert client.get("/api/lots").json()["total"] == 0


def test_bad_requests(agent):
    client, _ = agent
    assert client.post("/api/lots", json={**LOT, "type": "inconnu"}).status_code == 422
    assert client.post("/api/lots", json={**LOT, "lot_id": "../evil"}).status_code == 400
    assert client.post("/api/lots", json={**LOT, "min_free_bytes": 10**18}).status_code == 507
    assert not client.get("/api/status").json()["capturing"]


def test_disk_low_auto_stop(agent, monkeypatch):
    client, _ = agent
    import eggvision_shared.capture as cap

    assert client.post("/api/lots", json={**LOT, "lot_id": "lot-disque", "min_free_bytes": 1000}).status_code == 200
    time.sleep(0.3)
    monkeypatch.setattr(cap, "disk_free_bytes", lambda path: 10)
    _wait(lambda: not client.get("/api/capture").json()["active"], timeout=5)
    last = client.get("/api/capture").json()["last"]
    assert last["lot_id"] == "lot-disque" and last["stop_reason"] == "disk_low" and last["status"] == "termine"


def test_interrupted_lots(tmp_path):
    storage = tmp_path / "storage"
    # 1) arrêt brutal : lot resté « en_cours » sur disque → « interrompu » au redémarrage
    writer = LotWriter(storage / "lots" / "lot-crash", LotManifest(lot_id="lot-crash", name="x", type="viable"))
    for i in range(3):
        writer.write_frame(np.zeros((32, 32, 3), np.uint8), i * 33_333_333, seq=i)
    writer._meta.flush()  # le processus « meurt » sans close()
    cfg = make_test_config(storage)
    service = AgentService(cfg)
    assert service.interrupted_at_start == ["lot-crash"]
    m = LotManifest.load(storage / "lots" / "lot-crash")
    assert m.status == "interrompu" and m.stats.frames_written == 3
    # 2) arrêt du service pendant une capture → lot clos proprement, « interrompu »
    with TestClient(create_app(cfg, service)) as client:
        _wait(lambda: service.backend.ready)
        assert client.post("/api/lots", json={**LOT, "lot_id": "lot-arret"}).status_code == 200
        time.sleep(0.4)
    m = LotManifest.load(storage / "lots" / "lot-arret")
    assert m.status == "interrompu" and m.stats.frames_written > 3 and m.ended_at
    assert len(read_metadata(storage / "lots" / "lot-arret")) == m.stats.frames_written


def test_real_server_range_and_mjpeg(tmp_path):
    with running_agent(tmp_path / "storage") as (url, service):
        with httpx.Client(base_url=url, timeout=10) as c:
            assert c.post("/api/lots", json={**LOT, "lot_id": "lot-reseau"}).status_code == 200
            time.sleep(0.5)
            assert c.post("/api/lots/stop").status_code == 200
            files = c.get("/api/lots/lot-reseau/files").json()["files"]
            f = next(x for x in files if x["path"].startswith("frames/"))
            full = c.get(f"/api/lots/lot-reseau/files/{f['path']}")
            assert full.status_code == 200 and full.headers["accept-ranges"] == "bytes"
            part = c.get(f"/api/lots/lot-reseau/files/{f['path']}", headers={"Range": "bytes=100-"})
            assert part.status_code == 206 and part.content == full.content[100:]
            assert part.headers["content-range"] == f"bytes 100-{f['size'] - 1}/{f['size']}"
            mid = c.get(f"/api/lots/lot-reseau/files/{f['path']}", headers={"Range": "bytes=10-19"})
            assert mid.content == full.content[10:20]
            assert c.get(f"/api/lots/lot-reseau/files/{f['path']}", headers={"Range": f"bytes={f['size']}-"}).status_code == 416
            import hashlib

            assert hashlib.sha256(full.content).hexdigest() == f["sha256"] == sha256_file(service.lots_root / "lot-reseau" / f["path"])
            # flux MJPEG : au moins deux images distinctes
            frames = 0
            with c.stream("GET", "/api/preview.mjpeg", params={"fps": 20, "width": 120}) as r:
                assert r.headers["content-type"].startswith("multipart/x-mixed-replace")
                buf = b""
                for chunk in r.iter_bytes():
                    buf += chunk
                    frames = buf.count(b"Content-Type: image/jpeg")
                    if frames >= 2:
                        break
            assert frames >= 2


# ---------------------------------------------------------------------------
# Banc de mesure
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def package_zip(tmp_path_factory) -> bytes:
    pytest.importorskip("torch")
    from eggvision.testing import make_dummy_package
    from eggvision_shared.runtime import EggClassifier

    root = tmp_path_factory.mktemp("pkg")
    pkg = make_dummy_package(root / "package", n_views=4, n_lanes=2, input_size=32)
    # référence de parité : scores attendus calculés une fois
    clf = EggClassifier.load(str(pkg))
    rng = np.random.default_rng(0)
    eggs = [[rng.integers(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(4)] for _ in range(2)]
    scores, views = clf.score_eggs(eggs)
    ref = pkg / "reference"
    items = []
    for i, egg in enumerate(eggs):
        (ref / f"egg_{i:02d}").mkdir(parents=True)
        for k, v in enumerate(egg):
            cv2.imwrite(str(ref / f"egg_{i:02d}" / f"{k:02d}.png"), v)
        items.append({"dir": f"egg_{i:02d}", "score": scores[i], "view_scores": views[i]})
    (ref / "expected.json").write_text(json.dumps({"tolerance": 1e-4, "eggs": items}))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for p in pkg.rglob("*"):
            if p.is_file():
                zf.write(p, Path("package") / p.relative_to(pkg))  # paquet dans un sous-dossier
    shutil.rmtree(root)
    return buf.getvalue()


def test_bench(agent, package_zip, monkeypatch):
    client, _ = agent
    files = {"package": ("paquet.zip", package_zip, "application/zip")}
    r = client.post("/api/bench", files=files, data={"variant": "fp32", "n_columns": "5", "threads": "1"})
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["n_columns"] == 5 and res["n_lanes"] == 2 and res["n_views"] == 4 and res["threads"] == 1
    assert res["latency_ms"]["mean"] > 0 and res["variant"] == "fp32"
    assert res["parity"]["ok"] is True and res["parity"]["n_eggs"] == 2
    assert set(res["device"]) == {"model", "cpu_count", "temperature_c"} and res["onnxruntime"]
    # erreurs
    assert client.post("/api/bench", files=files, data={"variant": "int8"}).status_code == 422  # variante absente
    bad = {"package": ("x.zip", b"pas un zip", "application/zip")}
    assert client.post("/api/bench", files=bad).status_code == 400
    assert client.post("/api/lots", json=LOT).status_code == 200
    assert client.post("/api/bench", files=files).status_code == 409  # pendant une capture
    client.post("/api/lots/stop")
    import eggvision_agent.bench as bench

    monkeypatch.setattr(bench, "onnxruntime_version", lambda: None)
    r = client.post("/api/bench", files=files)
    assert r.status_code == 501 and "ONNX Runtime" in r.json()["detail"]
