"""Prise de vue avec l'agent Pi (agent réel en processus, caméra simulée) et synchronisation."""

from __future__ import annotations

import hashlib
import time

import httpx
import pytest
from eggvision import state
from eggvision.capture.pi import PiAgentSource
from eggvision.capture.sync import STATE_FILE, SyncError, sync_lot, verify_lot_dir
from eggvision.db import models as m
from eggvision.testing import wait_job
from eggvision_agent.testing import running_agent
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
def agent(tmp_path):
    with running_agent(tmp_path / "pi-storage") as (url, service):
        yield url, service


@pytest.fixture()
def empty_project(client, tmp_path):
    (tmp_path / "pc").mkdir()
    yield state.projects.create(tmp_path / "pc", "Essai synchro")


LOT = {"name": "Déclassés fêlure", "type": "declasse:felure", "farm": "Élevage des Chênes", "operator": "Alice"}


def _agent_lot(url: str, lot_id: str, seconds: float = 0.6) -> list[dict]:
    """Capture un lot directement sur l'agent et retourne son manifeste de fichiers."""
    with httpx.Client(base_url=url, timeout=10) as c:
        assert c.post("/api/lots", json={**LOT, "lot_id": lot_id}).status_code == 200
        time.sleep(seconds)
        assert c.post("/api/lots/stop").status_code == 200
        return c.get(f"/api/lots/{lot_id}/files").json()["files"]


# ---------------------------------------------------------------------------
# PiAgentSource contre un agent réel
# ---------------------------------------------------------------------------
def test_pi_agent_source(agent, tmp_path):
    url, service = agent
    src = PiAgentSource(url)
    src.open()
    try:
        st = src.status()
        assert st["connected"] and st["kind"] == "pi" and st["version"] and st["ready"]
        assert st["camera"]["backend"] == "simulated" and st["storage"]["free_bytes"] > 0
        assert abs(st["clock_offset_s"]) < 5
        assert src.set_settings({"exposure_us": 750}).exposure_us == 750
        assert src.get_settings().exposure_us == 750
        awb = src.calibrate_awb(0.05)
        assert awb["red_gain"] != 2.0
        assert src.preview_frame(120).shape[1] == 120
        assert src.output_size() == (368, 272)
        man = LotManifest(lot_id="lot-src", name="Source", type="viable", source={"pc": "test"})
        man.camera = src.get_settings()
        out = src.start_lot(man, None, 0)
        assert out["lot_id"] == "lot-src" and src.capturing
        with pytest.raises(Exception) as e:
            src.set_settings({"exposure_us": 900})
        assert getattr(e.value, "status", None) == 409
        time.sleep(0.4)
        assert src.capture_stats()["active"]
        final = src.stop_lot()
        assert final["manifest"]["status"] == "termine" and final["manifest"]["source"]["pc"] == "test"
        assert not src.capturing
        assert [x["lot_id"] for x in src.list_lots()["items"]] == ["lot-src"]
    finally:
        src.close()


# ---------------------------------------------------------------------------
# Synchronisation : coupure, corruption, annulation, reprise
# ---------------------------------------------------------------------------
class FaultyTransport(httpx.BaseTransport):
    """Transport HTTP qui coupe ou corrompt certains téléchargements (une fois par fichier)."""

    def __init__(self, cut: set[str] = frozenset(), corrupt: set[str] = frozenset()) -> None:
        self.inner = httpx.HTTPTransport()
        self.cut = set(cut)
        self.corrupt = set(corrupt)
        self.ranges: dict[str, list[str]] = {}

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        resp = self.inner.handle_request(request)
        path = request.url.path
        if "/files/" not in path:
            return resp
        rel = path.split("/files/", 1)[1]
        self.ranges.setdefault(rel, []).append(request.headers.get("range", ""))
        if rel in self.cut or rel in self.corrupt:
            data = resp.read()
            resp.close()
            headers = [(k, v) for k, v in resp.headers.items() if k.lower() != "content-length"]
            if rel in self.cut:
                self.cut.discard(rel)
                half = data[: len(data) // 2]

                class _Cut(httpx.SyncByteStream):
                    def __iter__(self):
                        yield half
                        raise httpx.ReadError("coupure réseau simulée")

                return httpx.Response(resp.status_code, headers=headers, stream=_Cut(), request=request)
            self.corrupt.discard(rel)
            bad = bytearray(data)
            bad[len(bad) // 3] ^= 0xFF
            return httpx.Response(resp.status_code, headers=headers, content=bytes(bad), request=request)
        return resp


def test_sync_with_cut_and_corruption(agent, tmp_path):
    url, service = agent
    files = _agent_lot(url, "lot-coupure")
    frames = [f["path"] for f in files if f["path"].startswith("frames/")]
    transport = FaultyTransport(cut={frames[2], frames[5], "metadata.jsonl"}, corrupt={frames[7]})
    dest = tmp_path / "pc" / "lot-coupure"
    logs: list[str] = []
    progress: list[dict] = []
    res = sync_lot(url, "lot-coupure", dest, transport=transport, backoff_s=0.01, log_fn=logs.append, progress=progress.append)
    assert res["files"] == len(files) and res["retries"] == 3 and res["corrupted"] == 1 and res["verified"]
    assert verify_lot_dir(dest, files) == []
    # reprise après coupure : la seconde requête demande la suite du fichier (Range)
    assert transport.ranges[frames[2]][0] == "" and transport.ranges[frames[2]][1].startswith("bytes=")
    assert transport.ranges[frames[7]] == ["", ""]  # fichier corrompu : retéléchargé en entier
    assert any("somme de contrôle" in line for line in logs) and any("coupure" in line for line in logs)
    assert progress[-1]["bytes_done"] == progress[-1]["bytes_total"] == res["bytes"]
    assert not list(dest.rglob("*.part"))
    # resynchronisation : rien n'est retéléchargé
    again = sync_lot(url, "lot-coupure", dest)
    assert again["bytes_downloaded"] == 0 and again["files_skipped"] == len(files)
    # un fichier local abîmé est détecté et remplacé
    victim = dest / frames[1]
    data = bytearray(victim.read_bytes())
    data[10] ^= 0xFF
    victim.write_bytes(bytes(data))
    fixed = sync_lot(url, "lot-coupure", dest)
    assert fixed["bytes_downloaded"] == victim.stat().st_size and verify_lot_dir(dest, files) == []


def test_sync_cancel_then_resume(agent, tmp_path):
    url, _ = agent
    files = _agent_lot(url, "lot-reprise", seconds=1.0)
    dest = tmp_path / "pc" / "lot-reprise"

    class Cancel(Exception):
        pass

    calls = {"n": 0}

    def check():
        calls["n"] += 1
        if calls["n"] > 40:
            raise Cancel()

    with pytest.raises(Cancel):
        sync_lot(url, "lot-reprise", dest, check_cancelled=check, concurrency=2)
    assert not (dest / "lot.json").exists()  # le manifeste n'arrive qu'en dernier
    done_before = sum(1 for f in files if (dest / f["path"]).is_file())
    assert 0 < done_before < len(files)
    assert (dest / STATE_FILE).is_file()
    res = sync_lot(url, "lot-reprise", dest)
    assert res["files_skipped"] == done_before
    assert res["bytes_downloaded"] < sum(f["size"] for f in files)
    assert verify_lot_dir(dest, files) == []


def test_sync_detects_corrupted_source_file(agent, tmp_path):
    url, service = agent
    files = _agent_lot(url, "lot-abime")
    f = next(x for x in files if x["path"].startswith("frames/"))
    path = service.lots_root / "lot-abime" / f["path"]
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF  # même taille, contenu altéré sur le SSD du Pi
    path.write_bytes(bytes(data))
    with pytest.raises(SyncError) as e:
        sync_lot(url, "lot-abime", tmp_path / "pc" / "lot-abime", retries=2, backoff_s=0.01)
    assert f["path"] in str(e.value) and "somme de contrôle" in str(e.value)


# ---------------------------------------------------------------------------
# Parcours complet par l'API de l'application
# ---------------------------------------------------------------------------
def test_api_pi_capture_and_auto_sync(client, empty_project, agent):
    url, service = agent
    r = client.post("/api/capture/connect", json={"kind": "pi", "url": url})
    assert r.status_code == 200, r.text
    assert r.json()["kind"] == "pi" and state.agent_url == url
    diag = client.get("/api/diagnostics").json()["agent"]
    assert diag["connected"] and diag["url"] == url and diag["version"] and "images/s" in diag["detail"]
    assert client.put("/api/capture/settings", json={"exposure_us": 450}).json()["exposure_us"] == 450

    live = client.post("/api/capture/lots", json=LOT).json()
    lot_id = live["lot"]["lot_id"]
    assert live["active"] and live["lot"]["kind"] == "pi"
    with empty_project.session() as s:
        lot = s.get(m.Lot, lot_id)
        assert lot.sync_status == "pending" and lot.source_kind == "pi" and lot.status == "en_cours"
    assert client.put("/api/capture/settings", json={"exposure_us": 900}).status_code == 409
    time.sleep(0.8)
    cur = client.get("/api/capture/lots/current").json()
    assert cur["stats"]["frames_written"] > 10 and cur["stats"]["lot_id"] == lot_id
    stop = client.post("/api/capture/lots/stop").json()
    job_id = stop["last"]["sync_job_id"]
    job = wait_job(job_id, timeout=60)
    assert job["kind"] == "sync" and job["result"]["verified"] and job["result"]["files"] > 10

    manifest = LotManifest.load(empty_project.lot_dir(lot_id))
    assert manifest.camera.exposure_us == 450 and manifest.source["kind"] == "pi" and manifest.farm == LOT["farm"]
    metas = read_metadata(empty_project.lot_dir(lot_id))
    with empty_project.session() as s:
        lot = s.get(m.Lot, lot_id)
        assert lot.sync_status == "done" and lot.status == "termine"
        assert lot.frame_count == len(metas) == manifest.stats.frames_written
        assert s.query(m.Frame).filter(m.Frame.lot_id == lot_id).count() == len(metas)
    for meta in metas[:5]:
        assert hashlib.sha256((empty_project.lot_dir(lot_id) / meta.file).read_bytes()).hexdigest() == meta.sha256
    assert client.get("/api/capture/sync").json()[0]["id"] == job_id

    agent_lots = client.get("/api/capture/agent/lots").json()
    assert agent_lots["unsynced"] == 0 and agent_lots["items"][0]["synced"]
    # lot présent uniquement sur l'agent : listé comme non synchronisé, puis synchronisé à la demande
    _agent_lot(url, "lot-oublie", 0.4)
    unsynced = client.get("/api/capture/agent/lots", params={"only_unsynced": True}).json()
    assert [x["lot_id"] for x in unsynced["items"]] == ["lot-oublie"] and not unsynced["items"][0]["in_project"]
    assert client.delete("/api/capture/agent/lots/lot-oublie").status_code == 409  # non synchronisé
    job = client.post("/api/capture/lots/lot-oublie/sync").json()
    wait_job(job["id"], timeout=60)
    with empty_project.session() as s:
        assert s.get(m.Lot, "lot-oublie").sync_status == "done"
    assert client.delete("/api/capture/agent/lots/lot-oublie").status_code == 200
    assert client.get("/api/capture/agent/lots").json()["total"] == 1
    client.post("/api/capture/disconnect")
    assert state.agent_url is None
    assert client.get("/api/diagnostics").json()["agent"]["url"] == url  # dernière adresse mémorisée


def test_api_agent_side_stop_detected(client, empty_project, agent):
    """Lot arrêté côté agent (ex. arrêt automatique disque) : l'application le détecte et synchronise."""
    url, service = agent
    client.post("/api/capture/connect", json={"kind": "pi", "url": url})
    lot_id = client.post("/api/capture/lots", json=LOT).json()["lot"]["lot_id"]
    time.sleep(0.5)
    with httpx.Client(base_url=url) as c:
        assert c.post("/api/lots/stop").status_code == 200
    live = _wait(lambda: (lambda x: x if not x["active"] else None)(client.get("/api/capture/lots/current").json()), timeout=10)
    assert live["last"]["lot_id"] == lot_id and live["last"]["stop_reason"] == "user"
    wait_job(live["last"]["sync_job_id"], timeout=60)
    with empty_project.session() as s:
        assert s.get(m.Lot, lot_id).sync_status == "done"


def test_api_sync_cancel(client, empty_project, agent):
    url, _ = agent
    _agent_lot(url, "lot-annule", 1.5)
    client.post("/api/capture/connect", json={"kind": "pi", "url": url})
    job = client.post("/api/capture/lots/lot-annule/sync").json()
    client.post(f"/api/jobs/{job['id']}/cancel")
    final = state.jobs.wait(job["id"], 30)
    with empty_project.session() as s:
        lot = s.get(m.Lot, "lot-annule")
        expected = "pending" if final["status"] == "cancelled" else "done"
        assert lot.sync_status == expected
    job2 = client.post("/api/capture/lots/lot-annule/sync").json()
    assert wait_job(job2["id"], timeout=60)["result"]["verified"]


def test_mdns_discovery(tmp_path):
    """Annonce mDNS de l'agent puis découverte par l'application (ignoré si le multicast est bloqué)."""
    from eggvision.capture.discovery import discover_agents

    with running_agent(tmp_path / "s", mdns=True) as (url, _):
        port = int(url.rsplit(":", 1)[1])
        found = []
        for _ in range(3):
            found = [a for a in discover_agents(1.5) if a["port"] == port]
            if found:
                break
        if not found:
            pytest.skip("multicast indisponible sur cette machine")
        assert found[0]["version"] and found[0]["url"].endswith(f":{port}") and found[0]["api"] == "1"
