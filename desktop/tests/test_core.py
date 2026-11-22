from __future__ import annotations

import time

from eggvision import state
from eggvision.db import models as m
from eggvision.jobs import JobContext
from eggvision.testing import wait_job


def test_health_and_no_project(client):
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get("/api/project").json() == {"open": False}
    assert client.get("/api/project/config").status_code == 409
    diag = client.get("/api/diagnostics").json()
    assert diag["device"]["kind"] in ("cuda", "mps", "cpu")
    assert "torch" in diag["packages"]


def test_create_open_close_project(client, tmp_path):
    r = client.post("/api/project/create", json={"parent_dir": str(tmp_path), "name": "Essai"})
    assert r.status_code == 200, r.text
    info = r.json()
    assert info["open"] and info["name"] == "Essai" and info["stats"]["lots"] == 0
    assert client.post("/api/project/create", json={"parent_dir": str(tmp_path), "name": "Essai"}).status_code == 400
    cfg = client.put("/api/project/config", json={"user": {"name": "Alice"}}).json()
    assert cfg["user"]["name"] == "Alice" and cfg["project"]["name"] == "Essai"
    assert client.post("/api/project/close").json() == {"open": False}
    assert client.post("/api/project/open", json={"path": str(tmp_path / "Essai")}).json()["open"]
    listing = client.get("/api/fs/list", params={"path": str(tmp_path)}).json()
    assert [e["is_project"] for e in listing["entries"] if e["name"] == "Essai"] == [True]
    assert any(r["path"].endswith("Essai") for r in client.get("/api/project/recent").json())


def test_demo_project_content(project):
    stats = project.stats()
    assert stats["lots"] == 22 and stats["tracks"] == 3 * (24 + 36) + 18
    with project.session() as s:
        track = s.query(m.Track).filter(m.Track.lot_id.like("%")).first()
        assert track.n_views == 12 and len(track.views) == 12
        assert project.abspath(track.views[0].path).is_file()
        lot = s.get(m.Lot, track.lot_id)
        assert track.label is not None and lot.session_date
        prod = s.query(m.Lot).filter(m.Lot.type == "production").one()
        labels = s.query(m.Label).join(m.Track).filter(m.Track.lot_id == prod.id).all()
        assert labels and all(lab.verdict is None for lab in labels)


def test_media(client, project, frames_template):
    with project.session() as s:
        track_id = s.query(m.Track.id).first()[0]
    r = client.get(f"/api/media/crop/{track_id}/0")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert client.get(f"/api/media/crop/{track_id}/99").status_code == 404


def test_frames_media(client, frames_project):
    r = client.get("/api/media/frame/lot-viable/3", params={"w": 200})
    assert r.status_code == 200 and len(r.content) > 1000
    assert frames_project.stats()["frames"] > 100


def test_jobs_progress_cancel_log(client, project):
    def work(ctx: JobContext):
        for i in range(200):
            ctx.check_cancelled()
            ctx.progress(i / 200, f"étape {i}")
            ctx.publish(step=i)
            time.sleep(0.01)
        return {"ok": True}

    job = state.jobs.submit("test", "Tâche de test", work, project=project)
    time.sleep(0.15)
    live = client.get(f"/api/jobs/{job.id}").json()
    assert live["status"] == "running" and live["result"]["step"] >= 1
    assert client.post(f"/api/jobs/{job.id}/cancel").status_code == 200
    final = state.jobs.wait(job.id, 10)
    assert final["status"] == "cancelled"
    assert "Annulé" in client.get(f"/api/jobs/{job.id}/log").json()["text"]

    quick = state.jobs.submit("test", "Rapide", lambda ctx: {"n": 3}, project=project)
    assert wait_job(quick.id)["result"] == {"n": 3}
    failing = state.jobs.submit("test", "Échec", lambda ctx: 1 / 0, project=project)
    assert state.jobs.wait(failing.id, 10)["status"] == "failed"
    listed = client.get("/api/jobs", params={"kind": "test"}).json()
    assert {j["id"] for j in listed} >= {job.id, quick.id, failing.id}


def test_job_groups_run_sequentially(client):
    order: list[str] = []

    def make(name: str):
        def work(ctx: JobContext):
            order.append(f"{name}:start")
            time.sleep(0.2)
            order.append(f"{name}:end")
        return work

    a = state.jobs.submit("test", "A", make("a"), group="g")
    b = state.jobs.submit("test", "B", make("b"), group="g")
    state.jobs.wait(a.id, 10)
    state.jobs.wait(b.id, 10)
    assert order in (["a:start", "a:end", "b:start", "b:end"], ["b:start", "b:end", "a:start", "a:end"])


def test_demo_job(client, tmp_path):
    from eggvision import demo

    demo.DEMO_SIZES["test"] = {"viable_columns": 1, "defect_columns": 1, "production_columns": 0, "scale": 0.25}
    r = client.post("/api/project/demo", json={"parent_dir": str(tmp_path), "name": "Demo", "size": "test"})
    assert r.status_code == 200, r.text
    job = wait_job(r.json()["job_id"])
    info = client.post("/api/project/open", json={"path": job["result"]["path"]}).json()
    assert info["stats"]["lots"] == 21 and info["stats"]["frames"] > 0 and info["stats"]["tracks"] > 0
