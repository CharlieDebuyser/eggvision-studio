"""Tests de l'export ONNX, du paquet de déploiement et du banc de mesure (local + faux agent Pi)."""

from __future__ import annotations

import io
import json
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
from eggvision import state
from eggvision.db import models as m
from eggvision.ml.testing import copy_project, trained_template
from eggvision.testing import wait_job
from fastapi import FastAPI, File, Form, UploadFile

FAKE_AGENT_PORT = 8814


@pytest.fixture(scope="module")
def exported_root(demo_template, tmp_path_factory):
    """Projet entraîné + exporté une seule fois pour le module."""
    from eggvision.ml.export import start_export
    from eggvision.project import Project

    root = copy_project(trained_template(demo_template, tmp_path_factory.mktemp("ml")), tmp_path_factory.mktemp("exp"))
    project = Project(root)
    try:
        with project.session() as s:
            model_id = s.query(m.Model.id).one()[0]
        job = state.jobs.wait(start_export(project, model_id)["job_id"], 600)
        assert job["status"] == "done", job["error"]
    finally:
        project.close()
    return root


@pytest.fixture()
def exported(client, exported_root, tmp_path):
    project = state.projects.open(copy_project(exported_root, tmp_path))
    with project.session() as s:
        row = s.query(m.Model).one()
        return project, row.id, dict(row.export)


def test_export_api_and_package(client, exported):
    project, model_id, export = exported
    status = client.get(f"/api/export/models/{model_id}").json()
    assert status["exported"] and not status["outdated"] and status["exportable"]
    parity = export["parity"]["fp32"]
    print(f"\nParité PyTorch ↔ ONNX FP32 : {parity['max_abs_diff']:.2e} ; INT8 vs FP32 : {export['parity']['int8'].get('vs_fp32')}")
    assert parity["ok"] and parity["max_abs_diff"] <= 1e-4
    assert set(parity["by_n_views"]) >= {"6", "12"}  # axes dynamiques : V = 6 et 12
    assert export["variants"] == ["fp32", "int8"] and export["version"] == "1.1"

    pkg = project.abspath(export["package_dir"])
    names = {p.relative_to(pkg).as_posix() for p in pkg.rglob("*") if p.is_file()}
    assert {"inference_config.json", "model_fp32.onnx", "model_int8.onnx", "preprocess.py", "infer.py", "README.txt", "reference/expected.json"} <= names
    from eggvision_shared import source_file

    assert (pkg / "infer.py").read_bytes() == Path(source_file("runtime.py")).read_bytes()
    assert (pkg / "preprocess.py").read_bytes() == Path(source_file("preprocess.py")).read_bytes()
    cfg = json.loads((pkg / "inference_config.json").read_text(encoding="utf-8"))
    assert cfg["format_version"] == 1 and cfg["model_id"] == model_id and cfg["n_views"] == 6 and cfg["n_lanes"] == 6
    assert cfg["preprocess"]["input_size"] == 96 and cfg["threshold"] == export["threshold"]
    assert cfg["scene"] == project.get_config("scene") and cfg["metrics"]["test"]["auc_roc"] is not None
    expected = json.loads((pkg / "reference" / "expected.json").read_text(encoding="utf-8"))
    assert len(expected["eggs"]) == 6 and len(expected["eggs"][0]["view_scores"]) == 6

    # ONNX : noms et axes dynamiques
    import onnx

    proto = onnx.load(str(pkg / "model_fp32.onnx"))
    assert [i.name for i in proto.graph.input] == ["views"]
    assert [o.name for o in proto.graph.output] == ["egg_scores", "view_scores"]
    dims = proto.graph.input[0].type.tensor_type.shape.dim
    assert dims[0].dim_param and dims[1].dim_param and proto.opset_import[0].version >= 17

    # zip téléchargeable et identique au dossier
    r = client.get(f"/api/export/models/{model_id}/download")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        assert set(zf.namelist()) == names


def test_package_runtime_parity_and_dynamic_views(exported):
    """Critère 4 : mêmes scores en PyTorch et en ONNX (V = 6 et 12, vues réelles du projet)."""
    import torch
    from eggvision.ml.dataset import load_records
    from eggvision.ml.store import load_classifier
    from eggvision_shared.preprocess import PreprocessConfig, load_view, preprocess_views
    from eggvision_shared.runtime import EggClassifier

    project, model_id, export = exported
    pkg = str(project.abspath(export["package_dir"]))
    fp32 = EggClassifier.load(pkg, variant="fp32")
    int8 = EggClassifier.load(pkg, variant="int8")
    model = load_classifier(project, model_id)
    with project.session() as s:
        row = s.query(m.Model).one()
        a = s.get(m.DatasetSplit, row.split_id).assignments
    recs = [r for r in load_records(project, a) if r.subset == "test"][:8]
    for v in (6, 12):
        eggs = [[load_view(p) for p in r.view_paths[:v]] for r in recs]
        scores, views = fp32.score_eggs(eggs)
        x = torch.from_numpy(np.stack([preprocess_views(e, PreprocessConfig(input_size=96)) for e in eggs]))
        with torch.no_grad():
            te, tv = model(x)
        assert np.max(np.abs(np.asarray(scores) - te.numpy())) <= 1e-4
        assert np.max(np.abs(np.asarray(views) - tv.numpy())) <= 1e-4
        assert len(views[0]) == v
        q_scores, _ = int8.score_eggs(eggs)
        assert all(0.0 <= q <= 1.0 for q in q_scores)
    # décision par colonne : 6 couloirs, dont un vide
    col = [[load_view(p) for p in r.view_paths[:6]] for r in recs[:5]] + [None]
    res = fp32.predict_column(col)
    assert len(res.decisions) == 6 and res.decisions[5] == 0 and res.scores[5] is None
    assert res.decisions[:5] == [int(s >= fp32.threshold) for s in res.scores[:5]]
    assert fp32.verify_reference(pkg)["ok"]


def test_infer_verify_subprocess_from_unzipped_package(exported, tmp_path):
    project, _, export = exported
    target = tmp_path / "pi"
    with zipfile.ZipFile(project.abspath(export["zip"])) as zf:
        zf.extractall(target)
    proc = subprocess.run([sys.executable, "-I", "infer.py", "--verify"], cwd=target, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["ok"] and report["max_abs_diff"] <= 1e-4
    proc = subprocess.run([sys.executable, "-I", "infer.py", "--bench", "3", "--variant", "int8"], cwd=target, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0 and json.loads(proc.stdout)["variant"] == "int8"


def test_threshold_change_marks_export_outdated(client, exported):
    _, model_id, _ = exported
    client.put(f"/api/training/models/{model_id}/threshold", json={"threshold": 0.4321})
    assert client.get(f"/api/export/models/{model_id}").json()["outdated"]
    assert client.get(f"/api/training/models/{model_id}").json()["export_outdated"]


def test_bench_local(client, exported):
    _, model_id, _ = exported
    r = client.post(f"/api/export/models/{model_id}/bench", json={"target": "local", "variant": "int8", "n_columns": 5, "threads": 2})
    assert r.status_code == 200, r.text
    wait_job(r.json()["job_id"], 300)
    r = client.post(f"/api/export/models/{model_id}/bench", json={"target": "local", "variant": "fp32", "n_columns": 5})
    wait_job(r.json()["job_id"], 300)
    items = client.get(f"/api/export/models/{model_id}/bench").json()["items"]
    assert [i["variant"] for i in items] == ["fp32", "int8"]
    res = items[0]["result"]
    assert res["n_lanes"] == 6 and res["n_views"] == 6 and res["latency_ms"]["mean"] > 0
    assert res["parity"]["ok"] and "onnxruntime" in res and res["device"]["machine"] == platform.machine()
    assert client.post(f"/api/export/models/{model_id}/bench", json={"variant": "int16"}).status_code == 422


def _fake_agent_app():
    """Faux agent Raspberry Pi respectant le contrat de POST /api/bench."""
    from eggvision_shared.runtime import EggClassifier

    app = FastAPI()
    app.state.calls = []

    @app.post("/api/bench")
    async def bench(package: UploadFile = File(...), variant: str = Form("fp32"), n_columns: int = Form(50), threads: int | None = Form(None)):
        import onnxruntime

        data = await package.read()
        app.state.calls.append({"filename": package.filename, "size": len(data), "variant": variant, "n_columns": n_columns, "threads": threads})
        d = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                zf.extractall(d)
            clf = EggClassifier.load(d, variant=variant, threads=threads or None)
            out = clf.benchmark(n_columns=n_columns)
            out["parity"] = clf.verify_reference(d)
            out["device"] = {"model": "Raspberry Pi 5 (factice)"}
            out["onnxruntime"] = onnxruntime.__version__
            return out
        finally:
            shutil.rmtree(d, ignore_errors=True)

    return app


@pytest.fixture()
def fake_agent():
    import uvicorn

    app = _fake_agent_app()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=FAKE_AGENT_PORT, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    t0 = time.time()
    while not server.started and time.time() - t0 < 10:
        time.sleep(0.05)
    assert server.started
    yield app, f"http://127.0.0.1:{FAKE_AGENT_PORT}"
    server.should_exit = True
    thread.join(5)


def test_bench_pi_with_fake_agent(client, exported, fake_agent, monkeypatch):
    _, model_id, export = exported
    app, url = fake_agent
    # aucun agent connecté → erreur claire
    monkeypatch.setattr(state, "agent_url", None)
    r = client.post(f"/api/export/models/{model_id}/bench", json={"target": "pi"})
    assert r.status_code == 400 and "agent" in r.json()["detail"].lower()

    monkeypatch.setattr(state, "agent_url", url)
    r = client.post(f"/api/export/models/{model_id}/bench", json={"target": "pi", "variant": "fp32", "n_columns": 4, "threads": 2})
    assert r.status_code == 200, r.text
    wait_job(r.json()["job_id"], 300)
    call = app.state.calls[-1]
    assert call["variant"] == "fp32" and call["n_columns"] == 4 and call["threads"] == 2 and call["filename"].endswith(".zip")
    item = client.get(f"/api/export/models/{model_id}/bench").json()["items"][0]
    assert item["target"] == "pi" and item["agent_url"] == url
    assert item["result"]["parity"]["ok"] and item["result"]["device"]["model"].startswith("Raspberry")

    # agent injoignable → tâche en échec avec un message explicite
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        dead = f"http://127.0.0.1:{sock.getsockname()[1]}"
    r = client.post(f"/api/export/models/{model_id}/bench", json={"target": "pi", "agent_url": dead, "n_columns": 1})
    job = state.jobs.wait(r.json()["job_id"], 60)
    assert job["status"] == "failed" and "injoignable" in job["error"]


def test_export_requires_trained_model(client, project):
    with project.session() as s:
        s.add(m.Model(id="brouillon", name="brouillon", status="created"))
    r = client.post("/api/export/models/brouillon")
    assert r.status_code == 400 and "entraîné" in r.json()["detail"]
    assert client.post("/api/export/models/inconnu").status_code == 404
    assert client.get("/api/export/models/brouillon/download").status_code == 404
    r = client.post("/api/export/models/brouillon/bench", json={"target": "local"})
    assert r.status_code == 400 and "exporté" in r.json()["detail"]


def test_bench_pi_with_real_simulated_agent(client, exported, tmp_path, monkeypatch):
    """Banc contre le VRAI agent (``eggvision_agent``, caméra simulée) : contrat POST /api/bench de bout en bout."""
    agent_testing = pytest.importorskip("eggvision_agent.testing")
    _, model_id, _ = exported
    with agent_testing.running_agent(tmp_path / "agent") as (url, _service):
        monkeypatch.setattr(state, "agent_url", url)
        for body in ({"variant": "fp32", "n_columns": 3}, {"variant": "int8", "n_columns": 2, "threads": 2}):
            r = client.post(f"/api/export/models/{model_id}/bench", json={"target": "pi", **body})
            assert r.status_code == 200, r.text
            wait_job(r.json()["job_id"], 300)
    items = client.get(f"/api/export/models/{model_id}/bench").json()["items"]
    fp32 = next(i for i in items if i["variant"] == "fp32")["result"]
    assert fp32["parity"]["ok"] and fp32["latency_ms"]["mean"] > 0 and "temperature_c" in fp32["device"]
    assert items[0]["variant"] == "int8" and items[0]["result"]["threads"] == 2
