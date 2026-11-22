"""Tests de l'entraînement, de l'évaluation, de Grad-CAM et du modèle d'anomalies.

Un petit modèle (MobileNetV3-Small, 96 px, 6 vues) est entraîné une fois par
session (``eggvision.ml.testing.trained_template``) ; les tests en ouvrent une copie.
"""

from __future__ import annotations

import time

import pytest
from eggvision import state
from eggvision.db import models as m
from eggvision.ml.testing import TINY_HPARAMS, copy_project, trained_template
from eggvision.testing import wait_job


@pytest.fixture(scope="module")
def trained_root(demo_template, tmp_path_factory):
    return trained_template(demo_template, tmp_path_factory.mktemp("ml"))


@pytest.fixture()
def trained(client, trained_root, tmp_path):
    project = state.projects.open(copy_project(trained_root, tmp_path))
    with project.session() as s:
        model_id = s.query(m.Model.id).filter(m.Model.kind == "classifier").one()[0]
    return project, model_id


def _wait_history(client, model_id, n, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = client.get(f"/api/training/models/{model_id}/history").json()
        if len(h["history"]) >= n:
            return h
        time.sleep(0.2)
    raise TimeoutError


def test_trained_model_learns_and_reports(client, trained):
    project, model_id = trained
    detail = client.get(f"/api/training/models/{model_id}").json()
    assert detail["status"] == "trained" and detail["kind"] == "classifier"
    assert detail["pretrained"]["loaded"] and detail["device"]["used"] in ("cuda", "mps", "cpu")
    ds = detail["data_summary"]
    assert ds["split_id"] == detail["split_id"] and ds["n_eggs"] == 180 and set(ds["counts"]) == {"train", "val", "test"}
    assert len(detail["history"]) == TINY_HPARAMS["epochs"] and detail["best_epoch"] is not None
    assert {"train_loss", "val_loss", "val_auc", "lr"} <= set(detail["history"][0])
    auc = detail["metrics"]["test"]["auc_roc"]
    print(f"\nAUC test (synthétique, 96 px, 6 époques) : {auc:.3f} ; métriques : {detail['metrics']['test']}")
    assert auc > 0.8  # hasard = 0,5
    # prédictions de toutes les pistes utilisables du découpage
    with project.session() as s:
        preds = s.query(m.Prediction).filter(m.Prediction.model_id == model_id).all()
    assert len(preds) == 180 and {p.subset for p in preds} == {"train", "val", "test"}
    assert all(len(p.view_scores) == 6 and 0 <= p.score <= 1 for p in preds)
    assert all(abs(max(p.view_scores) - p.score) < 1e-5 for p in preds)  # agrégation max
    rep = client.get(f"/api/training/models/{model_id}/report").json()
    test = rep["subsets"]["test"]
    assert test["n"] == test["n_pos"] + test["n_neg"] and len(test["recall_by_defect"]) == 6
    assert {"fpr", "tpr", "thresholds"} <= set(test["roc"]) and {"recall", "precision"} <= set(test["pr"])
    assert rep["training"]["epochs_done"] == TINY_HPARAMS["epochs"] and rep["threshold_suggestions"]
    listing = client.get("/api/training/models").json()
    assert listing["total"] == 1 and listing["items"][0]["id"] == model_id


def test_threshold_objective_and_errors(client, trained):
    _, model_id = trained
    r = client.post(f"/api/training/models/{model_id}/threshold/suggest", json={"objective": "recall", "target": 0.9, "subset": "val"})
    assert r.status_code == 200, r.text
    sug = r.json()
    assert sug["val"]["recall"] >= 0.9 and sug["val"]["objective_met"] and "test" in sug
    r = client.post(f"/api/training/models/{model_id}/threshold/suggest", json={"objective": "false_reject", "target": 0.0, "subset": "test"})
    assert r.json()["test"]["fp"] == 0
    assert client.post(f"/api/training/models/{model_id}/threshold/suggest", json={"objective": "recall", "target": 2}).status_code == 422

    ev = client.get(f"/api/training/models/{model_id}/threshold/evaluate", params={"threshold": sug["threshold"]}).json()
    assert ev["val"]["recall"] == sug["val"]["recall"] and len(ev["test"]["recall_by_defect"]) == 6
    assert client.get(f"/api/training/models/{model_id}/threshold/evaluate", params={"threshold": 1.5}).status_code == 422

    thr = sug["threshold"]
    r = client.put(f"/api/training/models/{model_id}/threshold", json={"threshold": thr})
    assert r.status_code == 200 and r.json()["threshold"] == thr
    rep = client.get(f"/api/training/models/{model_id}/report").json()
    assert rep["threshold"] == thr and rep["subsets"]["val"]["at_threshold"]["recall"] >= 0.9

    # explorateur d'erreurs : seuil élevé → des faux négatifs, triés du plus grave au moins grave
    fn = client.get(f"/api/training/models/{model_id}/errors", params={"kind": "fn", "subset": "test", "threshold": 0.99, "limit": 5}).json()
    assert fn["total"] > 0 and len(fn["items"]) <= 5
    scores = [it["score"] for it in fn["items"]]
    assert scores == sorted(scores) and all(it["verdict"] == "non_viable" and it["score"] < 0.99 for it in fn["items"])
    assert len(fn["items"][0]["views"]) == 6 and {"index", "score"} <= set(fn["items"][0]["views"][0])
    page2 = client.get(f"/api/training/models/{model_id}/errors", params={"kind": "fn", "subset": "test", "threshold": 0.99, "limit": 5, "offset": 5}).json()
    assert not {i["track_id"] for i in page2["items"]} & {i["track_id"] for i in fn["items"]}
    fp = client.get(f"/api/training/models/{model_id}/errors", params={"kind": "fp", "threshold": 0.01}).json()
    assert fp["total"] > 0 and all(it["verdict"] == "viable" for it in fp["items"])
    scores = [it["score"] for it in fp["items"]]
    assert scores == sorted(scores, reverse=True)
    preds = client.get(f"/api/training/models/{model_id}/predictions", params={"subset": "test", "limit": 3}).json()
    assert preds["total"] > 3 and len(preds["items"]) == 3


def test_report_refresh_after_relabel(client, trained):
    project, model_id = trained
    before = client.get(f"/api/training/models/{model_id}/report").json()["subsets"]["test"]["n"]
    with project.session() as s:
        tid = s.query(m.Prediction.track_id).filter(m.Prediction.model_id == model_id, m.Prediction.subset == "test").first()[0]
        s.get(m.Label, tid).verdict = "exclu"
    after = client.post(f"/api/training/models/{model_id}/report/refresh").json()["subsets"]["test"]["n"]
    assert after == before - 1


def test_gradcam(client, trained):
    project, model_id = trained
    with project.session() as s:
        tid = s.query(m.Prediction.track_id).filter(m.Prediction.model_id == model_id).first()[0]
    r = client.get(f"/api/training/models/{model_id}/gradcam/{tid}")
    assert r.status_code == 200, r.text
    meta = r.json()
    assert meta["method"] == "grad_cam" and len(meta["views"]) == 12 and max(v["peak"] for v in meta["views"]) == pytest.approx(1.0)
    img = client.get(meta["views"][3]["url"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg" and len(img.content) > 1000
    assert (project.model_dir(model_id) / "cam" / str(tid) / "03.jpg").is_file()
    assert client.get(f"/api/training/models/{model_id}/gradcam/999999").status_code == 404


def test_train_cancel_resume_extend(client, project):
    split = client.post("/api/training/splits", json={"name": "S", "group_by": "lot", "seed": 4}).json()
    hp = {**TINY_HPARAMS, "input_size": 64, "n_views": 4, "epochs": 5}
    r = client.post("/api/training/train", json={"split_id": split["id"], "preset": None, "hparams": hp, "name": "À reprendre"})
    assert r.status_code == 200, r.text
    model_id, job_id = r.json()["model_id"], r.json()["job_id"]
    live = _wait_history(client, model_id, 1)
    assert live["job"]["status"] == "running" and live["job"]["result"]["history"]
    assert client.post(f"/api/training/models/{model_id}/cancel").status_code == 200
    job = state.jobs.wait(job_id, 120)
    assert job["status"] == "cancelled"
    detail = client.get(f"/api/training/models/{model_id}").json()
    assert detail["status"] == "cancelled" and detail["resumable"]
    done_before = len(detail["history"])
    assert 1 <= done_before < 5
    assert client.post(f"/api/training/models/{model_id}/cancel").status_code == 409
    # une split utilisée par un modèle est verrouillée
    assert client.get(f"/api/training/splits/{split['id']}").json()["locked"]

    r = client.post(f"/api/training/models/{model_id}/resume", json={"mode": "continue"})
    assert r.status_code == 200, r.text
    wait_job(r.json()["job_id"], 300)
    detail = client.get(f"/api/training/models/{model_id}").json()
    assert detail["status"] == "trained" and [h["epoch"] for h in detail["history"]] == [1, 2, 3, 4, 5]
    assert client.post(f"/api/training/models/{model_id}/resume", json={"mode": "continue"}).status_code == 400

    r = client.post(f"/api/training/models/{model_id}/resume", json={"mode": "extend", "epochs": 1})
    assert r.status_code == 200, r.text
    wait_job(r.json()["job_id"], 300)
    child = client.get(f"/api/training/models/{r.json()['model_id']}").json()
    assert child["parent_id"] == model_id and child["status"] == "trained" and len(child["history"]) == 1
    assert child["number"] == detail["number"] + 1

    cmp = client.get("/api/training/compare", params={"ids": [model_id, child["id"]], "objective": "recall", "target": 0.9}).json()
    assert cmp["same_test_set"] and len(cmp["models"]) == 2 and not cmp["warnings"]
    assert all("own_threshold" in x and "common_objective" in x for x in cmp["models"])

    assert client.patch(f"/api/training/models/{child['id']}", json={"name": "Renommé"}).json()["name"] == "Renommé"
    assert client.delete(f"/api/training/models/{child['id']}").status_code == 200
    assert client.get(f"/api/training/models/{child['id']}").status_code == 404
    assert not project.model_dir(child["id"]).exists()


def test_train_validation_errors(client, project):
    split = client.post("/api/training/splits", json={"name": "S", "group_by": "lot"}).json()
    assert client.post("/api/training/train", json={"split_id": split["id"], "hparams": {"arch": "resnet50"}}).status_code == 400
    assert client.post("/api/training/train", json={"split_id": split["id"], "hparams": {"inconnu": 1}}).status_code == 400
    assert client.post("/api/training/train", json={"split_id": 999}).status_code == 404


def test_anomaly_model(client, trained):
    project, clf_id = trained
    with project.session() as s:
        split_id = s.get(m.Model, clf_id).split_id
    params = {"arch": "mobilenetv3_small_100", "input_size": 96, "n_views": 4, "bank_size": 300, "patches_per_view": 16, "max_train_eggs": 40}
    r = client.post("/api/training/anomaly", json={"split_id": split_id, "params": params})
    assert r.status_code == 200, r.text
    wait_job(r.json()["job_id"], 300)
    model_id = r.json()["model_id"]
    detail = client.get(f"/api/training/models/{model_id}").json()
    assert detail["kind"] == "anomaly" and detail["status"] == "trained"
    rep = client.get(f"/api/training/models/{model_id}/report").json()
    auc = rep["subsets"]["test"]["auc_roc"]
    print(f"\nAUC test du modèle d'anomalies (petit) : {auc:.3f}")
    assert auc is not None and rep["anomaly"]["bank_size"] == 300
    with project.session() as s:
        preds = s.query(m.Prediction).filter(m.Prediction.model_id == model_id).all()
        tid = preds[0].track_id
    assert len(preds) == 180 and all(0 <= p.score <= 1 for p in preds)
    cam = client.get(f"/api/training/models/{model_id}/gradcam/{tid}").json()
    assert cam["method"] == "distance_patch" and len(cam["views"]) == 12
    r = client.post(f"/api/export/models/{model_id}")
    assert r.status_code == 400 and "anomalies" in r.json()["detail"]
    cmp = client.get("/api/training/compare", params={"ids": [clf_id, model_id]}).json()
    assert cmp["same_test_set"] and len(cmp["models"]) == 2
