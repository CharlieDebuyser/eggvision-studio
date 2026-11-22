"""Parcours complet par l'API, sans matériel : images brutes → détection et suivi → labels →
découpage → entraînement → rapport → export ONNX → simulation avec trappes → CSV et vidéo."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from eggvision import state
from eggvision.demo import add_synthetic_lot
from eggvision.testing import wait_job
from eggvision_shared.synth import SceneConfig, SyntheticScene


def _job_id(body: dict) -> str:
    """Identifiant de tâche, quelle que soit la forme de la réponse du module."""
    if "job_id" in body:
        return body["job_id"]
    job = body.get("job") or body
    return job["id"] if "kind" in job else body["job"]["id"]


@pytest.mark.slow
def test_full_pipeline(client, tmp_path):
    assert client.post("/api/project/create", json={"parent_dir": str(tmp_path), "name": "Parcours"}).status_code == 200
    project = state.projects.require()
    project.update_config({"scene": SyntheticScene(SceneConfig.scaled(0.5)).calibration(), "user": {"name": "test"}})

    # 1. Trois sessions de lots bruts (images seules), comme après une prise de vue.
    day0 = datetime.now(UTC).replace(hour=8, minute=0, second=0, microsecond=0)
    seed = 100
    for s in range(3):
        day = day0 - timedelta(days=3 - s)
        add_synthetic_lot(project, f"Viables J{s}", "viable", 4, farm="Ferme", started_at=day, seed=(seed := seed + 1), with_tracks=False)
        for k, defect in enumerate(("souillure", "sang", "deformation")):
            add_synthetic_lot(project, f"{defect} J{s}", f"declasse:{defect}", 2, farm="Ferme",
                              started_at=day + timedelta(minutes=10 * (k + 1)), seed=(seed := seed + 1), with_tracks=False)
    lots = client.get("/api/detection/lots", params={"limit": 50}).json()
    lot_ids = [lot["id"] for lot in (lots["items"] if isinstance(lots, dict) else lots)]
    assert len(lot_ids) == 12

    # 2. Détection et suivi (critère 2 : ≥ 95 % d'œufs en pistes exploitables de ≥ 8 vues).
    r = client.post("/api/detection/run", json={"lot_ids": lot_ids})
    assert r.status_code == 200, r.text
    job = wait_job(_job_id(r.json()), 600)
    per_lot = job["result"]["lots"]
    n_expected = 3 * (4 * 6 + 3 * 2 * 6)
    usable = sum(v["usable"] for v in per_lot.values())
    assert usable >= 0.95 * n_expected, (usable, n_expected)
    assert sum(v["tracks"] for v in per_lot.values()) <= n_expected + 2

    # 3. Labels : pré-labels issus du type de lot, relecture d'une piste au passage.
    dash = client.get("/api/labeling/dashboard").json()
    assert dash
    tracks = client.get("/api/labeling/tracks", params={"limit": 5}).json()
    first = (tracks.get("items") or tracks.get("tracks"))[0]
    assert client.post(f"/api/labeling/tracks/{first['id']}/confirm").status_code == 200

    # 4. Découpage par session puis entraînement court.
    split = client.post("/api/training/splits", json={"name": "Par session", "group_by": "session", "seed": 1})
    assert split.status_code == 200, split.text
    split = split.json()
    assert split["summary"]["leakage_check"]["ok"]
    assert set(split["assignments"].values()) == {"train", "val", "test"}
    hp = {"arch": "mobilenetv3_small_100", "input_size": 96, "n_views": 6, "epochs": 5, "batch_size": 8, "patience": 5, "freeze_epochs": 0}
    r = client.post("/api/training/train", json={"split_id": split["id"], "preset": "rapide", "hparams": hp, "name": "Parcours"})
    assert r.status_code == 200, r.text
    model_id = r.json()["model_id"]
    wait_job(r.json()["job_id"], 900)
    model = client.get(f"/api/training/models/{model_id}").json()
    assert model["status"] == "trained"
    test = model["metrics"]["test"]
    assert test["n"] >= 40 and test["auc_roc"] > 0.8, test
    report = client.get(f"/api/training/models/{model_id}/report").json()
    assert "roc" in report["subsets"]["test"]

    # 5. Export ONNX + parité + banc local.
    r = client.post(f"/api/export/models/{model_id}", json={"variants": ["fp32", "int8"]})
    assert r.status_code == 200, r.text
    export_job = wait_job(r.json()["job_id"], 600)
    assert export_job["result"]["parity_max_abs_diff"] <= 1e-4
    exported = client.get(f"/api/export/models/{model_id}").json()
    assert exported["exported"] and exported["export"]["parity"]["fp32"]["ok"]
    assert client.get(f"/api/export/models/{model_id}/download").status_code == 200

    # 6. Simulation avec trappes sur le jeu de test, journal CSV et vidéo.
    r = client.post("/api/simulation", json={"model_id": model_id, "n_columns": 40, "defect_rate": 0.3, "seed": 7, "name": "Parcours"})
    assert r.status_code == 200, r.text
    body = r.json()
    sim_id = body.get("id") or body["simulation"]["id"]
    wait_job(_job_id(body), 600)
    sim = client.get(f"/api/simulation/{sim_id}").json()
    assert sim["status"] == "done"
    res = sim["results"]
    assert res["counts"]["eggs"] == 240 and res["counts"]["non_viable"] == 72, res["counts"]
    assert res["rates"]["recall"] > 0.6 and res["counts"]["late_decisions"] == 0, res["rates"]
    csv = client.get(f"/api/simulation/{sim_id}/journal.csv")
    assert csv.status_code == 200 and len(csv.text.splitlines()) == 241
    r = client.post(f"/api/simulation/{sim_id}/video", json={"max_duration_s": 4, "width": 640, "height": 360})
    assert r.status_code == 200, r.text
    vbody = r.json()
    wait_job(_job_id(vbody), 300)
    video = client.get(f"/api/simulation/{sim_id}/video")
    assert video.status_code == 200 and video.content[4:8] == b"ftyp"
