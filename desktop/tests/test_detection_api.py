"""API ``/api/detection`` : calibration, paramètres, aperçu, détecteur entraînable."""

from __future__ import annotations

import base64

import pytest
from eggvision.testing import wait_job
from eggvision.vision.evaluation import evaluate_synthetic_lot
from eggvision_shared.synth import SceneConfig, SyntheticScene


def test_api_requires_project(client):
    assert client.get("/api/detection/scene").status_code == 409
    assert client.get("/api/detection/params/schema").status_code == 200


def test_scene_calibration_endpoints(client, frames_project):
    r = client.get("/api/detection/scene", params={"lot_id": "lot-viable"}).json()
    assert r["configured"] and r["scene"]["direction"] == "down" and r["lot_scene"]["image_width"] == 728
    truth = SyntheticScene(SceneConfig.scaled(0.5)).calibration()

    # Proposition de couloirs déduite des détections : proche des vrais couloirs.
    prop = client.post("/api/detection/scene/propose-lanes", json={"lot_id": "lot-viable", "n_frames": 8}).json()
    assert prop["mode"] == "detections" and prop["n_points"] > 30 and len(prop["lanes"]) == 6
    for got, exp in zip(prop["lanes"], truth["lanes"]):
        assert got["polygon"][1][0] == pytest.approx(exp["polygon"][1][0], abs=8)
    reg = client.post("/api/detection/scene/propose-lanes", json={"lot_id": "lot-viable", "mode": "regular", "n_lanes": 4, "direction": "left"}).json()
    assert len(reg["lanes"]) == 4 and reg["lanes"][0]["polygon"][2][0] == 728

    # Règle : 2 points + longueur réelle.
    assert client.post("/api/detection/scene/ruler", json={"p1": [0, 0], "p2": [300, 400], "length_mm": 250}).json()["mm_per_px"] == 0.5
    assert client.post("/api/detection/scene/ruler", json={"p1": [0, 0], "p2": [0, 0], "length_mm": 250}).status_code == 400

    # Enregistrement validé, image de référence conservée.
    scene = {**truth, "lanes": prop["lanes"], "reference": {"lot_id": "lot-viable", "frame_index": 40}}
    saved = client.put("/api/detection/scene", json=scene).json()["scene"]
    assert saved["reference"]["frame_index"] == 40 and frames_project.get_config("scene")["lanes"] == saved["lanes"]
    bad = client.put("/api/detection/scene", json={**scene, "mm_per_px": -1})
    assert bad.status_code == 422 or bad.status_code == 400
    bad = client.put("/api/detection/scene", json={**scene, "lanes": [{"index": 0, "polygon": [[0, 0], [5, 5]]}]})
    assert bad.status_code == 400 and "Calibration invalide" in bad.json()["detail"]

    frames = client.get("/api/detection/lots/lot-viable/frames", params={"limit": 5, "offset": 10}).json()
    assert frames["total"] > 100 and [f["index"] for f in frames["items"]] == [10, 11, 12, 13, 14]
    assert frames["items"][0]["t"] == pytest.approx(10 / 30, abs=1e-3)
    assert client.get("/api/detection/defaults/scene", params={"width": 1456, "height": 1088}).json()["mm_per_px"] == 0.27


def test_params_endpoints(client, frames_project):
    r = client.get("/api/detection/params").json()
    assert r["detection"]["method"] == "classic" and r["tracking"]["n_views"] == 12
    assert {f["name"] for f in r["schema"]["detection"]} >= {"threshold", "min_major_mm"}
    r = client.put("/api/detection/params", json={"detection": {"threshold": 95}, "tracking": {"n_views": 10}}).json()
    assert r["detection"]["threshold"] == 95 and r["tracking"]["n_views"] == 10
    assert frames_project.get_config("detection")["threshold"] == 95
    bad = client.put("/api/detection/params", json={"detection": {"threshold": -5}})
    assert bad.status_code == 400 and "threshold" in bad.json()["detail"]
    r = client.post("/api/detection/params/reset", json={"section": "detection"}).json()
    assert r["detection"]["threshold"] == 80 and r["tracking"]["n_views"] == 10
    r = client.post("/api/detection/params/reset", json={}).json()
    assert r["tracking"]["n_views"] == 12


def test_preview(client, frames_project):
    r = client.post("/api/detection/preview", json={"lot_id": "lot-viable", "frame_index": 60, "mask": True})
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["count"] >= 12 and p["complete"] >= 12 and p["elapsed_ms"] < 200 and p["width"] == 728
    assert p["mask_png"].startswith("data:image/png;base64,")
    assert len(base64.b64decode(p["mask_png"].split(",", 1)[1])) > 100
    assert {d["lane"] for d in p["detections"]} == set(range(6))
    # Paramètres candidats (non enregistrés) : un seuil maximal ne détecte rien.
    p2 = client.post("/api/detection/preview", json={"lot_id": "lot-viable", "frame_index": 60, "params": {"threshold": 255}}).json()
    assert p2["count"] == 0
    assert frames_project.get_config("detection") in (None, {}) or frames_project.get_config("detection").get("threshold") != 255
    # Calibration candidate : un seul couloir.
    scene = {"image_width": 728, "image_height": 544, "direction": "down", "mm_per_px": 0.54,
             "lanes": [{"index": 0, "polygon": [[0, 0], [120, 0], [120, 544], [0, 544]]}]}
    p3 = client.post("/api/detection/preview", json={"lot_id": "lot-viable", "frame_index": 60, "scene": scene}).json()
    assert {d["lane"] for d in p3["detections"]} == {-1, 0}
    assert client.post("/api/detection/preview", json={"lot_id": "inconnu", "frame_index": 0}).status_code == 404
    assert client.post("/api/detection/preview", json={"lot_id": "lot-viable", "frame_index": 99999}).status_code == 404
    assert client.post("/api/detection/preview", json={"lot_id": "lot-viable", "params": {"work_mm_per_px": 99}}).status_code == 400


def test_lots_listing_pagination(client, frames_project):
    r = client.get("/api/detection/lots", params={"limit": 1}).json()
    assert r["total"] == 2 and len(r["items"]) == 1 and r["items"][0]["detection_status"] == "none"
    r = client.get("/api/detection/lots", params={"q": "Souill"}).json()
    assert [i["id"] for i in r["items"]] == ["lot-souillure"]
    assert client.get("/api/detection/lots", params={"status": "done"}).json()["total"] == 0
    assert client.post("/api/detection/run", json={"lot_ids": ["absent"]}).status_code == 400
    assert client.post("/api/detection/run", json={"lot_ids": []}).status_code == 422


def test_trainable_detector_end_to_end(client, frames_project):
    # 1. Images d'annotation avec boîtes automatiques (méthode classique).
    r = client.post("/api/detection/detector/annotations", json={"lot_id": "lot-viable", "n": 8})
    assert r.status_code == 200 and r.json()["added"] == 8
    assert client.post("/api/detection/detector/annotations", json={"lot_id": "lot-viable", "n": 8}).json()["added"] == 0  # déjà présentes
    listing = client.get("/api/detection/detector/annotations", params={"limit": 3}).json()
    assert listing["total"] == 8 and len(listing["items"]) == 3 and listing["n_boxes"] > 50
    ann = client.get(f"/api/detection/detector/annotations/{listing['items'][1]['id']}").json()
    assert ann["source"] == "auto" and ann["boxes"]

    # 2. Correction manuelle : suppression d'une boîte puis ajout d'une boîte corrigée.
    boxes = ann["boxes"][1:] + [ann["boxes"][0]]
    fixed = client.put(f"/api/detection/detector/annotations/{ann['id']}", json={"boxes": boxes}).json()
    assert fixed["source"] == "corrige" and len(fixed["boxes"]) == len(ann["boxes"])
    assert client.put(f"/api/detection/detector/annotations/{ann['id']}", json={"boxes": [{"x": 1, "y": 1}]}).status_code == 400
    assert client.put("/api/detection/detector/annotations/x:1", json={"boxes": []}).status_code == 404
    first = listing["items"][0]["id"]
    assert client.delete(f"/api/detection/detector/annotations/{first}").status_code == 200
    assert client.get("/api/detection/detector").json()["annotations"] == {"total": 7, "n_boxes": pytest.approx(7 * 20, abs=40), "n_corrected": 1}

    # 3. Entraînement minuscule (tâche detector_training, groupe gpu) et activation.
    r = client.post("/api/detection/detector/train", json={"epochs": 40, "input_width": 256, "name": "essai", "activate": True})
    assert r.status_code == 200, r.text
    job = wait_job(r.json()["id"], timeout=600)
    assert job["kind"] == "detector_training" and job["result"]["history"]
    meta = job["result"]["detector"]
    assert meta["metrics"]["recall"] >= 0.6, meta["metrics"]  # entraînement minuscule : seuil volontairement bas
    info = client.get("/api/detection/detector").json()
    assert info["active"] == meta["id"] and info["method"] == "classic" and info["detectors"][0]["id"] == meta["id"]
    assert (frames_project.models_dir / "detector" / meta["id"] / "weights.pt").is_file()

    # 4. Aperçu avec la méthode « nn » puis traitement d'un lot avec le détecteur entraîné.
    p = client.post("/api/detection/preview", json={"lot_id": "lot-souillure", "frame_index": 60, "params": {"method": "nn"}}).json()
    assert p["complete"] >= 10, p
    assert client.post("/api/detection/detector/activate", json={"detector_id": meta["id"], "use": True}).json()["method"] == "nn"
    assert client.delete(f"/api/detection/detector/{meta['id']}").status_code == 409  # utilisé
    r = client.post("/api/detection/run", json={"lot_ids": ["lot-souillure"]})
    job = wait_job(r.json()["id"], timeout=600)
    stats = job["result"]["lots"]["lot-souillure"]
    assert stats["status"] == "done" and stats["tracks"] > 0
    ev = evaluate_synthetic_lot(frames_project, "lot-souillure", SceneConfig.scaled(0.5, lot_type="declasse:souillure", seed=2))
    print(f"\n[mesure] détecteur entraînable (40 époques, 7 images) : critère 2 = {ev['rate'] * 100:.0f} % "
          f"({ev['eggs_ok']}/{ev['eggs_traversing']}), fantômes {ev['ghost_tracks']}")
    assert ev["rate"] >= 0.5 and ev["ghost_tracks"] == 0

    # 5. Retour à la méthode classique, suppression du détecteur.
    assert client.post("/api/detection/detector/activate", json={"use": False}).json()["method"] == "classic"
    assert client.delete(f"/api/detection/detector/{meta['id']}").status_code == 200
    assert client.get("/api/detection/detector").json()["active"] is None
    assert client.post("/api/detection/detector/activate", json={"detector_id": "absent"}).status_code == 404


def test_training_requires_annotations(client, frames_project):
    r = client.post("/api/detection/detector/train", json={"epochs": 1})
    assert r.status_code == 400 and "annotation" in r.json()["detail"]
