"""Suivi et traitement par lot : critère d'acceptation 2 mesuré contre la vérité terrain.

Critère 2 : sur un lot de test, au moins 95 % des œufs visibles sont transformés en pistes
exploitables (qualité ``ok``) avec au moins 8 vues chacune — ici : chaque œuf qui traverse
entièrement l'image doit donner exactement une piste ``ok`` d'au moins 8 vues, dans le bon
couloir, sans piste fantôme ni fusion de deux œufs.
"""

from __future__ import annotations

import time

import cv2
import pytest
from eggvision import state
from eggvision.db import models as m
from eggvision.testing import wait_job
from eggvision.vision import tracking_params
from eggvision.vision.evaluation import evaluate_synthetic_lot, write_synthetic_lot
from eggvision.vision.pipeline import DetectionRefused, frame_times, process_lot
from eggvision.vision.scene import SceneGeometry
from eggvision.vision.tracking import Tracker
from eggvision_shared.formats import read_image
from eggvision_shared.synth import SceneConfig
from sqlalchemy import func, select

MIN_RATE = 0.95


def _check_acceptance(ev: dict, label: str) -> None:
    print(f"\n[mesure] critère 2 — {label} : {ev['eggs_ok']}/{ev['eggs_traversing']} œufs = {ev['rate'] * 100:.1f} % "
          f"(fantômes {ev['ghost_tracks']}, mélangées {ev['impure_tracks']}, doublons {ev['duplicates']})")
    assert ev["eggs_traversing"] > 0
    assert ev["rate"] >= MIN_RATE, ev["failures"]
    assert ev["ghost_tracks"] == 0 and ev["impure_tracks"] == 0 and ev["duplicates"] == 0


def test_acceptance_frames_project_via_api(client, frames_project):
    project = frames_project
    r = client.post("/api/detection/run", json={"lot_ids": ["lot-viable", "lot-souillure"]})
    assert r.status_code == 200, r.text
    job = wait_job(r.json()["id"], timeout=300)
    assert job["kind"] == "detection" and set(job["result"]["lots"]) == {"lot-viable", "lot-souillure"}
    for lot_id, lot_type, seed in (("lot-viable", "viable", 1), ("lot-souillure", "declasse:souillure", 2)):
        ev = evaluate_synthetic_lot(project, lot_id, SceneConfig.scaled(0.5, lot_type=lot_type, seed=seed))
        _check_acceptance(ev, f"{lot_id} 728×544")
        stats = job["result"]["lots"][lot_id]
        print(f"[mesure] {lot_id} : {stats['ms_per_frame']} ms/image au total (détection {stats['detect_ms_per_frame']} ms/image/fil), "
              f"vitesse estimée {stats['speed_mm_s']} mm/s (réelle 152)")
        assert stats["speed_mm_s"] == pytest.approx(152.0, rel=0.02)

    with project.session() as s:
        lot = s.get(m.Lot, "lot-viable")
        assert lot.detection_status == "done" and lot.track_count > 0
        assert lot.detection_params["method"] == "classic" and "stats" in lot.detection_params
        track = s.scalars(select(m.Track).where(m.Track.lot_id == "lot-viable", m.Track.quality == "ok").order_by(m.Track.id)).first()
        assert track.n_views == 12 and len(track.views) == 12 and track.flags == []
        assert track.end_ts > track.start_ts > 0 and track.speed_px_s > 0
        # Vues : fichiers 224×224 recadrés, couvrant toute la traversée, dans l'ordre chronologique.
        frames = [v.frame_index for v in track.views]
        assert frames == sorted(frames) and frames[-1] - frames[0] > 0.6 * (track.end_frame - track.start_frame)
        view = read_image(project.abspath(track.views[0].path))
        assert view.shape == (224, 224, 3)
        assert track.views[0].path == f"crops/{track.id}/00.jpg"
        n_det = s.scalar(select(func.count()).select_from(m.Detection).where(m.Detection.track_id == track.id))
        assert n_det == track.n_detections
        # Pré-labels : le type du lot donne le label initial.
        assert track.label is not None and track.label.verdict == "viable" and track.label.source == "auto"

    # Statistiques de qualité et listes.
    st = client.get("/api/detection/lots/lot-viable/stats").json()
    assert st["usable"] >= 30 and st["usable_rate_complete"] == 1.0 and st["min_views"] == 8
    lots = client.get("/api/detection/lots", params={"limit": 10}).json()
    row = next(i for i in lots["items"] if i["id"] == "lot-viable")
    assert row["detection_status"] == "done" and row["usable"] == st["usable"] and row["track_count"] == st["tracks"]
    doubtful = client.get("/api/detection/lots/lot-viable/tracks", params={"quality": "douteuse"}).json()
    assert doubtful["total"] == st["doubtful"]
    assert all(t["quality"] == "douteuse" and t["flags"] for t in doubtful["items"])
    inc = client.get("/api/detection/lots/lot-viable/tracks", params={"flag": "incomplete", "limit": 5}).json()
    assert inc["total"] == st["flags"].get("incomplete", 0) and len(inc["items"]) <= 5
    detail = client.get(f"/api/detection/tracks/{track.id}").json()
    assert len(detail["views"]) == 12 and len(detail["detections"]) == track.n_detections
    assert client.get(detail["views"][0]["url"]).status_code == 200
    fd = client.get(f"/api/detection/lots/lot-viable/frames/{detail['views'][3]['frame_index']}/detections").json()
    assert any(d["track_id"] == track.id for d in fd["detections"])


@pytest.mark.parametrize(
    "label, cfg_kwargs, drop, direction",
    [
        ("images manquantes", {"seed": 21}, set(range(3, 400, 7)) | {40, 41, 42, 43, 90, 91, 92}, "down"),
        ("couloirs vides", {"seed": 22, "empty_lane_prob": 0.35, "lot_type": "production", "defect_rate": 0.3}, set(), "down"),
        ("défilement vers la gauche", {"seed": 23, "lot_type": "declasse:deformation"}, set(), "left"),
        ("défilement vers le haut", {"seed": 24}, set(), "up"),
        ("flou de bougé", {"seed": 25, "motion_blur_px": 5, "noise_sigma": 4.0}, set(), "down"),
    ],
)
def test_acceptance_scenarios(frames_project, label, cfg_kwargs, drop, direction):
    cfg = SceneConfig.scaled(0.5, **cfg_kwargs)
    _, cal = write_synthetic_lot(frames_project, "scenario", cfg, 6, drop_frames=drop, direction=direction)
    stats = process_lot(frames_project, "scenario", scene=cal)
    ev = evaluate_synthetic_lot(frames_project, "scenario", cfg, direction=direction)
    _check_acceptance(ev, f"{label} 728×544")
    assert stats["speed_mm_s"] == pytest.approx(152.0, rel=0.02)
    if drop:
        assert stats["sensor_clock"] and stats["frames"] == frames_project.stats()["frames"] - _frames(frames_project, exclude="scenario")


def _frames(project, exclude: str) -> int:
    with project.session() as s:
        return s.scalar(select(func.count()).select_from(m.Frame).where(m.Frame.lot_id != exclude)) or 0


def test_acceptance_full_resolution(frames_project):
    cfg = SceneConfig(seed=31, lot_type="production", defect_rate=0.3)
    _, cal = write_synthetic_lot(frames_project, "pleine-resolution", cfg, 4)
    frames_project.update_config({"scene": cal})  # calibration à la résolution du capteur
    t0 = time.perf_counter()
    stats = process_lot(frames_project, "pleine-resolution")
    elapsed = time.perf_counter() - t0
    ev = evaluate_synthetic_lot(frames_project, "pleine-resolution", cfg)
    _check_acceptance(ev, "1456×1088")
    print(f"[mesure] 1456×1088 : {stats['frames']} images en {elapsed:.1f} s = {stats['ms_per_frame']} ms/image au total "
          f"(lecture JPEG {stats['read_ms_per_frame']} ms, détection {stats['detect_ms_per_frame']} ms par image et par fil)")
    # Les anciens lots 728×544 restent traitables avec la calibration pleine résolution (adaptée automatiquement).
    stats_small = process_lot(frames_project, "lot-souillure")
    ev_small = evaluate_synthetic_lot(frames_project, "lot-souillure", SceneConfig.scaled(0.5, lot_type="declasse:souillure", seed=2))
    assert ev_small["rate"] >= MIN_RATE and stats_small["tracks"] > 0


def test_rerun_is_idempotent_and_protects_reviewed_labels(client, frames_project):
    project = frames_project
    first = process_lot(project, "lot-souillure")
    with project.session() as s:
        ids1 = sorted(s.scalars(select(m.Track.id).where(m.Track.lot_id == "lot-souillure")))
        n_det1 = s.scalar(select(func.count()).select_from(m.Detection).where(m.Detection.lot_id == "lot-souillure"))
    second = process_lot(project, "lot-souillure")
    assert (second["tracks"], second["ok"], second["usable"]) == (first["tracks"], first["ok"], first["usable"])
    assert second["replaced_tracks"] == len(ids1)
    with project.session() as s:
        ids2 = sorted(s.scalars(select(m.Track.id).where(m.Track.lot_id == "lot-souillure")))
        n_det2 = s.scalar(select(func.count()).select_from(m.Detection).where(m.Detection.lot_id == "lot-souillure"))
        assert len(ids2) == len(ids1) and not set(ids1) & set(ids2)
        assert n_det2 == n_det1
        assert s.scalar(select(func.count()).select_from(m.Label).where(m.Label.track_id.in_(ids1))) == 0
    assert not any(project.crop_dir(i).exists() for i in ids1 if i not in ids2)
    assert all(project.crop_dir(i).is_dir() for i in ids2[:5])

    # Un label relu par un humain : relance refusée sans « forcer ».
    with project.session() as s:
        label = s.get(m.Label, ids2[0])
        label.reviewed, label.source, label.verdict = True, "humain", "non_viable"
    with pytest.raises(DetectionRefused, match="relu"):
        process_lot(project, "lot-souillure")
    check = client.post("/api/detection/run/check", json={"lot_ids": ["lot-souillure", "lot-viable"]}).json()
    assert check["needs_force"] and check["lots"] == [{"lot_id": "lot-souillure", "reviewed": 1}]
    r = client.post("/api/detection/run", json={"lot_ids": ["lot-souillure"]})
    assert r.status_code == 409 and "forcer" in r.json()["detail"]
    with project.session() as s:  # rien n'a bougé
        assert sorted(s.scalars(select(m.Track.id).where(m.Track.lot_id == "lot-souillure"))) == ids2
    r = client.post("/api/detection/run", json={"lot_ids": ["lot-souillure"], "force": True})
    assert r.status_code == 200
    wait_job(r.json()["id"])
    with project.session() as s:
        assert s.get(m.Label, ids2[0]) is None
        assert s.scalar(select(func.count()).select_from(m.Label).join(m.Track).where(
            m.Track.lot_id == "lot-souillure", m.Label.reviewed.is_(True))) == 0


def test_cancel_keeps_previous_results(client, frames_project):
    project = frames_project
    process_lot(project, "lot-viable")
    with project.session() as s:
        before = sorted(s.scalars(select(m.Track.id).where(m.Track.lot_id == "lot-viable")))
    r = client.post("/api/detection/run", json={"lot_ids": ["lot-viable"], "overrides": {"tracking": {"view_size": 512}}})
    job_id = r.json()["id"]
    deadline = time.time() + 30
    while time.time() < deadline and state.jobs.get(job_id).progress < 0.1:
        time.sleep(0.01)
    assert client.post("/api/detection/cancel", json={"lot_id": "lot-viable"}).status_code == 200
    job = wait_job(job_id, expect="cancelled")
    assert job["status"] == "cancelled"
    with project.session() as s:
        lot = s.get(m.Lot, "lot-viable")
        assert lot.detection_status == "done"
        assert sorted(s.scalars(select(m.Track.id).where(m.Track.lot_id == "lot-viable"))) == before
        assert s.scalar(select(func.count()).select_from(m.Detection).where(m.Detection.lot_id == "lot-viable", m.Detection.track_id.is_(None))) == 0
        assert not s.scalars(select(m.View).where(m.View.track_id.not_in(before))).first()
    # Les vues des anciennes pistes sont intactes (224 px, pas 512).
    assert read_image(project.crop_path(before[0], 0)).shape[0] == 224
    assert client.post("/api/detection/cancel", json={"lot_id": "lot-viable"}).status_code == 404


def test_tracker_tolerates_missed_detections_and_keeps_lanes_apart():
    """Deux couloirs voisins, un œuf non détecté pendant 3 images, images à intervalle irrégulier."""
    scene = {"image_width": 600, "image_height": 400, "direction": "down", "mm_per_px": 0.5,
             "lanes": [{"index": 0, "polygon": [[0, 0], [300, 0], [300, 400], [0, 400]]},
                       {"index": 1, "polygon": [[300, 0], [600, 0], [600, 400], [300, 400]]}]}
    geo = SceneGeometry.from_scene(scene)
    tracker = Tracker(geo, tracking_params({}))
    speed = 300.0  # px/s
    t = 0.0
    closed = []
    for k in range(60):
        t += 1 / 30 if k % 9 else 3 / 30  # images perdues : saut de temps
        dets = []
        for lane, x, y0 in ((0, 150, -40), (1, 450, -60), (0, 150, -230)):
            y = y0 + speed * t
            if -50 < y < 450 and not (lane == 1 and 20 <= k < 23):
                partial = y < 45 or y > 355
                dets.append({"cx": x, "cy": y, "major": 110, "minor": 84, "angle": 0, "lane": lane, "partial": partial, "sharpness": 50.0})
        closed += tracker.update(k, t, dets)
    closed += tracker.flush()
    long = [tr for tr in closed if len(tr.obs) > 10]
    assert len(long) == 3
    assert {tr.lane()[0] for tr in long} == {0, 1}
    assert all(tr.lane()[1] == 1.0 for tr in long)
    assert tracker.speed == pytest.approx(speed, rel=0.01)


def test_frame_times_fallback():
    assert frame_times([(0, 10_000_000_000), (1, 10_033_000_000), (3, 10_100_000_000)], 30) == ([0.0, 0.033, 0.1], True)
    times, sensor = frame_times([(0, 0), (1, 0), (2, 0)], 20)
    assert not sensor and times == [0.0, 0.05, 0.1]


def test_views_are_shared_crops(frames_project):
    """Les vues enregistrées sont exactement le recadrage partagé PC/Pi (``preprocess.extract_crop``)."""
    from eggvision_shared.preprocess import extract_crop

    process_lot(frames_project, "lot-viable")
    with frames_project.session() as s:
        view = s.scalars(select(m.View).order_by(m.View.id)).first()
        det = s.scalars(select(m.Detection).where(m.Detection.track_id == view.track_id, m.Detection.frame_index == view.frame_index)).one()
    frame = read_image(frames_project.frame_path("lot-viable", view.frame_index))
    expected = extract_crop(frame, det.cx, det.cy, det.major, det.minor, out_size=224, margin=1.15)
    stored = read_image(frames_project.abspath(view.path))
    assert cv2.absdiff(expected, stored).mean() < 3.0  # écart dû au seul codage JPEG
