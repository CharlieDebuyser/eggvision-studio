"""Détection classique : précision contre la vérité terrain du générateur synthétique."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from eggvision.vision import DetectionParams, detect_frame, detection_params, tracking_params
from eggvision.vision.params import ParamsError, params_schema
from eggvision.vision.scene import (
    SceneError,
    SceneGeometry,
    default_scene,
    lanes_from_positions,
    mm_per_px_from_ruler,
    regular_lanes,
    rescale_scene,
    validate_scene,
)
from eggvision_shared.synth import SceneConfig, SyntheticScene


def _match(dets, truths):
    """Associe chaque œuf réel à la détection la plus proche (≤ 30 % du petit axe)."""
    pairs, missed = [], []
    used = set()
    for tr in truths:
        best, bd = None, 1e9
        for k, d in enumerate(dets):
            dist = math.hypot(d["cx"] - tr.cx, d["cy"] - tr.cy)
            if dist < bd:
                best, bd = k, dist
        if best is None or bd > 0.3 * tr.minor_px:
            missed.append(tr)
        else:
            used.add(best)
            pairs.append((tr, dets[best], bd))
    ghosts = [d for k, d in enumerate(dets) if k not in used]
    return pairs, missed, ghosts


def _visible_fraction(tr, height):
    return (min(tr.cy + tr.minor_px / 2, height) - max(tr.cy - tr.minor_px / 2, 0)) / tr.minor_px


def _run(cfg: SceneConfig, frames=range(30, 90, 4), edit=None):
    scene = SyntheticScene(cfg)
    if edit:
        edit(scene)
    cal = scene.calibration()
    stats = {"n": 0, "missed": 0, "ghosts": 0, "lane_err": 0, "partial_err": 0, "pos": [], "major": [], "minor": [], "ms": []}
    for i in frames:
        img, truths = scene.frame(i)
        t0 = time.perf_counter()
        dets = detect_frame(img, cal)
        stats["ms"].append((time.perf_counter() - t0) * 1000)
        pairs, missed, ghosts = _match(dets, truths)
        stats["n"] += len(truths)
        # Un œuf à peine entré (moins de 40 % visible) peut légitimement être ignoré.
        stats["missed"] += sum(1 for tr in missed if _visible_fraction(tr, cfg.height) > 0.4)
        stats["ghosts"] += len(ghosts)
        for tr, d, dist in pairs:
            stats["lane_err"] += d["lane"] != tr.lane
            border = min(tr.cy - tr.minor_px / 2, cfg.height - 1 - (tr.cy + tr.minor_px / 2))
            if abs(border) > 0.05 * tr.minor_px:  # hors cas limites
                stats["partial_err"] += d["partial"] == tr.fully_visible
            if tr.fully_visible and not d["partial"]:
                stats["pos"].append(dist / tr.minor_px)
                stats["major"].append(d["major"] / tr.major_px - 1)
                stats["minor"].append(d["minor"] / tr.minor_px - 1)
    return stats


@pytest.mark.parametrize("scale", [0.5, 1.0])
def test_detection_accuracy_viable(scale):
    cfg = SceneConfig.scaled(scale, seed=3) if scale != 1.0 else SceneConfig(seed=3)
    st = _run(cfg)
    assert st["missed"] == 0 and st["ghosts"] == 0
    assert st["lane_err"] == 0 and st["partial_err"] == 0
    assert np.percentile(st["pos"], 95) < 0.04  # < 4 % du petit axe (≈ 1,7 mm)
    assert abs(np.mean(st["major"])) < 0.02 and abs(np.mean(st["minor"])) < 0.03
    assert np.std(st["major"]) < 0.02
    print(f"\n[mesure] détection {cfg.width}×{cfg.height} : {np.median(st['ms']):.1f} ms/image (médiane), "
          f"erreur de position p95 {np.percentile(st['pos'], 95) * 100:.1f} % du petit axe")
    # Objectif : nettement moins de 30 ms par image en pleine résolution (marge pour machines lentes / CI).
    assert np.median(st["ms"]) < 30


@pytest.mark.parametrize("lot_type", ["declasse:souillure", "declasse:deformation", "production"])
def test_detection_defective_eggs(lot_type):
    st = _run(SceneConfig.scaled(0.5, lot_type=lot_type, seed=5, defect_rate=0.5))
    assert st["missed"] <= 0.01 * st["n"]
    assert st["ghosts"] <= 0.01 * st["n"]
    assert st["lane_err"] == 0
    tol = 0.04 if lot_type == "declasse:souillure" else 0.07  # œufs déformés : contour irrégulier
    assert np.percentile(st["pos"], 95) < tol


def _darken(scene: SyntheticScene, color=(70.0, 100.0, 140.0)):
    """Œufs bruns foncés (plus sombres que le plus foncé du générateur)."""
    for c in range(0, 40):
        for egg in scene.column(c):
            if egg is not None:
                egg.color_bgr = color
                egg._texture = None


def test_detection_dark_brown_eggs():
    st = _run(SceneConfig.scaled(0.5, seed=6), edit=_darken)
    assert st["missed"] == 0 and st["ghosts"] == 0 and st["lane_err"] == 0
    assert np.percentile(st["pos"], 95) < 0.04


def test_detection_full_resolution_soiled_and_partial():
    cfg = SceneConfig(lot_type="declasse:souillure", seed=7)
    scene = SyntheticScene(cfg)
    cal = scene.calibration()
    seen_partial = 0
    for i in (40, 47, 55):
        img, truths = scene.frame(i)
        dets = detect_frame(img, cal)
        pairs, missed, ghosts = _match(dets, truths)
        assert not ghosts
        assert all(_visible_fraction(tr, cfg.height) < 0.4 for tr in missed)
        for tr, d, _ in pairs:
            assert d["lane"] == tr.lane
            if not tr.fully_visible:
                assert d["partial"]
                seen_partial += 1
            assert d["sharpness"] > 0 or d["partial"]
            assert set(d) >= {"cx", "cy", "major", "minor", "angle", "lane", "partial", "sharpness", "score"}
    assert seen_partial > 0


def test_detect_frame_api_contract():
    scene = SyntheticScene(SceneConfig.scaled(0.5, seed=1))
    img, truths = scene.frame(50)
    # Sans calibration : couloirs réguliers et échelle du capteur de référence (même champ de vue).
    dets = detect_frame(img, None)
    assert len(dets) == len([t for t in truths if _visible_fraction(t, img.shape[0]) > 0.4])
    assert {d["lane"] for d in dets} <= set(range(6))
    dets2, info = detect_frame(img, scene.calibration(), {"threshold_mode": "auto"}, return_info=True, want_mask=True)
    assert info["mask"].dtype == np.uint8 and info["threshold"] >= 55
    assert len(dets2) == len(dets)
    # Calibration à une autre résolution : adaptée automatiquement.
    big = SyntheticScene(SceneConfig(seed=1)).calibration()
    assert len(detect_frame(img, big)) == len(dets)
    # Image vide de rouleaux : aucune détection.
    empty = SyntheticScene(SceneConfig.scaled(0.5, seed=1, empty_lane_prob=1.0)).frame(10)[0]
    assert detect_frame(empty, None) == []
    with pytest.raises(ValueError):
        detect_frame(img[:, :, 0], None)


def test_detection_params_validation_and_schema():
    p = detection_params({"threshold": 90, "inconnu": 1})
    assert isinstance(p, DetectionParams) and p.threshold == 90
    with pytest.raises(ParamsError):
        detection_params({"threshold": 999})
    with pytest.raises(ParamsError):
        detection_params({"min_major_mm": 90, "max_major_mm": 80})
    assert tracking_params({}).n_views == 12
    schema = params_schema()
    names = {f["name"]: f for f in schema["detection"]}
    assert names["threshold"]["min"] == 0 and names["threshold"]["max"] == 255 and names["threshold"]["label"]
    assert names["method"]["type"] == "enum" and set(names["method"]["options"]) == {"classic", "nn"}
    assert {f["name"] for f in schema["tracking"]} >= {"n_views", "min_views", "gate_mm"}


def test_scene_geometry_and_calibration_helpers():
    scene = default_scene(728, 544)
    assert scene["mm_per_px"] == pytest.approx(0.54)
    assert validate_scene(scene)["lanes"][0]["index"] == 0
    with pytest.raises(SceneError):
        validate_scene({**scene, "direction": "diagonale"})
    with pytest.raises(SceneError):
        validate_scene({**scene, "lanes": [{"index": 0, "polygon": [[0, 0], [1, 1]]}]})
    with pytest.raises(SceneError):
        validate_scene({**scene, "mm_per_px": 0})
    geo = SceneGeometry.from_scene(scene)
    assert geo.lane_of(10, 100) == 0 and geo.lane_of(720, 100) == 5
    assert geo.lane_of(60, -40) == 0  # centre hors image (œuf tronqué) : ramené dans l'image
    assert geo.s_range == (0.0, 544.0)
    up = SceneGeometry.from_scene({**scene, "direction": "up"})
    assert up.along(0, 100) < up.along(0, 50)
    # Règle : 100 mm sur 200 px.
    assert mm_per_px_from_ruler((10, 10), (210, 10), 100) == pytest.approx(0.5)
    with pytest.raises(SceneError):
        mm_per_px_from_ruler((10, 10), (10, 11), 100)
    # Couloirs horizontaux pour un défilement latéral.
    lanes = regular_lanes(400, 600, 3, "left")
    assert lanes[0]["polygon"][0][1] == 0.0 and len(lanes) == 3
    # Proposition à partir de positions détectées (couloirs décalés et inégaux).
    centers = [70, 190, 300, 430, 550, 660]
    pts = [(c + dx, y) for c in centers for dx in (-3, 0, 4) for y in (100, 300)]
    lanes = lanes_from_positions(pts, 728, 544, "down", 6)
    bounds = [lane["polygon"][1][0] for lane in lanes][:-1]
    assert bounds == pytest.approx([(a + b) / 2 for a, b in zip(centers, centers[1:])], abs=3)
    # Pas assez de points : découpage régulier.
    assert lanes_from_positions([(10, 10)], 728, 544, "down", 6) == regular_lanes(728, 544, 6, "down")
    # Changement de résolution.
    big = rescale_scene(scene, 1456, 1088)
    assert big["mm_per_px"] == pytest.approx(0.27) and big["lanes"][1]["polygon"][0][0] == pytest.approx(2 * scene["lanes"][1]["polygon"][0][0])
