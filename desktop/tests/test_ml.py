"""Tests du module d'apprentissage sans entraînement : métriques, découpages, jeu de données, réseau."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from eggvision.db import models as m
from eggvision.ml import metrics as mt
from eggvision.ml import splits as sp


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------
def test_roc_pr_against_hand_computed_values():
    y = [0, 0, 1, 1]
    s = [0.1, 0.4, 0.35, 0.8]
    roc = mt.roc_curve(y, s)
    # seuils : inf, 0.8, 0.4, 0.35, 0.1
    assert roc["fpr"].tolist() == [0.0, 0.0, 0.5, 0.5, 1.0]
    assert roc["tpr"].tolist() == [0.0, 0.5, 0.5, 1.0, 1.0]
    assert math.isinf(roc["thresholds"][0]) and roc["thresholds"][1:].tolist() == [0.8, 0.4, 0.35, 0.1]
    assert mt.roc_auc(y, s) == pytest.approx(0.75)
    pr = mt.precision_recall_curve(y, s)
    assert pr["recall"].tolist() == [0.0, 0.5, 0.5, 1.0, 1.0]
    assert pr["precision"].tolist() == pytest.approx([1.0, 1.0, 0.5, 2 / 3, 0.5])
    # AP = 0.5 × 1 + 0 × 0.5 + 0.5 × 2/3 + 0 × 0.5
    assert mt.average_precision(y, s) == pytest.approx(0.5 + 0.5 * 2 / 3)


def test_auc_with_ties_and_degenerate_cases():
    # Ex æquo : un positif et un négatif au même score comptent pour 1/2.
    assert mt.roc_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    assert mt.roc_auc([0, 0, 1, 1], [0.2, 0.6, 0.6, 0.9]) == pytest.approx((1 + 0.5 + 1 + 1) / 4)
    assert mt.roc_auc([1, 1], [0.2, 0.3]) is None
    assert mt.average_precision([0, 0], [0.2, 0.3]) is None
    # Comparaison à une implémentation par paires (Mann-Whitney)
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = np.round(rng.random(200), 2)  # nombreux ex æquo
    pos, neg = s[y == 1], s[y == 0]
    pairs = ((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])).mean()
    assert mt.roc_auc(y, s) == pytest.approx(pairs)


def test_confusion_uses_greater_or_equal():
    c = mt.confusion([1, 0, 1, 0], [0.5, 0.5, 0.2, 0.1], 0.5)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (1, 1, 1, 1)
    assert c["recall"] == 0.5 and c["false_reject_rate"] == 0.5


def test_threshold_for_recall_edge_cases():
    y = [1, 1, 1, 1, 0, 0]
    s = [0.9, 0.7, 0.7, 0.3, 0.2, 0.8]
    # rappel ≥ 0.5 → 2 positifs → 2e plus grand score positif = 0.7 (le plus haut seuil possible)
    assert mt.threshold_for_recall(y, s, 0.5) == 0.7
    assert mt.confusion(y, s, 0.7)["recall"] == 0.75  # ex æquo : on gagne un positif de plus
    assert mt.threshold_for_recall(y, s, 0.75) == 0.7
    assert mt.threshold_for_recall(y, s, 0.76) == 0.3
    assert mt.threshold_for_recall(y, s, 1.0) == 0.3
    assert mt.threshold_for_recall(y, s, 0.25) == 0.9
    assert mt.threshold_for_recall(y, s, 0.0) == 1.0
    assert mt.confusion(y, s, mt.threshold_for_recall([1], [1.0], 0.0))["tp"] == 0
    # le seuil trouvé est le plus haut : juste au-dessus, l'objectif n'est plus atteint
    for target in (0.3, 0.6, 0.95):
        thr = mt.threshold_for_recall(y, s, target)
        assert mt.confusion(y, s, thr)["recall"] >= target
        assert mt.confusion(y, s, np.nextafter(thr, 2))["recall"] < target
    with pytest.raises(mt.ObjectiveError):
        mt.threshold_for_recall([0, 0], [0.1, 0.2], 0.9)
    with pytest.raises(mt.ObjectiveError):
        mt.threshold_for_recall(y, s, 1.2)


def test_threshold_for_false_reject_edge_cases():
    y = [0, 0, 0, 0, 1, 1]
    s = [0.1, 0.4, 0.4, 0.6, 0.5, 0.9]
    # faux rejets ≤ 0 → au-dessus du plus grand score viable
    t0 = mt.threshold_for_false_reject(y, s, 0.0)
    assert 0.6 < t0 < 0.6 + 1e-12 and mt.confusion(y, s, t0)["fp"] == 0
    # ≤ 25 % → un viable toléré (0.6) ; les deux 0.4 doivent rester sous le seuil
    t25 = mt.threshold_for_false_reject(y, s, 0.25)
    assert 0.4 < t25 < 0.4 + 1e-12 and mt.confusion(y, s, t25)["false_reject_rate"] == 0.25
    # ≤ 50 % : les ex æquo à 0.4 feraient passer à 75 % → même seuil
    assert mt.threshold_for_false_reject(y, s, 0.5) == t25
    assert mt.threshold_for_false_reject(y, s, 1.0) == 0.0
    # le plus bas : juste en dessous, l'objectif n'est plus respecté
    for target in (0.0, 0.25, 0.6):
        thr = mt.threshold_for_false_reject(y, s, target)
        assert mt.confusion(y, s, thr)["false_reject_rate"] <= target
        if thr > 0:
            assert mt.confusion(y, s, np.nextafter(thr, -1))["false_reject_rate"] > target
    with pytest.raises(mt.ObjectiveError):
        mt.threshold_for_false_reject([1, 1], [0.1, 0.2], 0.1)


def test_binary_report_structure():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    s = np.clip(y * 0.4 + rng.random(500) * 0.6, 0, 1)
    defects = ["felure" if v else None for v in y]
    r = mt.binary_report(y, s, 0.5, defects=defects, max_points=50)
    assert r["n"] == 500 and len(r["roc"]["fpr"]) <= 52 and r["roc"]["thresholds"][0] is None
    assert r["roc"]["fpr"][-1] == 1.0 and r["pr"]["recall"][-1] == 1.0
    assert r["recall_by_defect"]["felure"]["n"] == int(y.sum())
    assert sum(r["histogram"]["viable"]) == int((y == 0).sum())


# ---------------------------------------------------------------------------
# Découpages
# ---------------------------------------------------------------------------
def _lots(project):
    with project.session() as s:
        return {li.id: li for li in sp.lot_inventory(s)}


def test_split_by_lot_invariants(project):
    lots = _lots(project)
    res = sp.preview_split(project, sp.SplitParams(group_by="lot", seed=3))
    a = res["assignments"]
    assert set(a) == set(lots)  # chaque lot utilisable dans exactement un sous-ensemble
    assert set(a.values()) <= set(sp.SUBSETS)
    summ = res["summary"]
    assert summ["leakage_check"]["ok"] and summ["leakage_check"]["lots_in_one_subset"]
    assert summ["total_eggs"] == sum(li.n for li in lots.values()) == 180
    # stratification : chaque sous-ensemble contient des viables et chaque type de défaut
    for sub in sp.SUBSETS:
        e = summ["subsets"][sub]
        assert e["viable"] > 0 and len(e["defects"]) == 6, (sub, e)
    # production (non labellisé) exclu
    assert not any(li.type == "production" for li in lots.values())
    # reproductible
    assert sp.preview_split(project, sp.SplitParams(group_by="lot", seed=3))["assignments"] == a


def test_split_by_session_and_holdouts(project):
    lots = _lots(project)
    res = sp.preview_split(project, sp.SplitParams(group_by="session"))
    by_date: dict[str, set[str]] = {}
    for lot_id, sub in res["assignments"].items():
        by_date.setdefault(lots[lot_id].session_date, set()).add(sub)
    assert all(len(v) == 1 for v in by_date.values())  # une session dans un seul sous-ensemble
    assert res["summary"]["leakage_check"]["ok"] and not res["summary"]["leakage_check"]["shared_sessions"]

    farm = "Ferme du Val"
    res = sp.preview_split(project, sp.SplitParams(group_by="lot", holdout_farms=[farm]))
    assert all(res["assignments"][i] == "test" for i, li in lots.items() if li.farm == farm)
    assert res["summary"]["leakage_check"]["holdout_respected"]

    day = sorted({li.session_date for li in lots.values()})[0]
    res = sp.preview_split(project, sp.SplitParams(group_by="session", holdout_dates=[day]))
    assert {res["assignments"][i] for i, li in lots.items() if li.session_date == day} == {"test"}
    assert {res["assignments"][i] for i, li in lots.items() if li.session_date != day} <= {"train", "val"}


def test_split_warnings_on_missing_class(project):
    lots = _lots(project)
    felure = [i for i, li in lots.items() if "felure" in li.counts]
    res = sp.preview_split(project, sp.SplitParams(group_by="lot", exclude_lots=felure[1:]))
    assert any("felure" in w for w in res["summary"]["warnings"])


def test_split_versioning_and_manual_adjust(client, project):
    r = client.post("/api/training/splits", json={"name": "Principal", "group_by": "lot", "seed": 2})
    assert r.status_code == 200, r.text
    s1 = r.json()
    assert s1["version"] == 1 and not s1["locked"]
    s2 = client.post("/api/training/splits", json={"name": "Principal", "group_by": "session"}).json()
    assert s2["version"] == 2

    lot_id = next(iter(s1["assignments"]))
    new_sub = "test" if s1["assignments"][lot_id] != "test" else "train"
    r = client.patch(f"/api/training/splits/{s1['id']}/assignments", json={"changes": {lot_id: new_sub}})
    assert r.status_code == 200, r.text
    assert r.json()["id"] == s1["id"] and not r.json()["new_version"]
    assert r.json()["assignments"][lot_id] == new_sub

    # Utilisé par un modèle → immuable : la modification crée une nouvelle version
    with project.session() as s:
        s.add(m.Model(id="factice", name="factice", split_id=s1["id"]))
    r = client.patch(f"/api/training/splits/{s1['id']}/assignments", json={"changes": {lot_id: "val"}})
    out = r.json()
    assert out["new_version"] and out["id"] != s1["id"] and out["version"] == 3 and out["name"] == "Principal"
    assert out["strategy"]["derived_from"] == s1["id"]
    detail = client.get(f"/api/training/splits/{s1['id']}").json()
    assert detail["locked"] and detail["assignments"][lot_id] == new_sub and detail["used_by"] == ["factice"]
    assert client.delete(f"/api/training/splits/{s1['id']}").status_code == 400
    assert client.delete(f"/api/training/splits/{s2['id']}").status_code == 200
    listing = client.get("/api/training/splits").json()
    assert listing["total"] == 2
    # session : déplacer un lot déplace toute sa session (move_group)
    s3 = client.post("/api/training/splits", json={"name": "Sessions", "group_by": "session"}).json()
    r = client.patch(f"/api/training/splits/{s3['id']}/assignments", json={"changes": {lot_id: "val"}}).json()
    assert r["summary"]["leakage_check"]["ok"]
    r = client.patch(f"/api/training/splits/{s3['id']}/assignments", json={"changes": {lot_id: "test"}, "move_group": False}).json()
    assert not r["summary"]["leakage_check"]["ok"] and any("Fuite" in w for w in r["summary"]["warnings"])


def test_split_preview_api_and_options(client, project):
    r = client.post("/api/training/splits/preview", json={"group_by": "lot", "ratios": [0.6, 0.2, 0.2]})
    assert r.status_code == 200 and len(r.json()["lots"]) == 21
    assert client.post("/api/training/splits/preview", json={"ratios": [0, 1, 1]}).status_code == 400
    opts = client.get("/api/training/options").json()
    assert {a["id"] for a in opts["architectures"]} == {"mobilenetv3_small_100", "mobilenetv3_large_100", "efficientnet_lite0"}
    assert all(a["pretrained"]["available"] for a in opts["architectures"])
    assert {p["id"] for p in opts["presets"]} == {"rapide", "standard", "int8"}
    assert len(opts["split"]["farms"]) == 2 and len(opts["split"]["dates"]) == 3


def test_no_usable_eggs(client, tmp_path):
    client.post("/api/project/create", json={"parent_dir": str(tmp_path), "name": "Vide"})
    r = client.post("/api/training/splits/preview", json={})
    assert r.status_code == 400 and "utilisable" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Jeu de données
# ---------------------------------------------------------------------------
def test_records_filter_verdicts(project):
    from eggvision.ml.dataset import count_records, load_records

    a = sp.preview_split(project, sp.SplitParams(group_by="lot"))["assignments"]
    with project.session() as s:
        labels = s.query(m.Label).join(m.Track).filter(m.Track.lot_id.in_(list(a))).order_by(m.Label.track_id).limit(4).all()
        labels[0].verdict = "incertain"
        labels[1].verdict = "exclu"
        labels[2].verdict = None
        excluded = {lab.track_id for lab in labels[:3]}
    recs = load_records(project, a)
    assert len(recs) == 177 and not excluded & {r.track_id for r in recs}
    assert all(len(r.view_paths) == 12 and os.path.isfile(r.view_paths[0]) for r in recs[:5])
    counts = count_records(recs)
    assert sum(c["n"] for c in counts.values()) == 177
    # labels relus uniquement : la 3e session n'est pas relue
    reviewed = load_records(project, a, reviewed_only=True)
    assert 0 < len(reviewed) < len(recs) and all(r.reviewed for r in reviewed)


def test_view_selection_and_augmentations(project):
    from eggvision.ml.dataset import (
        AugmentConfig,
        EggViewsDataset,
        ViewCache,
        augment_view,
        load_records,
        pick_view_indices,
    )
    from eggvision_shared.preprocess import PreprocessConfig, load_view, preprocess_views

    assert pick_view_indices(12, 6) == [0, 2, 4, 7, 9, 11]
    assert pick_view_indices(3, 7) == [0, 1, 2, 0, 1, 2, 0]
    rng = np.random.default_rng(0)
    idx = pick_view_indices(12, 5, rng)
    assert idx == sorted(idx) and len(set(idx)) == 5
    with pytest.raises(ValueError):
        pick_view_indices(0, 4)

    a = sp.preview_split(project, sp.SplitParams(group_by="lot"))["assignments"]
    recs = load_records(project, a)[:4]
    img = load_view(recs[0].view_paths[0])
    out = augment_view(img, AugmentConfig(), np.random.default_rng(1))
    assert out.shape == img.shape and out.dtype == np.uint8 and not np.array_equal(out, img)

    pre = PreprocessConfig(input_size=64)
    ds = EggViewsDataset(recs, 5, pre, augment=None, cache=ViewCache(10_000_000))
    x, y, i = ds[0]
    assert tuple(x.shape) == (5, 3, 64, 64) and float(y) in (0.0, 1.0) and i == 0
    # sans augmentation : exactement le prétraitement partagé
    ref = preprocess_views([load_view(recs[0].view_paths[k]) for k in pick_view_indices(12, 5)], pre)
    assert np.array_equal(x.numpy(), ref)
    ds[0]
    assert ds.cache.hits >= 5
    # augmentations reproductibles (graine, époque, rang)
    aug = EggViewsDataset(recs, 5, pre, augment=AugmentConfig(), seed=3, random_views=True)
    assert np.array_equal(aug[1][0].numpy(), aug[1][0].numpy())
    aug.set_epoch(1)
    first = aug[1][0].numpy()
    aug.set_epoch(2)
    assert not np.array_equal(first, aug[1][0].numpy())


# ---------------------------------------------------------------------------
# Réseau et poids pré-entraînés hors-ligne
# ---------------------------------------------------------------------------
def test_pretrained_weights_offline(monkeypatch, tmp_path):
    import torch
    from eggvision import settings
    from eggvision.ml import network
    from safetensors.torch import load_file

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    for arch in network.ARCHITECTURES:
        assert network.pretrained_status(arch)["available"], arch
    model, info = network.build_model("mobilenetv3_small_100")
    assert info["loaded"] and info["missing"] == 0 and info["warning"] is None
    ref = load_file(str(network.pretrained_path("mobilenetv3_small_100")))
    key = "conv_stem.weight"
    assert torch.equal(model.backbone.state_dict()[key], ref[key])
    assert network.pretrained_path("x").parent == settings.assets_dir() / "pretrained"

    # fichier absent → initialisation aléatoire + avertissement (jamais de téléchargement)
    monkeypatch.setattr(settings, "assets_dir", lambda: tmp_path)
    model2, info2 = network.build_model("mobilenetv3_small_100")
    assert not info2["loaded"] and "aléatoire" in info2["warning"]


def test_multiview_forward_and_aggregation():
    import torch
    from eggvision.ml.network import build_model

    for agg in ("max", "attention"):
        model, _ = build_model("mobilenetv3_small_100", aggregation=agg, pretrained=False)
        model.eval()
        with torch.no_grad():
            for v in (1, 4, 7):
                x = torch.randn(3, v, 3, 64, 64)
                egg, views = model(x)
                assert egg.shape == (3,) and views.shape == (3, v)
                assert torch.all((egg >= 0) & (egg <= 1))
                if agg == "max":
                    assert torch.allclose(egg, views.max(dim=1).values, atol=1e-6)
                else:
                    assert torch.all(egg <= views.max(dim=1).values + 1e-6)
                    assert torch.all(egg >= views.min(dim=1).values - 1e-6)


def test_dataloader_with_worker_processes(project):
    """``num_workers > 0`` (processus « spawn » sous macOS / Windows) : le jeu de données doit être sérialisable."""
    from eggvision.ml.dataset import AugmentConfig, EggViewsDataset, ViewCache, load_records
    from eggvision_shared.preprocess import PreprocessConfig
    from torch.utils.data import DataLoader

    a = sp.preview_split(project, sp.SplitParams(group_by="lot"))["assignments"]
    recs = load_records(project, a)[:6]
    ds = EggViewsDataset(recs, 3, PreprocessConfig(input_size=48), augment=AugmentConfig(), cache=ViewCache(1 << 24))
    batches = list(DataLoader(ds, batch_size=3, num_workers=2))
    assert len(batches) == 2 and tuple(batches[0][0].shape) == (3, 3, 3, 48, 48)
