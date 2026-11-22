"""Tests du module « simulation du tri avec trappes » (backend + API)."""

from __future__ import annotations

import csv
import io
import shutil
import time
from collections import Counter
from pathlib import Path

import pytest
from eggvision.db import models as m
from eggvision.simulation import service
from eggvision.simulation.composition import SimulationError, build_composition, load_test_pool, model_split
from eggvision.simulation.engine import (
    ACCEPTED_OK,
    EJECTED_OK,
    FALSE_REJECT,
    MISSED,
    MISSED_LATE,
    load_classifier,
    run,
)
from eggvision.simulation.params import SimulationParams
from eggvision.simulation.results import JOURNAL_FILE, aggregate
from eggvision.simulation.video import probe
from eggvision.testing import make_dummy_package, wait_job

MODEL_ID = "m-factice"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def dummy_package(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_dummy_package(tmp_path_factory.mktemp("pkg") / "package", input_size=64)


def _install_model(project, package: Path, model_id: str = MODEL_ID, *, test_sessions=(2,), name="Modèle factice", export=True, split=True) -> dict:
    """Ajoute un modèle exporté + un découpage par session (0 = train, 1 = val, 2 = test par défaut)."""
    with project.session() as s:
        lots = s.query(m.Lot).all()
        dates = sorted({lot.session_date for lot in lots})
        assignments = {}
        for lot in lots:
            idx = dates.index(lot.session_date)
            assignments[lot.id] = "test" if idx in test_sessions else ("train" if idx == 0 else "val")
        split_row = None
        if split:
            split_row = m.DatasetSplit(name="Découpage test", assignments=assignments, strategy={"group_by": "session"})
            s.add(split_row)
            s.flush()
        export_info = {}
        if export:
            dest = project.model_dir(model_id) / "package"
            shutil.copytree(package, dest)
            export_info = {"package_dir": project.relpath(dest), "variants": ["fp32"]}
        s.add(
            m.Model(
                id=model_id,
                name=name,
                status="trained",
                split_id=split_row.id if split_row else None,
                threshold=0.5,
                n_views=12,
                input_size=64,
                export=export_info,
            )
        )
    return assignments


@pytest.fixture()
def sim_project(project, dummy_package):
    assignments = _install_model(project, dummy_package)
    project.assignments = assignments  # type: ignore[attr-defined]
    return project


def _params(**kw) -> SimulationParams:
    base = {"model_id": MODEL_ID, "n_columns": 40, "seed": 123}
    base.update(kw)
    return SimulationParams(**base)


def _compose(project, params: SimulationParams):
    _, split = model_split(project, params.model_id)
    pool = load_test_pool(project, split)
    return pool, build_composition(pool, params, 12)


def _run(project, params: SimulationParams):
    _, split = model_split(project, params.model_id)
    with project.session() as s:
        export = dict(s.get(m.Model, params.model_id).export)
    clf = load_classifier(project, export, params)
    pool = load_test_pool(project, split)
    comp = build_composition(pool, params, clf.n_views)
    out = run(project, params, comp, clf)
    return comp, out, aggregate(out, params, comp.info)


def _test_track_ids(project) -> set[int]:
    test_lots = [lot for lot, sub in project.assignments.items() if sub == "test"]
    with project.session() as s:
        return {t for (t,) in s.query(m.Track.id).filter(m.Track.lot_id.in_(test_lots))}


# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------
def test_params_defaults_and_timing():
    p = SimulationParams(model_id="x")
    assert p.n_lanes == 6 and p.period_s == 0.5 and p.pitch_mm == 76
    assert p.speed == pytest.approx(152.0)
    t = p.timing()
    assert t["budget_ms"] == pytest.approx(600 / 152 * 1000, rel=1e-6)
    assert t["effective_budget_ms"] == pytest.approx(600 / 152 * 1000 - 40, rel=1e-6)
    assert t["columns_in_flight"] == 10 and not t["warnings"]
    # Saisie de la vitesse → le pas est recalculé.
    q = SimulationParams(model_id="x", speed_mm_s=200, period_s=0.4)
    assert q.pitch_mm == pytest.approx(80) and q.speed == pytest.approx(200)
    # Temps d'ouverture > budget : avertissement.
    assert SimulationParams(model_id="x", camera_to_trap_mm=5, trap_actuation_ms=100).timing()["warnings"]
    with pytest.raises(ValueError):
        SimulationParams(model_id="x", defect_rate=1.5)
    with pytest.raises(ValueError):
        SimulationParams(model_id="x", unknown=1)


# ---------------------------------------------------------------------------
# Constitution des colonnes
# ---------------------------------------------------------------------------
def test_test_set_only_and_defect_rate(sim_project):
    pool, comp = _compose(sim_project, _params(n_columns=300, defect_rate=0.3))
    test_ids = _test_track_ids(sim_project)
    assert pool.summary()["n_viable"] == 24 and pool.summary()["n_non_viable"] == 36
    assert pool.excluded_no_verdict == 18  # le lot « production » (sans verdict) est exclu
    slots = [s for col in comp.columns for s in col if s is not None]
    assert len(slots) == 300 * 6
    assert {s.egg.track_id for s in slots} <= test_ids  # aucune fuite train / val
    assert all(sim_project.assignments[s.egg.lot_id] == "test" for s in slots)
    n_def = sum(1 for s in slots if s.egg.verdict == "non_viable")
    assert n_def == round(0.3 * len(slots))
    info = comp.info
    assert info["with_replacement"] and info["n_distinct_eggs"] == 60 and info["reuse_rate"] > 0.9
    # Tirage avec remise équilibré : chaque œuf viable réutilisé le même nombre de fois (à un près).
    uses = Counter(s.egg.track_id for s in slots if s.egg.verdict == "viable")
    assert max(uses.values()) - min(uses.values()) <= 1
    # Petit tirage : sans remise.
    _, small = _compose(sim_project, _params(n_columns=5, defect_rate=0.2))
    assert not small.info["with_replacement"] and small.info["reuse_rate"] == 0


def test_reproducible_by_seed(sim_project):
    _, a = _compose(sim_project, _params(seed=7, empty_lane_rate=0.2, missing_view_rate=0.1))
    _, b = _compose(sim_project, _params(seed=7, empty_lane_rate=0.2, missing_view_rate=0.1))
    _, c = _compose(sim_project, _params(seed=8, empty_lane_rate=0.2, missing_view_rate=0.1))
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()
    # Exécution complète : mêmes décisions et scores (seules les latences mesurées diffèrent).
    _, r1, _ = _run(sim_project, _params(seed=7))
    _, r2, _ = _run(sim_project, _params(seed=7))
    key = lambda out: [(e["column"], e["lane"], e["track_id"], e["decision"], e["score"], e["outcome"]) for e in out.eggs]  # noqa: E731
    assert key(r1) == key(r2)


def test_empty_lanes(sim_project):
    comp, out, summary = _run(sim_project, _params(n_columns=60, empty_lane_rate=0.5))
    n_empty = sum(1 for col in comp.columns for s in col if s is None)
    assert 0.35 < n_empty / 360 < 0.65
    assert summary["counts"]["eggs"] == 360 - n_empty == len(out.eggs)
    assert all(col["lanes"][e["lane"]] is not None for e in out.eggs for col in [out.columns[e["column"]]])
    with pytest.raises(SimulationError, match="vides"):
        _compose(sim_project, _params(n_columns=1, empty_lane_rate=0.95, seed=1, n_lanes=1))


def test_composition_errors(project, dummy_package):
    # Modèle non exporté.
    _install_model(project, dummy_package, "m-non-exporte", export=False)
    with pytest.raises(SimulationError, match="pas exporté"):
        service.create(project, _params(model_id="m-non-exporte"))
    # Modèle sans découpage.
    _install_model(project, dummy_package, "m-sans-split", split=False)
    with pytest.raises(SimulationError, match="découpage"):
        service.create(project, _params(model_id="m-sans-split"))
    # Jeu test vide.
    _install_model(project, dummy_package, "m-test-vide", test_sessions=())
    with pytest.raises(SimulationError, match="aucun lot de test"):
        service.create(project, _params(model_id="m-test-vide"))
    with pytest.raises(SimulationError, match="introuvable"):
        service.create(project, _params(model_id="inconnu"))


def test_no_non_viable_in_test_set(project, dummy_package):
    _install_model(project, dummy_package, "m-viables")
    with project.session() as s:
        split = s.get(m.DatasetSplit, s.get(m.Model, "m-viables").split_id)
        lots = {lot.id: lot.type for lot in s.query(m.Lot)}
        split.assignments = {k: ("test" if (v == "test" and lots[k] == "viable") else "train") for k, v in split.assignments.items()}
    with pytest.raises(SimulationError, match="aucun œuf non viable"):
        service.create(project, _params(model_id="m-viables", defect_rate=0.05))
    # Sans défaut demandé, la simulation est possible.
    _, comp = _compose(project, _params(model_id="m-viables", defect_rate=0.0, n_columns=3))
    assert comp.info["n_non_viable"] == 0


# ---------------------------------------------------------------------------
# Temps : budget, retards, file d'attente
# ---------------------------------------------------------------------------
def test_budget_and_late_decisions(sim_project):
    # Sans latence injectée : le modèle factice répond en quelques ms ≪ budget (≈ 3,9 s).
    _, out, summary = _run(sim_project, _params(defect_rate=0.3))
    assert summary["counts"]["late_decisions"] == 0
    assert summary["latency"]["min_margin_ms"] > 3000
    assert summary["latency"]["budget_ms"] == pytest.approx(600 / 152 * 1000, rel=1e-4)
    col = out.columns[5]
    assert col["t_enter"] == pytest.approx(2.5)
    assert col["t_acq"] == pytest.approx(2.5 + 1.0)  # zone caméra 152 mm à 152 mm/s
    assert col["t_trap"] == pytest.approx(col["t_acq"] + 600 / 152, abs=1e-5)
    assert col["t_deadline"] == pytest.approx(col["t_trap"] - 0.040, abs=1e-5)

    # Latence injectée > budget : toutes les décisions en retard, aucune trappe ne s'ouvre.
    _, out, summary = _run(sim_project, _params(defect_rate=0.3, injected_latency_ms=5000, workers=16))
    c = summary["counts"]
    assert c["late_decisions"] == c["eggs"]
    assert c["ejected_ok"] == 0 and c["false_reject"] == 0 and c["bin_reject"] == 0
    assert c["missed_late"] == sum(1 for e in out.eggs if e["truth"] == "non_viable" and e["decision"] == 1)
    assert c["missed_late"] > 0
    assert all(e["outcome"] in (MISSED, MISSED_LATE, ACCEPTED_OK) for e in out.eggs)


def test_queue_accumulates_latency(sim_project):
    # 1 unité d'inférence, 700 ms par colonne pour une colonne toutes les 500 ms : l'attente s'accumule.
    params = _params(n_columns=40, injected_latency_ms=700)
    _, out, summary = _run(sim_project, params)
    waits = [c["wait_ms"] for c in out.columns]
    assert waits[0] == pytest.approx(0, abs=1e-6)
    assert all(b >= a for a, b in zip(waits, waits[1:], strict=False))
    # attente(k) = k × (700 + mesurée − 500) ms environ
    assert waits[20] == pytest.approx(20 * 200, rel=0.15)
    late = [c["late"] for c in out.columns]
    first_late = late.index(True)
    assert all(late[first_late:]) and not any(late[:first_late])
    # Budget effectif 3907 ms : retard dès que k·200 + 700 > 3907, soit k ≥ 17 (à la latence mesurée près).
    assert 13 <= first_late <= 17
    assert summary["latency"]["wait_ms"]["max"] > 7000
    assert summary["throughput"]["capacity_ratio"] < 1

    # Deux unités en parallèle : capacité suffisante, aucune attente, aucun retard.
    _, out2, summary2 = _run(sim_project, params.model_copy(update={"workers": 2}))
    assert max(c["wait_ms"] for c in out2.columns) == pytest.approx(0, abs=1e-6)
    assert summary2["counts"]["late_decisions"] == 0


def test_missing_views(sim_project):
    _, out, summary = _run(sim_project, _params(missing_view_rate=1.0, defect_rate=0.3))
    assert summary["counts"]["no_decision"] == summary["counts"]["eggs"]
    assert summary["counts"]["bin_reject"] == 0
    assert all(e["decision"] is None and e["n_views"] == 0 for e in out.eggs)
    assert summary["counts"]["missed"] == summary["counts"]["non_viable"]
    comp, out, summary = _run(sim_project, _params(missing_view_rate=0.5))
    n = [e["n_views"] for e in out.eggs]
    assert 0.35 < sum(n) / (12 * len(n)) < 0.65
    assert comp.info["n_views_missing"] == 12 * len(n) - sum(n)


def test_aggregates_consistent_with_journal(client, sim_project):
    r = client.post("/api/simulation", json={"model_id": MODEL_ID, "n_columns": 80, "defect_rate": 0.3, "seed": 5, "empty_lane_rate": 0.1})
    assert r.status_code == 200, r.text
    sim_id = r.json()["simulation"]["id"]
    wait_job(r.json()["job"]["id"], 120)
    res = client.get(f"/api/simulation/{sim_id}").json()["results"]
    c = res["counts"]
    text = client.get(f"/api/simulation/{sim_id}/journal.csv").content.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text), delimiter=";"))
    assert len(rows) == c["eggs"]
    assert list(rows[0].keys())[:6] == ["colonne", "couloir", "piste", "lot", "verite", "defaut"]
    issues = Counter(row["issue"] for row in rows)
    assert issues["Éjecté à raison"] == c[EJECTED_OK]
    assert issues["Manqué"] == c[MISSED] and issues["Manqué (retard)"] == c[MISSED_LATE]
    assert issues["Éjecté à tort"] == c[FALSE_REJECT] and issues["Accepté à raison"] == c[ACCEPTED_OK]
    assert c[EJECTED_OK] + c[MISSED] + c[MISSED_LATE] == c["non_viable"]
    assert c[FALSE_REJECT] + c[ACCEPTED_OK] == c["viable"]
    assert c["bin_reject"] + c["bin_viable"] == c["eggs"] == c["viable"] + c["non_viable"]
    assert sum(d["n"] for d in res["by_defect"].values()) == c["non_viable"]
    assert sum(d[EJECTED_OK] for d in res["by_defect"].values()) == c[EJECTED_OK]
    assert sum(1 for row in rows if row["verite"] == "non_viable") == c["non_viable"]
    assert sum(1 for row in rows if row["en_retard"] == "oui") == c["late_decisions"]
    assert res["rates"]["recall"] == pytest.approx(c[EJECTED_OK] / c["non_viable"], abs=1e-6)
    assert sum(res["latency"]["histogram"]["counts"]) == res["n_columns"]
    # Le modèle factice répond selon l'image : les scores varient d'un œuf à l'autre.
    scores = [float(row["score"]) for row in rows if row["score"]]
    assert all(0.0 <= x <= 1.0 for x in scores) and max(scores) - min(scores) > 1e-3
    # Fichiers relatifs au projet.
    assert (sim_project.root / res["files"]["journal"]).is_file()
    assert (sim_project.simulation_dir(sim_id) / JOURNAL_FILE).is_file()
    assert (sim_project.simulation_dir(sim_id) / "params.json").is_file()


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
def test_api_flow(client, sim_project):
    assert client.get("/api/simulation/defaults").json()["params"]["n_lanes"] == 6
    t = client.post("/api/simulation/timing", json={"pitch_mm": 100, "period_s": 0.5, "camera_to_trap_mm": 400}).json()
    assert t["speed_mm_s"] == 200 and t["budget_ms"] == 2000
    models = client.get("/api/simulation/models").json()
    assert [md["id"] for md in models] == [MODEL_ID]
    assert models[0]["eligible"] and models[0]["test_set"]["n_non_viable"] == 36 and models[0]["variants"] == ["fp32"]

    r = client.post("/api/simulation", json={"model_id": MODEL_ID, "n_columns": 30, "seed": 11, "defect_rate": 0.3, "name": "Essai A"})
    assert r.status_code == 200, r.text
    sim_a = r.json()["simulation"]["id"]
    wait_job(r.json()["job"]["id"], 60)
    d = client.get(f"/api/simulation/{sim_a}").json()
    assert d["status"] == "done" and d["name"] == "Essai A" and d["results"]["threshold"] == 0.5
    assert d["timing"]["budget_ms"] > 0 and d["job"]["status"] == "done"

    tl = client.get(f"/api/simulation/{sim_a}/timeline", params={"start": 10, "count": 5}).json()
    assert tl["total"] == 30 and tl["start"] == 10 and len(tl["columns"]) == 5 and tl["columns"][0]["k"] == 10
    assert tl["geometry"]["n_lanes"] == 6 and "view_scores" not in (tl["columns"][0]["lanes"][0] or {})
    lane = next(i for i, e in enumerate(tl["columns"][0]["lanes"]) if e)
    egg = client.get(f"/api/simulation/{sim_a}/egg/10/{lane}").json()
    assert egg["labeling_url"] == f"/labellisation/piste/{egg['track_id']}"
    assert len(egg["views"]) == 12 and all(v["presented"] for v in egg["views"]) and egg["views"][0]["score"] is not None
    assert client.get(egg["views"][0]["url"]).status_code == 200
    assert client.get(f"/api/simulation/{sim_a}/egg/999/0").status_code == 404

    # Duplication avec un autre seuil : même composition, décisions différentes.
    r = client.post(f"/api/simulation/{sim_a}/duplicate", json={"threshold": 0.999})
    assert r.status_code == 200, r.text
    sim_b = r.json()["simulation"]["id"]
    wait_job(r.json()["job"]["id"], 60)
    b = client.get(f"/api/simulation/{sim_b}").json()
    assert b["params"]["seed"] == 11 and b["results"]["threshold"] == 0.999 and "seuil 0,999" in b["name"]
    assert b["results"]["counts"]["bin_reject"] <= d["results"]["counts"]["bin_reject"]

    cmp = client.get("/api/simulation/compare", params=[("ids", sim_a), ("ids", sim_b)]).json()
    assert [s["id"] for s in cmp["simulations"]] == [sim_a, sim_b]
    differing = {p["key"] for p in cmp["params"] if p["differs"]}
    assert differing == {"threshold"}
    recall = next(row for row in cmp["metrics"] if row["key"] == "rates.recall")
    assert len(recall["values"]) == 2
    assert {row["defect"] for row in cmp["by_defect"]} == set(d["results"]["by_defect"])
    assert client.get("/api/simulation/compare", params={"ids": "absent"}).status_code == 404

    listing = client.get("/api/simulation", params={"limit": 1}).json()
    assert listing["total"] == 2 and len(listing["items"]) == 1
    assert client.post("/api/simulation", json={"model_id": "absent"}).status_code == 400
    assert client.post("/api/simulation", json={"model_id": MODEL_ID, "defect_rate": 2}).status_code == 422

    assert client.delete(f"/api/simulation/{sim_b}").json() == {"deleted": sim_b}
    assert not sim_project.simulation_dir(sim_b).exists()
    assert client.get(f"/api/simulation/{sim_b}").status_code == 404
    assert client.get(f"/api/simulation/{sim_a}/video").status_code == 404


def test_cancel(client, sim_project):
    r = client.post("/api/simulation", json={"model_id": MODEL_ID, "n_columns": 20000, "seed": 3})
    assert r.status_code == 200, r.text
    sim_id = r.json()["simulation"]["id"]
    time.sleep(0.5)
    assert client.delete(f"/api/simulation/{sim_id}").status_code == 409  # en cours
    client.post(f"/api/simulation/{sim_id}/cancel")
    wait_job(r.json()["job"]["id"], 60, expect="cancelled")
    d = client.get(f"/api/simulation/{sim_id}").json()
    assert d["status"] == "cancelled"
    assert client.get(f"/api/simulation/{sim_id}/timeline").status_code == 409
    assert client.delete(f"/api/simulation/{sim_id}").status_code == 200


def test_leak_guard_raises(sim_project):
    """Le garde-fou détecte une piste hors jeu test même si le chargement du jeu était fautif."""
    _, split = model_split(sim_project, MODEL_ID)
    pool = load_test_pool(sim_project, split)
    train_lot = next(lot for lot, sub in sim_project.assignments.items() if sub == "train")
    pool.viable[0].lot_id = train_lot
    with pytest.raises(SimulationError, match="Fuite"):
        build_composition(pool, _params(n_columns=50, defect_rate=0.0), 12)


# ---------------------------------------------------------------------------
# Critère d'acceptation 5 : 1 000 colonnes de bout en bout, rapport chiffré, vidéo MP4 lisible
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_acceptance_1000_columns_report_and_video(client, sim_project, capsys):
    t0 = time.perf_counter()
    r = client.post("/api/simulation", json={"model_id": MODEL_ID, "n_columns": 1000, "seed": 2024, "defect_rate": 0.05, "name": "Critère 5"})
    assert r.status_code == 200, r.text
    sim_id = r.json()["simulation"]["id"]
    job = wait_job(r.json()["job"]["id"], 600)
    t_sim = time.perf_counter() - t0
    assert job["result"]["columns_done"] == 1000

    d = client.get(f"/api/simulation/{sim_id}").json()
    res = d["results"]
    c = res["counts"]
    assert d["status"] == "done" and res["n_columns"] == 1000
    assert c["eggs"] == 6000 and c["non_viable"] == 300  # 5 % exactement
    assert c["ejected_ok"] + c["missed_total"] == c["non_viable"]
    assert c["false_reject"] + c["accepted_ok"] == c["viable"]
    for key in ("recall", "miss_rate", "false_reject_rate", "late_rate"):
        assert res["rates"][key] is not None
    assert res["latency"]["compute_ms"]["p99"] is not None and res["latency"]["budget_ms"] > 0
    assert res["composition"]["n_distinct_eggs"] == 60 and res["composition"]["with_replacement"]
    tl = client.get(f"/api/simulation/{sim_id}/timeline", params={"start": 900, "count": 200}).json()
    assert len(tl["columns"]) == 100
    csv_text = client.get(f"/api/simulation/{sim_id}/journal.csv").content.decode("utf-8-sig")
    assert len(csv_text.strip().splitlines()) == 6001

    # Vidéo accélérée et courte (rendu rapide).
    t1 = time.perf_counter()
    r = client.post(f"/api/simulation/{sim_id}/video", json={"fps": 10, "width": 640, "height": 360, "max_duration_s": 20})
    assert r.status_code == 200, r.text
    vjob = wait_job(r.json()["job"]["id"], 600)
    t_video = time.perf_counter() - t1
    info = vjob["result"]["video"]
    assert info["encoder"] == "libx264" and info["speed"] > 20
    v = client.get(f"/api/simulation/{sim_id}/video")
    assert v.status_code == 200 and v.headers["content-type"] == "video/mp4"
    assert v.content[4:8] == b"ftyp"  # conteneur MP4
    path = sim_project.simulation_dir(sim_id) / "video.mp4"
    probed = probe(path)
    assert abs(probed["n_frames"] - info["n_frames"]) <= 1
    assert probed["duration_s"] == pytest.approx(info["duration_s"], abs=0.25)
    assert client.get(f"/api/simulation/{sim_id}").json()["results"]["video"]["n_frames"] == info["n_frames"]

    with capsys.disabled():
        print(
            f"\n[critère 5] simulation 1000 colonnes : {t_sim:.1f} s (moteur {res['throughput']['wall_s']:.1f} s, "
            f"latence de calcul moyenne {res['latency']['compute_ms']['mean']:.2f} ms, p99 {res['latency']['compute_ms']['p99']:.2f} ms) ; "
            f"rappel effectif {res['rates']['recall']:.3f}, faux rejets {res['rates']['false_reject_rate']:.3f}, retards {c['late_decisions']} ; "
            f"vidéo {info['n_frames']} images {info['width']}x{info['height']} ×{info['speed']:.1f} en {t_video:.1f} s "
            f"({info['size_bytes'] / 1e6:.2f} Mo ; contrôle {probed['tool']} : {probed['n_frames']} images, {probed['duration_s']:.2f} s)"
        )
