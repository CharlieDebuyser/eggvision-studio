"""Volumétrie de la labellisation : 100 000 pistes (sans images) et 500 labellisations successives.

Les temps mesurés sont affichés (``pytest -s``) ; les assertions gardent une marge large pour ne pas
dépendre de la machine, l'objectif visé étant < 150 ms par page.
"""

from __future__ import annotations

import random
import statistics
import time
from pathlib import Path

import pytest
from eggvision.db import models as m

API = "/api/labeling"
N_TRACKS = 100_000
N_LOTS = 100
TARGET_MS = 150.0
LIMIT_MS = 600.0  # garde-fou des assertions (machines lentes / CI)


@pytest.fixture(scope="module")
def big_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from eggvision.project import Project
    from eggvision_shared.constants import DEFECT_TYPES

    root = tmp_path_factory.mktemp("volume") / "projet"
    project = Project.create(root, "Volumétrie")
    project.update_config({"user": {"name": "Relecteur"}})
    rng = random.Random(7)
    per_lot = N_TRACKS // N_LOTS
    t0 = time.perf_counter()
    with project.session() as s:
        lots, tracks, labels, history, preds = [], [], [], [], []
        tid = 0
        for i in range(N_LOTS):
            if i % 10 == 9:
                lot_type = "production"
            elif i % 2 == 0:
                lot_type = "viable"
            else:
                lot_type = f"declasse:{DEFECT_TYPES[i % len(DEFECT_TYPES)]}"
            lot_id = f"2026{i:04d}-lot"
            lots.append({"id": lot_id, "name": f"Lot {i}", "type": lot_type, "farm": f"Élevage {i % 4}", "session_date": f"2026-01-{1 + i % 28:02d}",
                         "status": "termine", "created_at": m.utcnow(), "track_count": per_lot, "detection_status": "done"})
            verdict, defect = ("viable", None) if lot_type == "viable" else (None, None) if lot_type == "production" else ("non_viable", lot_type.split(":")[1])
            for k in range(per_lot):
                tid += 1
                tracks.append({"id": tid, "lot_id": lot_id, "lane": k % 6, "start_frame": (k // 6) * 7, "end_frame": (k // 6) * 7 + 20,
                               "n_views": 12, "n_detections": 20, "quality": "douteuse" if rng.random() < 0.02 else "ok", "flags": [], "extra": {}})
                reviewed = i < 40
                labels.append({"track_id": tid, "verdict": verdict, "defect": defect, "source": "humain" if reviewed else "auto",
                               "reviewed": reviewed, "reviewer": "démo" if reviewed else "", "updated_at": m.utcnow(), "note": ""})
                history.append({"track_id": tid, "user": "auto", "at": m.utcnow(), "action": "prelabel", "before": {}, "after": {"verdict": verdict, "defect": defect}})
                preds.append({"model_id": "m-vol", "track_id": tid, "subset": "", "score": rng.random(), "view_scores": []})
        s.execute(m.Lot.__table__.insert(), lots)
        s.add(m.Model(id="m-vol", name="Modèle volumétrie", status="trained", threshold=0.5))
        s.flush()
        for table, rows in ((m.Track, tracks), (m.Label, labels), (m.LabelHistory, history), (m.Prediction, preds)):
            for j in range(0, len(rows), 20_000):
                s.execute(table.__table__.insert(), rows[j : j + 20_000])
    print(f"\n[volumétrie] insertion de {N_TRACKS} pistes + labels + historique + prédictions : {time.perf_counter() - t0:.1f} s")
    project.close()
    return root


def _timed(client, url: str, params=None, repeat: int = 5) -> tuple[float, dict]:
    times, data = [], None
    for _ in range(repeat):
        t = time.perf_counter()
        r = client.get(url, params=params)
        times.append((time.perf_counter() - t) * 1000)
        assert r.status_code == 200, r.text
        data = r.json()
    return statistics.median(times), data


def test_volume_queries(client, big_template):
    from eggvision import state

    state.projects.open(big_template)
    client.get(f"{API}/meta")  # création des index complémentaires (une fois)

    results: dict[str, float] = {}
    cases = {
        "page naturelle + total": (f"{API}/tracks", {"limit": 100}),
        "page naturelle sans total": (f"{API}/tracks", {"limit": 100, "with_total": False}),
        "filtre à relire + total": (f"{API}/tracks", {"limit": 100, "reviewed": "false"}),
        "filtre défaut coulure + total": (f"{API}/tracks", {"limit": 100, "defect": "coulure"}),
        "filtre lot + verdict": (f"{API}/tracks", {"limit": 100, "lot": "20260051-lot", "verdict": "non_viable"}),
        "filtre sans label + douteuse": (f"{API}/tracks", {"limit": 100, "verdict": "aucun", "quality": "douteuse"}),
        "filtre confiance 0.45–0.55": (f"{API}/tracks", {"limit": 100, "score_min": 0.45, "score_max": 0.55}),
        "tri récent": (f"{API}/tracks", {"limit": 100, "sort": "recent"}),
        "tri priorité": (f"{API}/tracks", {"limit": 100, "sort": "priorite"}),
        "file apprentissage actif": (f"{API}/active", {"limit": 100}),
        "historique global": (f"{API}/history", {"limit": 100}),
        "historique d'une piste": (f"{API}/history", {"limit": 50, "track_id": 77_777}),
        "détail d'une piste": (f"{API}/tracks/55555", None),
        "tableau de bord": (f"{API}/dashboard", None),
    }
    for name, (url, params) in cases.items():
        results[name], data = _timed(client, url, params)
        if name == "page naturelle + total":
            assert data["total"] == N_TRACKS
        if name == "filtre à relire + total":
            assert data["total"] == 60_000

    # page lointaine par curseur (fin du jeu de données) : temps constant
    page = client.get(f"{API}/tracks", params={"limit": 100, "anchor": 99_000, "with_total": False}).json()
    results["page lointaine (ancre 99 000)"], _ = _timed(client, f"{API}/tracks", {"limit": 100, "cursor": page["next_cursor"], "with_total": False})
    results["page lointaine (offset 90 000)"], _ = _timed(client, f"{API}/tracks", {"limit": 100, "offset": 90_000, "with_total": False})

    print("\n[volumétrie] temps médians (100 000 pistes) :")
    for name, ms in results.items():
        flag = "" if ms < TARGET_MS else "  (> objectif 150 ms)"
        print(f"  {name:<34} {ms:7.1f} ms{flag}")

    page_cases = [k for k in results if k.startswith(("page", "filtre", "tri", "file", "historique", "détail"))]
    for name in page_cases:
        assert results[name] < LIMIT_MS, (name, results[name])
    assert results["tableau de bord"] < 3 * LIMIT_MS


def test_500_successive_labels(client, big_template):
    """Boucle de relecture au clavier : une écriture par œuf (+ détail de la piste suivante, préchargé)."""
    from eggvision import state

    state.projects.open(big_template)
    client.get(f"{API}/meta")
    keys = [{"verdict": "viable"}] * 6 + [{"verdict": "non_viable"}, {"defect": "felure"}, {"verdict": "incertain"}, {"verdict": "exclu"}]
    put_ms, detail_ms, page_ms = [], [], []
    cursor = None
    done = 0
    t_all = time.perf_counter()
    while done < 500:
        params = {"limit": 100, "reviewed": "false", "lot_type": "production", "with_total": cursor is None}
        if cursor:
            params["cursor"] = cursor
        t = time.perf_counter()
        page = client.get(f"{API}/tracks", params=params).json()
        page_ms.append((time.perf_counter() - t) * 1000)
        cursor = page["next_cursor"]
        for item in page["items"]:
            if done >= 500:
                break
            t = time.perf_counter()
            assert client.get(f"{API}/tracks/{item['id']}").status_code == 200
            detail_ms.append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            r = client.put(f"{API}/tracks/{item['id']}/label", json=keys[done % len(keys)])
            put_ms.append((time.perf_counter() - t) * 1000)
            assert r.status_code == 200 and r.json()["label"]["reviewed"]
            done += 1
    total_s = time.perf_counter() - t_all
    print(
        f"\n[500 labellisations] écriture : moyenne {statistics.mean(put_ms):.1f} ms, p95 {sorted(put_ms)[474]:.1f} ms ; "
        f"détail : moyenne {statistics.mean(detail_ms):.1f} ms ; page de 100 : moyenne {statistics.mean(page_ms):.1f} ms ; "
        f"total {total_s:.1f} s ({total_s / 500 * 1000:.1f} ms par œuf, hors temps humain)"
    )
    assert statistics.mean(put_ms) < 100
    dash = client.get(f"{API}/dashboard").json()
    assert dash["totals"]["reviewed"] == 40_000 + 500
    u = client.post(f"{API}/undo").json()
    assert u["undone"] == 1
