"""Module de labellisation : filtres, tris, pagination, règles, historique, annulation, masse,
vues marquées, tableau de bord, apprentissage actif.

Fixture ``project`` : 198 pistes (3 sessions × (24 viables + 6 × 6 déclassées) + 18 en production sans label),
les deux premières sessions relues, relecteur « démo ».
"""

from __future__ import annotations

import pytest
from eggvision.db import models as m
from eggvision.labeling import ActiveModel, evaluate
from sqlalchemy import select

API = "/api/labeling"


def _tracks(client, **params):
    r = client.get(f"{API}/tracks", params=params)
    assert r.status_code == 200, r.text
    return r.json()


def _all_ids(client, page_size: int = 37, **params) -> list[int]:
    ids: list[int] = []
    cursor = None
    while True:
        q = dict(params, limit=page_size, with_total=False)
        if cursor:
            q["cursor"] = cursor
        page = _tracks(client, **q)
        ids += [i["id"] for i in page["items"]]
        cursor = page["next_cursor"]
        if not cursor:
            return ids


def _ids_where(project, *conds) -> list[int]:
    with project.session() as s:
        q = select(m.Track.id).outerjoin(m.Label, m.Label.track_id == m.Track.id).join(m.Lot, m.Lot.id == m.Track.lot_id)
        return list(s.scalars(q.where(*conds)))


# -- recherche ---------------------------------------------------------------
def test_search_pagination_natural_order(client, project):
    first = _tracks(client, limit=10)
    assert first["total"] == 198 and len(first["items"]) == 10 and first["next_cursor"]
    item = first["items"][0]
    assert {"id", "lot_id", "lot_type", "lane", "n_views", "verdict", "defect", "reviewed", "prelabel", "quality"} <= set(item)
    assert item["n_views"] == 12

    ids = _all_ids(client)
    assert len(ids) == 198 and len(set(ids)) == 198
    with project.session() as s:
        expected = list(s.scalars(select(m.Track.id).order_by(m.Track.lot_id, m.Track.start_frame, m.Track.lane, m.Track.id)))
    assert ids == expected

    # saut direct (offset) et ancre (lien profond)
    assert [i["id"] for i in _tracks(client, limit=5, offset=20)["items"]] == expected[20:25]
    anchored = _tracks(client, limit=5, anchor=expected[50], with_total=False)
    assert [i["id"] for i in anchored["items"]] == expected[50:55]
    assert client.get(f"{API}/tracks", params={"anchor": 999999}).status_code == 404
    assert client.get(f"{API}/tracks", params={"cursor": "n'importe-quoi"}).status_code == 400


def test_search_filters(client, project):
    assert _tracks(client, verdict="aucun")["total"] == 18  # lot production
    assert _tracks(client, lot_type="production")["total"] == 18
    assert _tracks(client, lot_type="declasse")["total"] == 108
    assert _tracks(client, lot_type="declasse:felure")["total"] == 18
    assert _tracks(client, defect="felure")["total"] == 18
    assert _tracks(client, verdict="non_viable", reviewed="true")["total"] == 72
    assert _tracks(client, reviewed="false")["total"] == 78
    assert _tracks(client, verdict=["viable", "aucun"])["total"] == 72 + 18
    assert _tracks(client, verdict="viable,aucun")["total"] == 72 + 18
    assert _tracks(client, quality="douteuse")["total"] == 0
    assert _tracks(client, source="auto")["total"] == 78

    lane0 = _ids_where(project, m.Track.lane == 0)
    assert _tracks(client, lane=0)["total"] == len(lane0) > 0
    with project.session() as s:
        lot_id = s.scalar(select(m.Lot.id).where(m.Lot.type == "declasse:sang"))
    page = _tracks(client, lot=lot_id, verdict="non_viable", defect="sang", lane=[0, 1, 2, 3, 4, 5])
    assert page["total"] == 6 and all(i["lot_id"] == lot_id and i["defect"] == "sang" for i in page["items"])
    assert all(i["prelabel"] == {"verdict": "non_viable", "defect": "sang"} for i in page["items"])

    # filtres combinés + pagination par curseur
    ids = _all_ids(client, page_size=7, lot_type="declasse", reviewed="false")
    assert sorted(ids) == sorted(_ids_where(project, m.Lot.type.like("declasse:%"), m.Label.reviewed.is_(False)))

    for bad in ({"verdict": "bof"}, {"defect": "rouille"}, {"quality": "parfaite"}, {"sort": "hasard"}, {"source": "x"}):
        assert client.get(f"{API}/tracks", params=bad).status_code == 400, bad
    # filtre de confiance sans modèle : aucun résultat ; tri par priorité impossible
    assert _tracks(client, score_min=0.5)["total"] == 0
    assert client.get(f"{API}/tracks", params={"sort": "priorite"}).status_code == 400


def test_sort_recent(client, project):
    ids = [i["id"] for i in _tracks(client, limit=3)["items"]]
    for tid in ids:
        assert client.put(f"{API}/tracks/{tid}/label", json={"verdict": "incertain"}).status_code == 200
    recent = _tracks(client, sort="recent", limit=3)
    assert [i["id"] for i in recent["items"]] == ids[::-1]
    all_recent = _all_ids(client, page_size=50, sort="recent")
    assert all_recent[:3] == ids[::-1] and len(all_recent) == 198 == len(set(all_recent))


def test_track_detail(client, project):
    tid = _tracks(client, verdict="aucun", limit=1)["items"][0]["id"]
    d = client.get(f"{API}/tracks/{tid}").json()
    assert d["id"] == tid and d["lot"]["type"] == "production" and d["prelabel"] == {"verdict": None, "defect": None}
    assert d["label"]["verdict"] is None and len(d["views"]) == 12
    assert d["views"][3] == {**d["views"][3], "index": 3, "defect_visible": False, "url": f"/api/media/crop/{tid}/3"}
    assert client.get(d["views"][0]["url"]).status_code == 200
    # vérité synthétique exposée à part, jamais comme label
    assert d["synthetic_truth"] is not None and "viable" in d["synthetic_truth"]
    assert d["prediction"] is None and d["history"][0]["action"] == "prelabel"
    assert client.get(f"{API}/tracks/999999").status_code == 404


# -- écritures ---------------------------------------------------------------
def test_label_rules(client, project):
    tid = _tracks(client, verdict="aucun", limit=1)["items"][0]["id"]
    put = lambda body: client.put(f"{API}/tracks/{tid}/label", json=body)  # noqa: E731

    r = put({"defect": "plume"}).json()["label"]  # défaut ⇒ non viable
    assert (r["verdict"], r["defect"], r["reviewed"], r["source"], r["reviewer"]) == ("non_viable", "plume", True, "humain", "démo")
    r = put({"verdict": "non_viable"}).json()["label"]  # le défaut est conservé
    assert r["defect"] == "plume"
    r = put({"verdict": "viable"}).json()["label"]  # viable ⇒ défaut vidé
    assert (r["verdict"], r["defect"]) == ("viable", None)
    r = put({"verdict": "non_viable"}).json()["label"]  # pas de défaut hérité d'un autre verdict
    assert (r["verdict"], r["defect"]) == ("non_viable", None)
    for verdict in ("incertain", "exclu"):
        put({"defect": "sang"})
        r = put({"verdict": verdict}).json()["label"]
        assert (r["verdict"], r["defect"]) == (verdict, None)
    r = put({"verdict": "non_viable", "defect": "sang", "note": "tache nette"}).json()["label"]
    assert (r["verdict"], r["defect"], r["note"]) == ("non_viable", "sang", "tache nette")
    r = put({"defect": None}).json()["label"]  # effacement du type seul
    assert (r["verdict"], r["defect"]) == ("non_viable", None)
    r = put({"verdict": "viable", "reviewed": False}).json()["label"]
    assert r["reviewed"] is False

    assert put({"verdict": "viable", "defect": "sang"}).status_code == 400
    assert put({"defect": "rouille"}).status_code == 400
    assert put({"verdict": "peut-etre"}).status_code == 400
    assert client.put(f"{API}/tracks/999999/label", json={"verdict": "viable"}).status_code == 404

    # le relecteur suit la configuration du projet
    client.put("/api/project/config", json={"user": {"name": "Alice"}})
    assert put({"verdict": "viable"}).json()["label"]["reviewer"] == "Alice"
    assert client.get(f"{API}/meta").json()["user"] == "Alice"


def test_confirm_reopen(client, project):
    page = _tracks(client, reviewed="false", verdict="viable", limit=1)
    tid = page["items"][0]["id"]
    r = client.post(f"{API}/tracks/{tid}/confirm").json()["label"]
    assert (r["verdict"], r["reviewed"], r["source"]) == ("viable", True, "humain")
    r = client.post(f"{API}/tracks/{tid}/reopen").json()["label"]
    assert (r["verdict"], r["reviewed"]) == ("viable", False)
    unlabeled = _tracks(client, verdict="aucun", limit=1)["items"][0]["id"]
    assert client.post(f"{API}/tracks/{unlabeled}/confirm").status_code == 400
    assert client.post(f"{API}/tracks/999999/confirm").status_code == 404


def test_view_marks(client, project):
    tid = _tracks(client, limit=1)["items"][0]["id"]
    url = f"{API}/tracks/{tid}/views/4/defect-visible"
    assert client.post(url).json()["defect_visible"] is True  # bascule
    assert client.post(url, json={"visible": True}).json()["defect_visible"] is True
    assert client.post(url).json()["defect_visible"] is False
    assert client.post(f"{API}/tracks/{tid}/views/99/defect-visible").status_code == 404
    assert client.post(url, json={"visible": True}).status_code == 200
    d = client.get(f"{API}/tracks/{tid}").json()
    assert [v["index"] for v in d["views"] if v["defect_visible"]] == [4]
    marks = [h for h in d["history"] if h["action"] == "marquage_vue"]
    assert len(marks) == 4 and marks[0]["after"] == {"view_index": 4, "defect_visible": True}
    # annulation du dernier marquage
    u = client.post(f"{API}/undo").json()
    assert u["undone"] == 1 and u["action"] == "marquage_vue"
    assert not any(v["defect_visible"] for v in client.get(f"{API}/tracks/{tid}").json()["views"])


def test_history_and_undo_stack(client, project):
    ids = [i["id"] for i in _tracks(client, verdict="aucun", limit=3)["items"]]
    a, b, c = ids
    client.put(f"{API}/tracks/{a}/label", json={"verdict": "viable"})
    client.put(f"{API}/tracks/{b}/label", json={"defect": "coulure"})
    client.put(f"{API}/tracks/{b}/label", json={"defect": "sang"})
    client.put(f"{API}/tracks/{c}/label", json={"verdict": "exclu"})

    hist = client.get(f"{API}/history", params={"user": "démo", "limit": 2, "with_total": True}).json()
    assert hist["total"] == 4 and [h["track_id"] for h in hist["items"]] == [c, b]
    assert hist["items"][1]["before"]["defect"] == "coulure" and hist["items"][1]["after"]["defect"] == "sang"
    more = client.get(f"{API}/history", params={"user": "démo", "limit": 2, "before_id": hist["next_before_id"]}).json()
    assert [h["track_id"] for h in more["items"]] == [b, a] and more["next_before_id"] is None
    per_track = client.get(f"{API}/history", params={"track_id": b}).json()["items"]
    assert [h["action"] for h in per_track] == ["label", "label", "prelabel"]
    assert {"démo", "auto"} <= set(client.get(f"{API}/history/users").json())
    assert client.get(f"{API}/history", params={"action": "prelabel", "limit": 500}).json()["items"]

    def label(tid):
        return client.get(f"{API}/tracks/{tid}").json()["label"]

    # annulations successives (pile)
    assert client.post(f"{API}/undo").json()["track_ids"] == [c]
    assert label(c)["verdict"] is None and label(c)["reviewed"] is False and label(c)["source"] == "auto"
    assert client.post(f"{API}/undo").json()["track_ids"] == [b]
    assert (label(b)["verdict"], label(b)["defect"]) == ("non_viable", "coulure")
    assert client.post(f"{API}/undo").json()["track_ids"] == [b]
    assert label(b)["verdict"] is None
    # conflit : la piste a été modifiée par quelqu'un d'autre entre-temps
    client.put("/api/project/config", json={"user": {"name": "Bob"}})
    client.put(f"{API}/tracks/{a}/label", json={"verdict": "incertain"})
    client.put("/api/project/config", json={"user": {"name": "démo"}})
    u = client.post(f"{API}/undo").json()
    assert u["undone"] == 0 and u["conflicts"] == 1 and label(a)["verdict"] == "incertain"
    assert client.post(f"{API}/undo").json()["message"] == "Rien à annuler."
    # l'annulation est elle-même historisée
    assert client.get(f"{API}/history", params={"action": "annulation"}).json()["items"]


def test_bulk(client, project):
    page = _tracks(client, lot_type="viable", reviewed="false", limit=20)
    ids = [i["id"] for i in page["items"]]
    r = client.post(f"{API}/bulk", json={"track_ids": ids, "verdict": "viable"})
    assert r.status_code == 200 and r.json()["updated"] == 20
    assert _tracks(client, lot_type="viable", reviewed="false")["total"] == 4
    hist = client.get(f"{API}/history", params={"action": "label_masse", "limit": 100}).json()["items"]
    assert len(hist) == 20 and len({h["after"]["batch"] for h in hist}) == 1

    # annulation d'un bloc
    u = client.post(f"{API}/undo").json()
    assert u["undone"] == 20 and sorted(u["track_ids"]) == sorted(ids)
    assert _tracks(client, lot_type="viable", reviewed="false")["total"] == 24

    # confirmation du pré-label : les pistes sans label sont ignorées
    prod = [i["id"] for i in _tracks(client, verdict="aucun", limit=3)["items"]]
    r = client.post(f"{API}/bulk", json={"track_ids": ids[:5] + prod, "confirm": True}).json()
    assert r["updated"] == 5 and sorted(r["skipped"]) == sorted(prod)
    # exclusion rapide d'intrus
    r = client.post(f"{API}/bulk", json={"track_ids": ids[5:7], "verdict": "exclu"}).json()
    assert r["updated"] == 2 and _tracks(client, verdict="exclu")["total"] == 2
    # défaut en masse ⇒ non viable
    client.post(f"{API}/bulk", json={"track_ids": prod, "defect": "deformation"})
    assert _tracks(client, defect="deformation", lot_type="production")["total"] == 3

    assert client.post(f"{API}/bulk", json={"track_ids": [ids[0], 999999], "verdict": "viable"}).status_code == 404
    assert client.post(f"{API}/bulk", json={"track_ids": ids, "verdict": "viable", "defect": "sang"}).status_code == 400
    assert client.post(f"{API}/bulk", json={"track_ids": []}).status_code == 422


# -- tableau de bord ---------------------------------------------------------
def test_dashboard(client, project):
    d = client.get(f"{API}/dashboard").json()
    t = d["totals"]
    assert (t["tracks"], t["labeled"], t["reviewed"], t["unreviewed"]) == (198, 180, 120, 78)
    assert t["progress"] == round(120 / 198, 4)
    verdicts = {v["verdict"]: v for v in d["verdicts"]}
    assert verdicts["viable"]["total"] == 72 and verdicts["viable"]["reviewed"] == 48
    assert verdicts["non_viable"]["total"] == 108 and verdicts[None]["total"] == 18
    defects = {x["defect"]: x for x in d["defects"]}
    assert all(defects[k]["total"] == 18 and defects[k]["reviewed"] == 12 for k in ("souillure", "sang", "deformation"))
    assert len(d["lots"]) == 22 == d["lots_total"]
    assert sum(lo["total"] for lo in d["lots"]) == 198
    assert len(d["sessions"]) == 3 and sum(s["total"] for s in d["sessions"]) == 198
    codes = {a["code"] for a in d["alerts"]}
    assert "classe_sous_representee" in codes and "defaut_sans_type" not in codes
    under = {a["class"] for a in d["alerts"] if a["code"] == "classe_sous_representee"}
    assert under == {f"defaut:{k}" for k in ("souillure", "coulure", "felure", "plume", "sang", "deformation")}
    unreviewed = next(a for a in d["alerts"] if a["code"] == "lots_non_relus")
    assert unreviewed["count"] == 8
    assert d["active"] is None

    # non viable sans type ⇒ alerte ; seuil d'alerte paramétrable
    tid = _tracks(client, verdict="aucun", limit=1)["items"][0]["id"]
    client.put(f"{API}/tracks/{tid}/label", json={"verdict": "non_viable"})
    d = client.get(f"{API}/dashboard", params={"min_examples": 10}).json()
    assert next(a for a in d["alerts"] if a["code"] == "defaut_sans_type")["count"] == 1
    assert not [a for a in d["alerts"] if a["code"] == "classe_sous_representee"]
    assert next(x for x in d["defects"] if x["defect"] is None)["total"] == 1


# -- apprentissage actif -----------------------------------------------------
def _add_model(project, model_id: str, scores: dict[int, float], *, threshold: float = 0.5, status: str = "trained", created_at=None) -> None:
    with project.session() as s:
        s.add(m.Model(id=model_id, name=f"Modèle {model_id}", status=status, threshold=threshold, **({"created_at": created_at} if created_at else {})))
        s.flush()
        for tid, score in scores.items():
            s.add(m.Prediction(model_id=model_id, track_id=tid, subset="test", score=score, view_scores=[score] * 12))


def test_active_learning(client, project):
    empty = client.get(f"{API}/active").json()
    assert empty["items"] == [] and empty["model"] is None and "Aucun modèle" in empty["message"]

    with project.session() as s:
        def pick(*conds, n=2):
            return list(s.scalars(select(m.Track.id).join(m.Label).join(m.Lot).where(*conds).order_by(m.Track.id).limit(n)))

        viable_rev = pick(m.Label.verdict == "viable", m.Label.reviewed.is_(True))
        viable_unrev = pick(m.Label.verdict == "viable", m.Label.reviewed.is_(False), n=4)
        nv_unrev = pick(m.Label.verdict == "non_viable", m.Label.reviewed.is_(False))
        unlabeled = pick(m.Label.verdict.is_(None), n=3)
    scores = {
        viable_rev[0]: 0.97,      # contredit (relu) : priorité ≈ 1.94
        viable_rev[1]: 0.10,      # d'accord, confiant : hors file
        viable_unrev[0]: 0.70,    # contredit (non relu) : 2 + 1.4
        viable_unrev[1]: 0.99,    # contredit, écart maximal : 2 + 1.98
        viable_unrev[2]: 0.45,    # hésite : 2 + 0.667
        viable_unrev[3]: 0.20,    # d'accord : hors file
        nv_unrev[0]: 0.02,        # contredit le label non viable : 2 + 1.96
        nv_unrev[1]: 0.90,        # d'accord
        unlabeled[0]: 0.51,       # hésite (sans label)
        unlabeled[1]: 0.80,       # sans label, confiant : hors file
        unlabeled[2]: 0.50,       # hésite au maximum
    }
    _add_model(project, "m-ancien", {k: 0.5 for k in scores}, created_at=m.utcnow().replace(year=2020))
    _add_model(project, "m-echec", {k: 0.5 for k in scores}, status="failed")
    _add_model(project, "m1", scores, threshold=0.5)

    models = client.get(f"{API}/models").json()
    assert models["default"]["id"] == "m1" and {x["id"] for x in models["items"]} == {"m1", "m-ancien", "m-echec"}

    q = client.get(f"{API}/active").json()
    assert q["model"]["id"] == "m1" and q["message"] is None
    got = [(i["id"], i["reason"]) for i in q["items"]]
    expected_order = [viable_unrev[1], nv_unrev[0], viable_unrev[0], unlabeled[2], unlabeled[0], viable_unrev[2], viable_rev[0]]
    assert [g[0] for g in got] == expected_order, got
    reasons = dict(got)
    assert reasons[viable_unrev[1]] == "contredit" and reasons[nv_unrev[0]] == "contredit" and reasons[viable_rev[0]] == "contredit"
    assert reasons[unlabeled[0]] == "hesite" and reasons[viable_unrev[2]] == "hesite"
    assert q["total"] == 7

    # priorité SQL = priorité Python
    model = ActiveModel("m1", "", 0.5, None)
    with project.session() as s:
        labels = {lab.track_id: lab for lab in s.scalars(select(m.Label).where(m.Label.track_id.in_(list(scores))))}
    for item in q["items"]:
        lab = labels[item["id"]]
        ev = evaluate(scores[item["id"]], lab.verdict, lab.reviewed, model)
        assert item["priority"] == pytest.approx(ev["priority"]) and item["reason"] == ev["reason"]

    # pagination de la file, raison, marge
    p1 = client.get(f"{API}/active", params={"limit": 3}).json()
    p2 = client.get(f"{API}/active", params={"limit": 3, "cursor": p1["next_cursor"]}).json()
    p3 = client.get(f"{API}/active", params={"limit": 3, "cursor": p2["next_cursor"]}).json()
    assert [i["id"] for i in p1["items"] + p2["items"] + p3["items"]] == expected_order and p3["next_cursor"] is None
    contra = client.get(f"{API}/active", params={"reason": "contredit"}).json()
    assert {i["id"] for i in contra["items"]} == {viable_unrev[1], nv_unrev[0], viable_unrev[0], viable_rev[0]}
    narrow = client.get(f"{API}/active", params={"margin": 0.02}).json()
    assert viable_unrev[2] not in {i["id"] for i in narrow["items"]}

    # tri « priorite » dans la recherche générale : toutes les pistes, celles sans prédiction à la fin
    # (les pistes non relues sur lesquelles le modèle est confiant passent avant les pistes relues)
    pr = _tracks(client, sort="priorite", limit=20)
    assert [i["id"] for i in pr["items"]][:6] == expected_order[:6] and pr["total"] == 198
    prios = [i["priority"] for i in pr["items"]]
    assert prios == sorted(prios, reverse=True) and prios[-1] == -1.0
    assert [i["id"] for i in pr["items"] if i["priority"] == 2.0] == sorted([viable_unrev[3], nv_unrev[1], unlabeled[1]])
    assert len(_all_ids(client, page_size=40, sort="priorite")) == 198
    # filtre de confiance (modèle par défaut ou explicite), score dans la liste
    hi = _tracks(client, score_min=0.9)
    assert {i["id"] for i in hi["items"]} == {viable_rev[0], viable_unrev[1], nv_unrev[1]}
    assert all(i["score"] >= 0.9 for i in hi["items"])
    assert _tracks(client, model_id="m-ancien", score_min=0.4, score_max=0.6)["total"] == 11
    assert _tracks(client, model_id="aucun", limit=1)["model"] is None
    assert client.get(f"{API}/tracks", params={"model_id": "inexistant"}).status_code == 404

    # le détail donne la prédiction et la raison ; relire fait baisser la priorité
    d = client.get(f"{API}/tracks/{viable_unrev[1]}").json()
    assert d["prediction"]["reason"] == "contredit" and d["prediction"]["predicted"] == "non_viable"
    client.put(f"{API}/tracks/{viable_unrev[1]}/label", json={"defect": "felure"})
    q2 = client.get(f"{API}/active").json()
    assert viable_unrev[1] not in {i["id"] for i in q2["items"]}  # désormais d'accord avec le modèle
    client.put(f"{API}/tracks/{nv_unrev[0]}/label", json={"verdict": "exclu"})
    assert nv_unrev[0] not in {i["id"] for i in client.get(f"{API}/active").json()["items"]}  # exclues jamais proposées

    dash = client.get(f"{API}/dashboard").json()
    assert dash["active"]["model"]["id"] == "m1" and dash["active"]["contradictions"] == 2
    assert any(a["code"] == "contradictions_modele" for a in dash["alerts"])


def test_evaluate_formula():
    model = ActiveModel("x", "", 0.3, None)
    assert evaluate(None, "viable", False, model)["reason"] is None
    ev = evaluate(0.65, "viable", True, model)  # contradiction : (0.65 - 0.3) / 0.7 = 0.5
    assert ev["reason"] == "contredit" and ev["priority"] == pytest.approx(1.5)
    ev = evaluate(0.15, "non_viable", False, model)  # (0.3 - 0.15) / 0.3 = 0.5, + 2 non relu
    assert ev["reason"] == "contredit" and ev["priority"] == pytest.approx(3.5)
    ev = evaluate(0.3, None, True, model)
    assert ev["reason"] == "hesite" and ev["priority"] == pytest.approx(1.0)
    assert evaluate(0.9, "non_viable", True, model)["reason"] is None
    assert evaluate(0.9, "exclu", False, model)["priority"] == -1.0
