"""Opérations sur les simulations : création, lancement (tâche de fond), lecture, suppression,
duplication, vidéo, comparaison. Utilisé par l'API (:mod:`eggvision.api.simulation`)."""

from __future__ import annotations

import json
import secrets
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from .. import state
from ..db import models as m
from ..jobs import ACTIVE, JobCancelled, JobContext
from ..project import Project
from . import engine, results
from .composition import SimulationError, build_composition, load_test_pool, model_split
from .params import SimulationParams
from .video import VideoOptions, render_video

JOB_GROUP = "simulation"  # simulations et vidéos l'une après l'autre : latences mesurées non perturbées


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:  # SQLite ne conserve pas le fuseau : les dates sont enregistrées en UTC
        dt = dt.replace(tzinfo=UTC)
    return dt.isoformat(timespec="seconds")


def new_id() -> str:
    return f"sim-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"


# ---------------------------------------------------------------------------
# Modèles éligibles
# ---------------------------------------------------------------------------
def eligible_models(project: Project) -> list[dict[str, Any]]:
    """Modèles exportés (paquet présent) avec la taille de leur jeu test."""
    with project.session() as s:
        models = s.scalars(select(m.Model).order_by(m.Model.created_at.desc())).all()
        splits = {sp.id: sp for sp in s.scalars(select(m.DatasetSplit)).all()}
        for sp in splits.values():
            s.expunge(sp)
        rows = [(md.id, md.name, md.threshold, md.n_views, md.split_id, dict(md.export or {}), md.created_at, md.status) for md in models]
    out = []
    pools: dict[int, dict[str, Any]] = {}
    for mid, name, threshold, n_views, split_id, export, created_at, status in rows:
        pkg = engine.package_dir_of(project, export)
        if pkg is None:
            continue  # non exporté : pas éligible
        reason = None
        pool_summary = None
        if split_id is None or split_id not in splits:
            reason = "Pas de découpage entraînement / validation / test."
        else:
            if split_id not in pools:
                pools[split_id] = load_test_pool(project, splits[split_id]).summary()
            pool_summary = pools[split_id]
            if pool_summary["n_total"] == 0:
                reason = "Jeu test vide (aucun œuf viable ou non viable labellisé)."
        variants = export.get("variants")
        if not variants:
            try:
                cfg = json.loads((pkg / "inference_config.json").read_text(encoding="utf-8"))
                variants = sorted((cfg.get("model_files") or {}).keys())
            except Exception:
                variants = ["fp32"]
        out.append(
            {
                "id": mid,
                "name": name,
                "status": status,
                "created_at": _iso(created_at),
                "threshold": threshold,
                "n_views": n_views,
                "variants": variants,
                "split_id": split_id,
                "test_set": pool_summary,
                "eligible": reason is None,
                "reason": reason,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Création et exécution
# ---------------------------------------------------------------------------
def _model_name(project: Project, model_id: str | None) -> str | None:
    if not model_id:
        return None
    with project.session() as s:
        md = s.get(m.Model, model_id)
        return md.name if md else None


def _set(project: Project, sim_id: str, **fields: Any) -> None:
    with project.session() as s:
        row = s.get(m.Simulation, sim_id)
        if row is None:
            return
        for k, v in fields.items():
            setattr(row, k, v)


def _merge_results(project: Project, sim_id: str, **data: Any) -> None:
    with project.session() as s:
        row = s.get(m.Simulation, sim_id)
        if row is None:
            return
        merged = dict(row.results or {})
        merged.update(data)
        row.results = merged  # nouvelle valeur : la colonne JSON est bien marquée modifiée


def create(project: Project, params: SimulationParams, name: str | None = None) -> dict[str, Any]:
    """Valide, constitue les colonnes, enregistre la simulation et lance la tâche de fond.

    Les erreurs de données (modèle non exporté, pas de découpage, jeu test vide…)
    sont levées immédiatement (:class:`SimulationError`), avant toute tâche.
    """
    if params.seed is None:
        params = params.model_copy(update={"seed": secrets.randbelow(2**31 - 1)})
    model, split = model_split(project, params.model_id)
    pkg = engine.package_dir_of(project, model.export)
    if pkg is None:
        raise SimulationError(f"Le modèle « {model.name} » n'est pas exporté (paquet de déploiement introuvable).")
    classifier = engine.load_classifier(project, model.export, params)  # valide variante et seuil dès maintenant
    pool = load_test_pool(project, split)
    composition = build_composition(pool, params, classifier.n_views)

    sim_id = new_id()
    name = (name or "").strip() or f"{model.name} — {datetime.now():%d/%m %H:%M}"
    sim_dir = project.simulation_dir(sim_id)
    sim_dir.mkdir(parents=True, exist_ok=True)
    with project.session() as s:
        s.add(m.Simulation(id=sim_id, name=name[:200], model_id=model.id, status="created", params=params.model_dump(), results={}))

    def _job(ctx: JobContext) -> dict[str, Any]:
        return _run_job(ctx, project, sim_id, params, composition, classifier, {"id": model.id, "name": model.name})

    job = state.jobs.submit(
        "simulation",
        f"Simulation « {name} » ({params.n_columns} colonnes)",
        _job,
        params={"simulation_id": sim_id, "model_id": model.id, "n_columns": params.n_columns},
        project=project,
        group=JOB_GROUP,
    )
    _set(project, sim_id, job_id=job.id)
    return {"simulation": get(project, sim_id), "job": job.to_dict()}


def _run_job(
    ctx: JobContext,
    project: Project,
    sim_id: str,
    params: SimulationParams,
    composition: Any,
    classifier: Any,
    model_info: dict[str, Any],
) -> dict[str, Any]:
    sim_dir = project.simulation_dir(sim_id)
    _set(project, sim_id, status="running")
    ctx.publish(simulation_id=sim_id)
    try:
        timing = params.timing()
        ctx.log(
            f"Modèle {model_info['name']} ({getattr(classifier, 'variant', '?')}, seuil {classifier.threshold:.3f}) ; "
            f"{params.n_columns} colonnes, budget {timing['budget_ms']:.0f} ms, graine {params.seed}"
        )
        info = composition.info
        ctx.log(
            f"Composition : {info['n_eggs']} œufs ({info['n_non_viable']} non viables), "
            f"{info['n_distinct_eggs']} œufs distincts du jeu test (réutilisation {info['reuse_rate']:.1%})"
        )
        for w in timing["warnings"]:
            ctx.log("Attention : " + w)
        results.write_params(
            sim_dir,
            {"id": sim_id, "model": model_info, "params": params.model_dump(), "timing": timing, "composition": info},
        )

        def _progress(done: int, total: int, extra: dict[str, Any]) -> None:
            ctx.progress(0.95 * done / total, f"Colonne {done}/{total}")
            ctx.publish(simulation_id=sim_id, columns_done=done, columns_total=total, **extra)

        run = engine.run(project, params, composition, classifier, check_cancelled=ctx.check_cancelled, progress=_progress)
        ctx.progress(0.96, "Écriture des résultats")
        summary = results.aggregate(run, params, info)
        summary["model"] = model_info
        results.write_timeline(sim_dir, run, params)
        results.write_journal(sim_dir, run)
        summary["files"] = {
            "journal": project.relpath(sim_dir / results.JOURNAL_FILE),
            "timeline": project.relpath(sim_dir / results.TIMELINE_FILE),
            "params": project.relpath(sim_dir / results.PARAMS_FILE),
        }
        summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _set(project, sim_id, status="done", results=summary)
        c, r, lat = summary["counts"], summary["rates"], summary["latency"]
        ctx.log(
            f"Terminé en {run.wall_s:.1f} s : {c['ejected_ok']}/{c['non_viable']} non viables éjectés, "
            f"{c['missed_total']} manqués (dont {c['missed_late']} par retard), {c['false_reject']} viables éjectés à tort, "
            f"{c['late_decisions']} décisions en retard ; latence de calcul moyenne {lat['compute_ms']['mean']} ms "
            f"pour un budget effectif de {lat['effective_budget_ms']:.0f} ms"
        )
        return {"simulation_id": sim_id, "counts": c, "rates": r}
    except JobCancelled:
        _set(project, sim_id, status="cancelled")
        raise
    except Exception as exc:
        _merge_results(project, sim_id, error=str(exc) or exc.__class__.__name__)
        _set(project, sim_id, status="failed")
        raise


# ---------------------------------------------------------------------------
# Lecture
# ---------------------------------------------------------------------------
def _reconcile(project: Project, row: m.Simulation) -> str:
    """Statut réel : une simulation « en cours » dont la tâche a disparu ou s'est arrêtée est corrigée."""
    if row.status not in ("created", "running"):
        return row.status
    job = state.jobs.get(row.job_id) if row.job_id else None
    if job is not None and job.status in ACTIVE:
        return "running" if job.status == "running" else row.status
    if job is not None and job.status == "done":
        return row.status  # l'écriture finale est en cours
    new = "cancelled" if (job is not None and job.status == "cancelled") else "failed"
    row.status = new
    if job is None:
        row.results = {**(row.results or {}), "error": "Interrompue (fermeture de l'application)."}
    elif job.error:
        row.results = {**(row.results or {}), "error": job.error}
    return new


def _summary_row(row: m.Simulation, model_names: dict[str, str]) -> dict[str, Any]:
    res = row.results or {}
    p = row.params or {}
    rates = res.get("rates") or {}
    return {
        "id": row.id,
        "name": row.name,
        "model_id": row.model_id,
        "model_name": model_names.get(row.model_id or "", None),
        "status": row.status,
        "created_at": _iso(row.created_at),
        "job_id": row.job_id,
        "n_columns": p.get("n_columns"),
        "defect_rate": p.get("defect_rate"),
        "seed": p.get("seed"),
        "threshold": res.get("threshold", p.get("threshold")),
        "variant": res.get("variant", p.get("variant")),
        "recall": rates.get("recall"),
        "false_reject_rate": rates.get("false_reject_rate"),
        "late_rate": rates.get("late_rate"),
        "has_video": bool(res.get("video")),
        "error": res.get("error"),
    }


def list_simulations(project: Project, limit: int = 50, offset: int = 0, model_id: str | None = None) -> dict[str, Any]:
    with project.session() as s:
        q = select(m.Simulation)
        cq = select(func.count()).select_from(m.Simulation)
        if model_id:
            q = q.where(m.Simulation.model_id == model_id)
            cq = cq.where(m.Simulation.model_id == model_id)
        total = s.scalar(cq) or 0
        rows = s.scalars(q.order_by(m.Simulation.created_at.desc()).limit(limit).offset(offset)).all()
        names = dict(s.execute(select(m.Model.id, m.Model.name)).all())
        for row in rows:
            _reconcile(project, row)
        items = [_summary_row(r, names) for r in rows]
    return {"items": items, "total": total, "limit": limit, "offset": offset}


def _row(s: Any, sim_id: str) -> m.Simulation:
    row = s.get(m.Simulation, sim_id)
    if row is None:
        raise KeyError(sim_id)
    return row


def get(project: Project, sim_id: str) -> dict[str, Any]:
    """Détail : paramètres, grandeurs dérivées, résultats, tâche en cours."""
    with project.session() as s:
        row = _row(s, sim_id)
        _reconcile(project, row)
        names = dict(s.execute(select(m.Model.id, m.Model.name)).all())
        out = _summary_row(row, names)
        out["params"] = row.params or {}
        out["results"] = row.results or {}
    try:
        out["timing"] = SimulationParams(**out["params"]).timing()
    except Exception:
        out["timing"] = None
    job = state.jobs.get_dict(out["job_id"], project) if out["job_id"] else None
    out["job"] = job
    video_job_id = (out["results"].get("video_job_id") if out["results"] else None)
    out["video_job"] = state.jobs.get_dict(video_job_id, project) if video_job_id else None
    return out


def raw(project: Project, sim_id: str) -> dict[str, Any]:
    """Dictionnaire brut (paramètres + résultats) pour la comparaison."""
    with project.session() as s:
        row = _row(s, sim_id)
        _reconcile(project, row)
        names = dict(s.execute(select(m.Model.id, m.Model.name)).all())
        return {
            "id": row.id,
            "name": row.name,
            "model_id": row.model_id,
            "model_name": names.get(row.model_id or ""),
            "status": row.status,
            "created_at": _iso(row.created_at),
            "params": dict(row.params or {}),
            "results": dict(row.results or {}),
        }


def is_active(project: Project, sim_id: str) -> bool:
    with project.session() as s:
        row = _row(s, sim_id)
        job_ids = [row.job_id, (row.results or {}).get("video_job_id")]
    for jid in job_ids:
        job = state.jobs.get(jid) if jid else None
        if job is not None and job.status in ACTIVE:
            return True
    return False


def cancel(project: Project, sim_id: str) -> dict[str, Any] | None:
    with project.session() as s:
        row = _row(s, sim_id)
        job_id = row.job_id
    job = state.jobs.cancel(job_id) if job_id else None
    return job.to_dict() if job else None


def delete(project: Project, sim_id: str) -> None:
    if is_active(project, sim_id):
        raise SimulationError("Simulation ou vidéo en cours : l'annuler avant de la supprimer.")
    with project.session() as s:
        s.delete(_row(s, sim_id))
    sim_dir = project.simulation_dir(sim_id)
    results.timelines.forget(sim_dir / results.TIMELINE_FILE)
    if sim_dir.is_dir():
        shutil.rmtree(sim_dir, ignore_errors=True)


def duplicate(project: Project, sim_id: str, overrides: dict[str, Any], name: str | None = None) -> dict[str, Any]:
    """Relance une simulation avec les mêmes paramètres (même graine), modèle / seuil / variante modifiables.

    Avec le même découpage, la composition des colonnes est identique : la comparaison
    porte alors uniquement sur le modèle ou le seuil.
    """
    with project.session() as s:
        row = _row(s, sim_id)
        base = dict(row.params or {})
        base_name = row.name
    if "model_id" in overrides and overrides["model_id"] != base.get("model_id"):
        base["variant"] = None  # la variante d'origine peut ne pas exister dans l'autre paquet
        base["threshold"] = None
    base.update(overrides)
    params = SimulationParams(**base)
    if not name:
        parts = []
        if "model_id" in overrides:
            parts.append(_model_name(project, params.model_id) or params.model_id)
        if "threshold" in overrides:
            parts.append("seuil " + ("du modèle" if params.threshold is None else f"{params.threshold:.3f}".replace(".", ",")))
        if "variant" in overrides and params.variant:
            parts.append(params.variant)
        name = f"{base_name} — " + (", ".join(parts) if parts else "copie")
    return create(project, params, name=name)


# ---------------------------------------------------------------------------
# Chronologie, œuf, journal
# ---------------------------------------------------------------------------
def _require_done(project: Project, sim_id: str) -> Path:
    with project.session() as s:
        row = _row(s, sim_id)
        status = _reconcile(project, row)
    sim_dir = project.simulation_dir(sim_id)
    if status != "done" or not (sim_dir / results.TIMELINE_FILE).is_file():
        raise SimulationError("La simulation n'est pas terminée : résultats indisponibles.")
    return sim_dir


def timeline_slice(project: Project, sim_id: str, start: int = 0, count: int = 200) -> dict[str, Any]:
    sim_dir = _require_done(project, sim_id)
    data = results.timelines.load(sim_dir / results.TIMELINE_FILE)
    cols = data["columns"]
    start = max(0, min(start, len(cols)))
    count = max(0, min(count, 2000))
    return {
        "geometry": data["geometry"],
        "threshold": data.get("threshold"),
        "total": len(cols),
        "start": start,
        "columns": [results.slim_column(c) for c in cols[start : start + count]],
    }


def egg_detail(project: Project, sim_id: str, column: int, lane: int) -> dict[str, Any]:
    sim_dir = _require_done(project, sim_id)
    data = results.timelines.load(sim_dir / results.TIMELINE_FILE)
    cols = data["columns"]
    if not (0 <= column < len(cols)):
        raise KeyError("colonne")
    col = cols[column]
    if not (0 <= lane < len(col["lanes"])) or col["lanes"][lane] is None:
        raise KeyError("œuf")
    egg = col["lanes"][lane]
    track_id = egg["track_id"]
    with project.session() as s:
        all_views = s.execute(select(m.View.index).where(m.View.track_id == track_id).order_by(m.View.index)).scalars().all()
        track = s.get(m.Track, track_id)
        lot = s.get(m.Lot, track.lot_id) if track else None
        label = s.get(m.Label, track_id)
        current = {"verdict": label.verdict, "defect": label.defect} if label else None
    presented = list(egg.get("views") or [])
    vs = egg.get("view_scores") or []
    score_by_view = {vi: (vs[i] if i < len(vs) else None) for i, vi in enumerate(presented)}
    views = [
        {
            "index": int(vi),
            "url": f"/api/media/crop/{track_id}/{int(vi)}",
            "presented": vi in score_by_view,
            "score": score_by_view.get(vi),
        }
        for vi in all_views
    ]
    return {
        "column": column,
        "lane": lane,
        "track_id": track_id,
        "lot_id": egg.get("lot_id"),
        "lot_name": lot.name if lot else None,
        "truth": egg["truth"],
        "defect": egg["defect"],
        "current_label": current,  # label actuel (il a pu être corrigé depuis la simulation)
        "score": egg["score"],
        "threshold": data.get("threshold"),
        "decision": egg["decision"],
        "late": egg["late"],
        "ejected": egg["ejected"],
        "outcome": egg["outcome"],
        "outcome_label": engine.OUTCOME_LABELS[egg["outcome"]],
        "n_views": egg["n_views"],
        "n_views_expected": egg.get("n_views_expected"),
        "views": views,
        "timing": {k: col[k] for k in ("t_enter", "t_acq", "t_decision", "t_deadline", "t_trap", "measured_ms", "injected_ms", "compute_ms", "wait_ms", "response_ms", "margin_ms")},
        "labeling_url": f"/labellisation/piste/{track_id}",
    }


def journal_path(project: Project, sim_id: str) -> Path:
    return _require_done(project, sim_id) / results.JOURNAL_FILE


# ---------------------------------------------------------------------------
# Vidéo
# ---------------------------------------------------------------------------
def start_video(project: Project, sim_id: str, opts: VideoOptions) -> dict[str, Any]:
    sim_dir = _require_done(project, sim_id)
    with project.session() as s:
        row = _row(s, sim_id)
        prev = (row.results or {}).get("video_job_id")
        name = row.name
    if prev:
        pj = state.jobs.get(prev)
        if pj is not None and pj.status in ACTIVE:
            raise SimulationError("Une vidéo est déjà en cours de rendu pour cette simulation.")
    timeline_path = sim_dir / results.TIMELINE_FILE

    def _job(ctx: JobContext) -> dict[str, Any]:
        timeline = results.timelines.load(timeline_path)
        ctx.log(f"Rendu vidéo : {opts.model_dump()}")
        info = render_video(
            timeline,
            sim_dir / results.VIDEO_FILE,
            opts,
            check_cancelled=ctx.check_cancelled,
            progress=lambda f, msg=None: ctx.progress(f, msg),
        )
        info["path"] = project.relpath(sim_dir / results.VIDEO_FILE)
        info["rendered_at"] = datetime.now().isoformat(timespec="seconds")
        info["options"] = opts.model_dump()
        _merge_results(project, sim_id, video=info)
        ctx.log(f"Vidéo : {info['n_frames']} images, {info['duration_s']} s, lecture ×{info['speed']}, {info['size_bytes'] / 1e6:.1f} Mo ({info['encoder']})")
        return {"simulation_id": sim_id, "video": info}

    job = state.jobs.submit(
        "video",
        f"Vidéo de la simulation « {name} »",
        _job,
        params={"simulation_id": sim_id, **opts.model_dump()},
        project=project,
        group=JOB_GROUP,
    )
    _merge_results(project, sim_id, video_job_id=job.id)
    return job.to_dict()


def video_path(project: Project, sim_id: str) -> Path | None:
    with project.session() as s:
        _row(s, sim_id)
    path = project.simulation_dir(sim_id) / results.VIDEO_FILE
    return path if path.is_file() else None


def test_set_summary(project: Project, model_id: str) -> dict[str, Any]:
    _, split = model_split(project, model_id)
    return load_test_pool(project, split).summary()


__all__ = [
    "SimulationError",
    "create",
    "get",
    "list_simulations",
    "delete",
    "duplicate",
    "cancel",
    "timeline_slice",
    "egg_detail",
    "journal_path",
    "start_video",
    "video_path",
    "eligible_models",
    "raw",
]
