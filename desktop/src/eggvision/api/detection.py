"""API du module 2 — détection et suivi des œufs (``/api/detection/...``).

Référence : ``docs/api/detection.md``.
"""

from __future__ import annotations

import base64
import time
from collections import Counter
from typing import Any, Literal

import cv2
from eggvision_shared.formats import read_image
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import String, and_, cast, func, or_, select, true

from .. import state
from ..db import models as m
from ..project import Project
from ..vision import nn
from ..vision.detect import detect_frame
from ..vision.params import (
    DetectionParams,
    ParamsError,
    TrackingParams,
    detection_params,
    load_params,
    params_schema,
    save_params,
)
from ..vision.pipeline import DetectionRefused, check_lots, lot_scene, submit_detection
from ..vision.scene import (
    DIRECTION_LABELS,
    SceneError,
    default_scene,
    lanes_from_positions,
    mm_per_px_from_ruler,
    regular_lanes,
    rescale_scene,
    validate_scene,
)
from ..vision.tracking import FLAG_LABELS

router = APIRouter(prefix="/api/detection", tags=["detection"])


def _bad(exc: Exception, status: int = 400) -> HTTPException:
    return HTTPException(status, str(exc))


def _lot(project: Project, lot_id: str) -> m.Lot:
    with project.session() as s:
        lot = s.get(m.Lot, lot_id)
        if lot is None:
            raise state.not_found("Lot")
        s.expunge(lot)
        return lot


def _frame_image(project: Project, lot_id: str, frame_index: int):
    path = project.frame_path(lot_id, frame_index)
    if not path.is_file():
        raise HTTPException(404, "Image introuvable.")
    return read_image(path)


# ---------------------------------------------------------------------------
# Calibration de la scène
# ---------------------------------------------------------------------------
@router.get("/scene")
def get_scene(lot_id: str | None = None, project: Project = Depends(state.get_project)) -> dict[str, Any]:
    """Calibration enregistrée (ou ``null``) ; avec ``lot_id``, aussi la version adaptée à la résolution du lot."""
    scene = project.get_config("scene")
    out: dict[str, Any] = {"scene": scene, "directions": DIRECTION_LABELS, "configured": bool(scene)}
    if lot_id:
        lot = _lot(project, lot_id)
        out["lot_scene"] = lot_scene(project, lot)
    return out


class ScenePayload(BaseModel):
    image_width: int
    image_height: int
    direction: Literal["up", "down", "left", "right"] = "down"
    mm_per_px: float
    lanes: list[dict[str, Any]] = Field(default_factory=list)
    reference: dict[str, Any] | None = None


@router.put("/scene")
def put_scene(body: ScenePayload, project: Project = Depends(state.get_project)) -> dict[str, Any]:
    """Valide et enregistre la calibration (section ``scene`` de ``config.yaml``)."""
    try:
        scene = validate_scene(body.model_dump())
    except SceneError as exc:
        raise _bad(exc) from exc
    project.set_config("scene", scene)
    return {"scene": scene}


class RulerPayload(BaseModel):
    p1: tuple[float, float]
    p2: tuple[float, float]
    length_mm: float


@router.post("/scene/ruler")
def ruler(body: RulerPayload) -> dict[str, Any]:
    """Échelle (mm/px) à partir de deux points cliqués sur une règle et de la distance réelle."""
    try:
        mm = mm_per_px_from_ruler(body.p1, body.p2, body.length_mm)
    except SceneError as exc:
        raise _bad(exc) from exc
    return {"mm_per_px": round(mm, 6), "distance_px": round(body.length_mm / mm, 2)}


class ProposeLanesPayload(BaseModel):
    lot_id: str
    frame_index: int | None = None
    mode: Literal["regular", "detections"] = "detections"
    n_lanes: int = Field(6, ge=1, le=32)
    direction: Literal["up", "down", "left", "right"] | None = None
    n_frames: int = Field(12, ge=1, le=200, description="Images échantillonnées dans le lot (mode « detections »)")
    mm_per_px: float | None = None


@router.post("/scene/propose-lanes")
def propose_lanes(body: ProposeLanesPayload, project: Project = Depends(state.get_project)) -> dict[str, Any]:
    """Propose des couloirs : découpage régulier ou déduit des œufs détectés dans des images du lot."""
    lot = _lot(project, body.lot_id)
    base = lot_scene(project, lot)
    direction = body.direction or base["direction"]
    w, h = base["image_width"], base["image_height"]
    if body.mode == "regular":
        return {"lanes": regular_lanes(w, h, body.n_lanes, direction), "n_points": 0, "mode": "regular"}
    scene = {**base, "direction": direction, "lanes": regular_lanes(w, h, body.n_lanes, direction)}
    if body.mm_per_px:
        scene["mm_per_px"] = body.mm_per_px
    with project.session() as s:
        indices = [int(i) for i in s.scalars(select(m.Frame.index).where(m.Frame.lot_id == lot.id).order_by(m.Frame.index))]
    if not indices:
        raise HTTPException(400, "Le lot ne contient aucune image.")
    if body.frame_index is not None:
        picks = [body.frame_index]
    else:
        n = min(body.n_frames, len(indices))
        picks = [indices[round(k * (len(indices) - 1) / max(1, n - 1))] for k in range(n)]
    params, _ = load_params(project)
    params = params.model_copy(update={"method": "classic"})
    points = []
    for fi in picks:
        dets = detect_frame(_frame_image(project, lot.id, fi), scene, params)
        points.extend((d["cx"], d["cy"]) for d in dets if not d["partial"])  # type: ignore[index]
    lanes = lanes_from_positions(points, w, h, direction, body.n_lanes)
    return {"lanes": lanes, "n_points": len(points), "mode": "detections", "frames": picks}


@router.get("/lots/{lot_id}/frames")
def lot_frames(lot_id: str, limit: int = Query(200, ge=1, le=5000), offset: int = Query(0, ge=0),
               project: Project = Depends(state.get_project)) -> dict[str, Any]:
    """Images d'un lot (rang, séquence capteur, instant relatif), paginées — choix de l'image de référence."""
    with project.session() as s:
        total = s.scalar(select(func.count()).select_from(m.Frame).where(m.Frame.lot_id == lot_id)) or 0
        first = s.scalar(select(m.Frame.sensor_ts_ns).where(m.Frame.lot_id == lot_id).order_by(m.Frame.index).limit(1)) or 0
        rows = s.execute(
            select(m.Frame.index, m.Frame.seq, m.Frame.sensor_ts_ns).where(m.Frame.lot_id == lot_id)
            .order_by(m.Frame.index).limit(limit).offset(offset)
        ).all()
    return {"total": total, "items": [{"index": r[0], "seq": r[1], "t": round(((r[2] or 0) - first) / 1e9, 4)} for r in rows]}


# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------
@router.get("/params")
def get_params(project: Project = Depends(state.get_project)) -> dict[str, Any]:
    try:
        det, trk = load_params(project)
    except ParamsError as exc:
        raise _bad(exc) from exc
    return {
        "detection": det.model_dump(mode="json"),
        "tracking": trk.model_dump(mode="json"),
        "defaults": {"detection": DetectionParams().model_dump(mode="json"), "tracking": TrackingParams().model_dump(mode="json")},
        "schema": params_schema(),
    }


@router.get("/params/schema")
def get_params_schema() -> dict[str, Any]:
    return params_schema()


class ParamsPayload(BaseModel):
    detection: dict[str, Any] | None = None
    tracking: dict[str, Any] | None = None


@router.put("/params")
def put_params(body: ParamsPayload, project: Project = Depends(state.get_project)) -> dict[str, Any]:
    """Enregistre les paramètres (fusion avec les valeurs existantes, puis validation)."""
    try:
        if body.detection is not None:
            save_params(project, "detection", body.detection)
        if body.tracking is not None:
            save_params(project, "tracking", body.tracking)
    except ParamsError as exc:
        raise _bad(exc) from exc
    return get_params(project)


class ResetPayload(BaseModel):
    section: Literal["detection", "tracking", "all"] = "all"


@router.post("/params/reset")
def reset_params(body: ResetPayload, project: Project = Depends(state.get_project)) -> dict[str, Any]:
    """Revient aux valeurs par défaut (le détecteur entraînable sélectionné est conservé)."""
    if body.section in ("detection", "all"):
        keep = (project.get_config("detection", {}) or {}).get("nn_model")
        project.set_config("detection", {**DetectionParams().model_dump(mode="json"), "nn_model": keep})
    if body.section in ("tracking", "all"):
        project.set_config("tracking", TrackingParams().model_dump(mode="json"))
    return get_params(project)


# ---------------------------------------------------------------------------
# Aperçu en direct
# ---------------------------------------------------------------------------
class PreviewPayload(BaseModel):
    lot_id: str
    frame_index: int = 0
    params: dict[str, Any] | None = Field(None, description="Paramètres de détection candidats (partiels acceptés)")
    scene: dict[str, Any] | None = Field(None, description="Calibration candidate ; défaut : celle du projet")
    mask: bool = False
    mask_width: int = Field(480, ge=64, le=2000)


@router.post("/preview")
def preview(body: PreviewPayload, project: Project = Depends(state.get_project)) -> dict[str, Any]:
    """Détecte les œufs d'une image d'un lot avec des paramètres candidats (aperçu en direct, sans rien enregistrer)."""
    lot = _lot(project, body.lot_id)
    img = _frame_image(project, lot.id, body.frame_index)
    h, w = img.shape[:2]
    try:
        current, _ = load_params(project)
        params = detection_params({**current.model_dump(), **(body.params or {})})
        scene = rescale_scene(validate_scene(body.scene), w, h) if body.scene else lot_scene(project, lot)
        t0 = time.perf_counter()
        dets, info = detect_frame(img, scene, params, return_info=True, want_mask=body.mask)  # type: ignore[misc]
        elapsed = (time.perf_counter() - t0) * 1000
    except (ParamsError, SceneError, nn.DetectorError) as exc:
        raise _bad(exc) from exc
    out: dict[str, Any] = {
        "lot_id": lot.id,
        "frame_index": body.frame_index,
        "width": w,
        "height": h,
        "detections": dets,
        "count": len(dets),
        "complete": sum(1 for d in dets if not d["partial"]),
        "elapsed_ms": round(elapsed, 2),
        "threshold": info.get("threshold"),
        "scene": scene,
    }
    if body.mask and "mask" in info:
        mask = info["mask"]
        mw = min(body.mask_width, mask.shape[1])
        mh = max(1, round(mask.shape[0] * mw / mask.shape[1]))
        small = cv2.resize(mask, (mw, mh), interpolation=cv2.INTER_NEAREST)
        ok, buf = cv2.imencode(".png", small)
        if ok:
            out["mask_png"] = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    return out


# ---------------------------------------------------------------------------
# Lots et traitement
# ---------------------------------------------------------------------------
@router.get("/lots")
def list_lots(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    status: str | None = Query(None, description="none | running | done | error"),
    q: str | None = None,
    project: Project = Depends(state.get_project),
) -> dict[str, Any]:
    """Lots avec statut de détection et compteurs (pistes, ok, exploitables), paginés, plus récents d'abord."""
    _, trk = load_params(project)
    cond = []
    if status:
        cond.append(m.Lot.detection_status == status)
    if q:
        like = f"%{q}%"
        cond.append(or_(m.Lot.name.like(like), m.Lot.id.like(like), m.Lot.farm.like(like)))
    where = and_(*cond) if cond else true()
    with project.session() as s:
        total = s.scalar(select(func.count()).select_from(m.Lot).where(where)) or 0
        lots = s.scalars(select(m.Lot).where(where).order_by(m.Lot.created_at.desc()).limit(limit).offset(offset)).all()
        ids = [lot.id for lot in lots]
        counts: dict[str, dict[str, int]] = {i: {"ok": 0, "douteuse": 0, "usable": 0, "reviewed": 0} for i in ids}
        if ids:
            for lot_id, quality, n in s.execute(
                select(m.Track.lot_id, m.Track.quality, func.count()).where(m.Track.lot_id.in_(ids)).group_by(m.Track.lot_id, m.Track.quality)
            ):
                counts[lot_id][quality] = int(n)
            for lot_id, n in s.execute(
                select(m.Track.lot_id, func.count()).where(m.Track.lot_id.in_(ids), m.Track.quality == "ok", m.Track.n_views >= trk.min_views)
                .group_by(m.Track.lot_id)
            ):
                counts[lot_id]["usable"] = int(n)
            for lot_id, n in s.execute(
                select(m.Track.lot_id, func.count()).join(m.Label, m.Label.track_id == m.Track.id)
                .where(m.Track.lot_id.in_(ids), or_(m.Label.reviewed.is_(True), m.Label.source == "humain")).group_by(m.Track.lot_id)
            ):
                counts[lot_id]["reviewed"] = int(n)
    active = _active_jobs(project)
    items = []
    for lot in lots:
        c = counts[lot.id]
        n_tracks = c["ok"] + c["douteuse"]
        stats = (lot.detection_params or {}).get("stats") or {}
        items.append({
            "id": lot.id, "name": lot.name, "type": lot.type, "farm": lot.farm, "session_date": lot.session_date,
            "status": lot.status, "frame_count": lot.frame_count, "width": lot.width, "height": lot.height,
            "detection_status": lot.detection_status, "track_count": n_tracks, "tracks_ok": c["ok"],
            "tracks_doubtful": c["douteuse"], "usable": c["usable"], "usable_rate": round(c["usable"] / n_tracks, 4) if n_tracks else None,
            "reviewed": c["reviewed"], "method": (lot.detection_params or {}).get("method"),
            "ms_per_frame": stats.get("ms_per_frame"), "speed_mm_s": stats.get("speed_mm_s"),
            "error": (lot.detection_params or {}).get("error") if lot.detection_status == "error" else None,
            "job_id": active.get(lot.id),
        })
    return {"total": total, "items": items, "min_views": trk.min_views}


def _active_jobs(project: Project) -> dict[str, str]:
    out: dict[str, str] = {}
    for job in state.jobs.active(kind="detection", project=project):
        for lot_id in job.params.get("lot_ids", []):
            out[lot_id] = job.id
    return out


class RunPayload(BaseModel):
    lot_ids: list[str] = Field(min_length=1)
    force: bool = False
    overrides: dict[str, Any] | None = Field(None, description="{detection: {...}, tracking: {...}} pour ce traitement seulement")


@router.post("/run/check")
def run_check(body: RunPayload, project: Project = Depends(state.get_project)) -> dict[str, Any]:
    """Lots dont la relance supprimerait des labels relus (à confirmer avec ``force``)."""
    refused = check_lots(project, body.lot_ids, False)
    return {"needs_force": bool(refused), "lots": refused}


@router.post("/run")
def run(body: RunPayload, project: Project = Depends(state.get_project)) -> dict[str, Any]:
    """Lance la détection et le suivi d'un ou plusieurs lots (tâche de fond ``detection``, relançable)."""
    busy = [lot_id for lot_id in body.lot_ids if lot_id in _active_jobs(project)]
    if busy:
        raise HTTPException(409, "Traitement déjà en cours pour : " + ", ".join(busy))
    try:
        job = submit_detection(project, body.lot_ids, force=body.force, overrides=body.overrides)
    except DetectionRefused as exc:
        raise HTTPException(409, str(exc)) from exc
    except (ParamsError, SceneError, ValueError) as exc:
        raise _bad(exc) from exc
    return job.to_dict()


class CancelPayload(BaseModel):
    job_id: str | None = None
    lot_id: str | None = None


@router.post("/cancel")
def cancel(body: CancelPayload, project: Project = Depends(state.get_project)) -> dict[str, Any]:
    """Annule un traitement (par identifiant de tâche ou par lot). Les résultats précédents du lot sont conservés."""
    job_id = body.job_id or (_active_jobs(project).get(body.lot_id) if body.lot_id else None)
    if not job_id:
        raise HTTPException(404, "Aucun traitement en cours.")
    job = state.jobs.cancel(job_id)
    if job is None:
        raise state.not_found("Tâche")
    return job.to_dict()


@router.get("/jobs")
def detection_jobs(limit: int = Query(20, ge=1, le=200), project: Project = Depends(state.get_project)) -> list[dict[str, Any]]:
    """Tâches de détection et d'entraînement du détecteur (plus récentes d'abord)."""
    jobs = state.jobs.list(project, kind="detection", limit=limit) + state.jobs.list(project, kind="detector_training", limit=limit)
    jobs.sort(key=lambda j: j["created_at"] or "", reverse=True)
    return jobs[:limit]


# ---------------------------------------------------------------------------
# Pistes
# ---------------------------------------------------------------------------
def _track_dict(t: m.Track) -> dict[str, Any]:
    return {
        "id": t.id, "lot_id": t.lot_id, "lane": t.lane, "start_frame": t.start_frame, "end_frame": t.end_frame,
        "start_ts": t.start_ts, "end_ts": t.end_ts, "n_detections": t.n_detections, "n_views": t.n_views,
        "quality": t.quality, "flags": t.flags or [], "mean_sharpness": t.mean_sharpness, "partial_ratio": t.partial_ratio,
        "speed_px_s": t.speed_px_s, "mean_major_px": t.mean_major_px, "mean_minor_px": t.mean_minor_px,
        "entered": (t.extra or {}).get("entered"), "exited": (t.extra or {}).get("exited"),
    }


@router.get("/lots/{lot_id}/tracks")
def list_tracks(
    lot_id: str,
    quality: Literal["ok", "douteuse"] | None = None,
    flag: str | None = Query(None, description="motif de signalement (ex. courte, floue)"),
    lane: int | None = None,
    order: Literal["start", "quality", "views", "sharpness"] = "start",
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    project: Project = Depends(state.get_project),
) -> dict[str, Any]:
    """Pistes d'un lot, paginées et filtrables (qualité, motif, couloir)."""
    cond = [m.Track.lot_id == lot_id]
    if quality:
        cond.append(m.Track.quality == quality)
    if lane is not None:
        cond.append(m.Track.lane == lane)
    if flag:
        cond.append(cast(m.Track.flags, String).like(f'%"{flag}"%'))
    order_by = {
        "start": (m.Track.start_frame, m.Track.lane),
        "quality": (m.Track.quality.desc(), m.Track.start_frame),
        "views": (m.Track.n_views, m.Track.start_frame),
        "sharpness": (m.Track.mean_sharpness, m.Track.start_frame),
    }[order]
    with project.session() as s:
        total = s.scalar(select(func.count()).select_from(m.Track).where(*cond)) or 0
        rows = s.scalars(select(m.Track).where(*cond).order_by(*order_by).limit(limit).offset(offset)).all()
        items = [_track_dict(t) for t in rows]
    return {"total": total, "items": items, "flag_labels": FLAG_LABELS}


@router.get("/tracks/{track_id}")
def track_detail(track_id: int, project: Project = Depends(state.get_project)) -> dict[str, Any]:
    """Détail d'une piste : vues (URL des vignettes), détections successives, label."""
    with project.session() as s:
        t = s.get(m.Track, track_id)
        if t is None:
            raise state.not_found("Piste")
        out = _track_dict(t)
        out["extra"] = t.extra or {}
        out["flag_labels"] = {f: FLAG_LABELS.get(f, f) for f in out["flags"]}
        out["views"] = [
            {"index": v.index, "frame_index": v.frame_index, "sharpness": v.sharpness, "partial": v.partial,
             "url": f"/api/media/crop/{t.id}/{v.index}"}
            for v in t.views
        ]
        dets = s.scalars(select(m.Detection).where(m.Detection.track_id == track_id).order_by(m.Detection.frame_index)).all()
        out["detections"] = [_det_dict(d) for d in dets]
        label = t.label
        out["label"] = None if label is None else {"verdict": label.verdict, "defect": label.defect, "source": label.source, "reviewed": label.reviewed}
    return out


def _det_dict(d: m.Detection) -> dict[str, Any]:
    return {"id": d.id, "frame_index": d.frame_index, "track_id": d.track_id, "lane": d.lane, "cx": d.cx, "cy": d.cy,
            "major": d.major, "minor": d.minor, "angle": d.angle, "partial": d.partial, "sharpness": d.sharpness, "score": d.score}


@router.get("/lots/{lot_id}/frames/{frame_index}/detections")
def frame_detections(lot_id: str, frame_index: int, project: Project = Depends(state.get_project)) -> dict[str, Any]:
    """Détections enregistrées d'une image (superposition sur l'image)."""
    with project.session() as s:
        rows = s.scalars(select(m.Detection).where(m.Detection.lot_id == lot_id, m.Detection.frame_index == frame_index)).all()
        return {"lot_id": lot_id, "frame_index": frame_index, "detections": [_det_dict(d) for d in rows]}


@router.get("/lots/{lot_id}/stats")
def lot_stats(lot_id: str, project: Project = Depends(state.get_project)) -> dict[str, Any]:
    """Statistiques de qualité d'un lot, dont la part de pistes exploitables (``ok`` et au moins ``min_views`` vues)."""
    lot = _lot(project, lot_id)
    _, trk = load_params(project)
    flags: Counter[str] = Counter()
    lanes: dict[int, dict[str, int]] = {}
    views_hist: Counter[int] = Counter()
    total = ok = usable = complete = usable_complete = 0
    with project.session() as s:
        last_id = 0
        while True:  # parcours par paquets (mémoire bornée)
            rows = s.execute(
                select(m.Track.id, m.Track.quality, m.Track.flags, m.Track.n_views, m.Track.lane)
                .where(m.Track.lot_id == lot_id, m.Track.id > last_id).order_by(m.Track.id).limit(5000)
            ).all()
            if not rows:
                break
            last_id = rows[-1][0]
            for _, quality, fl, n_views, lane in rows:
                fl = fl or []
                total += 1
                is_ok = quality == "ok"
                is_usable = is_ok and n_views >= trk.min_views
                ok += is_ok
                usable += is_usable
                flags.update(fl)
                views_hist[min(int(n_views), trk.n_views)] += 1
                lane_c = lanes.setdefault(int(lane), {"tracks": 0, "usable": 0})
                lane_c["tracks"] += 1
                lane_c["usable"] += is_usable
                if "incomplete" not in fl:
                    complete += 1
                    usable_complete += is_usable
    params = lot.detection_params or {}
    return {
        "lot_id": lot_id,
        "detection_status": lot.detection_status,
        "min_views": trk.min_views,
        "tracks": total,
        "ok": ok,
        "doubtful": total - ok,
        "usable": usable,
        "usable_rate": round(usable / total, 4) if total else None,
        "complete_tracks": complete,
        "usable_rate_complete": round(usable_complete / complete, 4) if complete else None,
        "flags": dict(flags.most_common()),
        "flag_labels": FLAG_LABELS,
        "lanes": {str(k): v for k, v in sorted(lanes.items())},
        "views_histogram": {str(k): v for k, v in sorted(views_hist.items())},
        "run": params.get("stats"),
        "method": params.get("method"),
        "finished_at": params.get("finished_at"),
    }


# ---------------------------------------------------------------------------
# Détecteur entraînable
# ---------------------------------------------------------------------------
@router.get("/detector")
def detector_info(project: Project = Depends(state.get_project)) -> dict[str, Any]:
    det, _ = load_params(project)
    ann = nn.list_annotations(project, limit=1)
    return {"detectors": nn.list_detectors(project), "active": det.nn_model, "method": det.method,
            "annotations": {"total": ann["total"], "n_boxes": ann["n_boxes"], "n_corrected": ann["n_corrected"]}}


@router.get("/detector/annotations")
def detector_annotations(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
                         project: Project = Depends(state.get_project)) -> dict[str, Any]:
    return nn.list_annotations(project, limit, offset)


class AddAnnotationsPayload(BaseModel):
    lot_id: str
    frame_indices: list[int] | None = None
    n: int = Field(10, ge=1, le=200)


@router.post("/detector/annotations")
def add_annotations(body: AddAnnotationsPayload, project: Project = Depends(state.get_project)) -> dict[str, Any]:
    """Ajoute des images d'annotation, boîtes générées automatiquement par la méthode classique."""
    lot = _lot(project, body.lot_id)
    try:
        det, _ = load_params(project)
        added = nn.add_annotation_images(project, lot.id, body.frame_indices, body.n, scene=lot_scene(project, lot), params=det)
    except nn.DetectorError as exc:
        raise _bad(exc) from exc
    return {"added": len(added), "items": [{k: v for k, v in a.items() if k != "boxes"} | {"n_boxes": len(a["boxes"])} for a in added]}


@router.get("/detector/annotations/{ann_id}")
def get_annotation(ann_id: str, project: Project = Depends(state.get_project)) -> dict[str, Any]:
    try:
        return nn.get_annotation(project, ann_id)
    except nn.DetectorError as exc:
        raise _bad(exc, 404) from exc


class BoxesPayload(BaseModel):
    boxes: list[dict[str, float]]


@router.put("/detector/annotations/{ann_id}")
def put_annotation(ann_id: str, body: BoxesPayload, project: Project = Depends(state.get_project)) -> dict[str, Any]:
    """Remplace les boîtes d'une image (correction manuelle : ajout, suppression, ajustement)."""
    try:
        return nn.set_annotation_boxes(project, ann_id, body.boxes)
    except nn.DetectorError as exc:
        raise _bad(exc, 404 if "introuvable" in str(exc) else 400) from exc


@router.delete("/detector/annotations/{ann_id}")
def delete_annotation(ann_id: str, project: Project = Depends(state.get_project)) -> dict[str, Any]:
    try:
        nn.delete_annotation(project, ann_id)
    except nn.DetectorError as exc:
        raise _bad(exc, 404) from exc
    return {"deleted": ann_id}


class TrainPayload(BaseModel):
    epochs: int = Field(60, ge=1, le=2000)
    input_width: int = Field(256, ge=64, le=1024)
    name: str = ""
    activate: bool = False


@router.post("/detector/train")
def train(body: TrainPayload, project: Project = Depends(state.get_project)) -> dict[str, Any]:
    """Lance l'entraînement du détecteur (tâche ``detector_training``, groupe ``gpu``)."""
    try:
        job = nn.submit_training(project, epochs=body.epochs, input_width=body.input_width, name=body.name, activate=body.activate)
    except nn.DetectorError as exc:
        raise _bad(exc) from exc
    return job.to_dict()


class ActivatePayload(BaseModel):
    detector_id: str | None = None
    use: bool = Field(True, description="Sélectionner aussi la méthode « nn » pour la détection")


@router.post("/detector/activate")
def activate(body: ActivatePayload, project: Project = Depends(state.get_project)) -> dict[str, Any]:
    """Sélectionne un détecteur entraînable (``detector_id`` vide + ``use=false`` : retour à la méthode classique)."""
    if body.detector_id and body.detector_id not in {d["id"] for d in nn.list_detectors(project)}:
        raise state.not_found("Détecteur")
    if body.use and not body.detector_id:
        raise HTTPException(400, "Choisissez un détecteur pour utiliser la méthode « nn ».")
    changes: dict[str, Any] = {"method": "nn" if body.use else "classic"}
    if body.detector_id:
        changes["nn_model"] = body.detector_id
    try:
        save_params(project, "detection", changes)
    except ParamsError as exc:
        raise _bad(exc) from exc
    return detector_info(project)


@router.delete("/detector/{detector_id}")
def delete_detector(detector_id: str, project: Project = Depends(state.get_project)) -> dict[str, Any]:
    det, _ = load_params(project)
    if det.nn_model == detector_id and det.method == "nn":
        raise HTTPException(409, "Ce détecteur est utilisé par la détection : revenez d'abord à la méthode classique.")
    try:
        nn.delete_detector(project, detector_id)
    except nn.DetectorError as exc:
        raise _bad(exc, 404) from exc
    if det.nn_model == detector_id:
        save_params(project, "detection", {"nn_model": None})
    return {"deleted": detector_id}


@router.get("/defaults/scene")
def scene_default(width: int = Query(..., gt=0), height: int = Query(..., gt=0), n_lanes: int = Query(6, ge=1, le=32),
                  direction: Literal["up", "down", "left", "right"] = "down") -> dict[str, Any]:
    """Calibration par défaut pour une taille d'image (couloirs réguliers, échelle du capteur de référence)."""
    return default_scene(width, height, n_lanes, direction)
