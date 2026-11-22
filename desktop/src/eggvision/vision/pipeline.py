"""Traitement d'un lot : images → détections → pistes → qualité → vues recadrées → base.

Déroulement (:func:`process_lot`) :

1. contrôle : si des pistes du lot portent un label relu par un humain, la relance est
   refusée sauf ``force=True`` (les labels relus seraient perdus) ;
2. lecture des images dans l'ordre, **multi-fil** et à fenêtre bornée ; chaque fil
   lit l'image, détecte les œufs (:func:`detect_frame`) et recadre dès ce moment
   chaque détection complète avec :func:`eggvision_shared.preprocess.extract_crop`
   (vue encodée en JPEG, gardée en mémoire tant que la piste est active) ;
3. suivi séquentiel (:class:`.tracking.Tracker`) à partir des horodatages capteur ;
4. à la clôture d'une piste : qualité, choix des vues (``select_view_indices`` parmi
   les détections complètes et nettes, couvrant toute la traversée), écriture des
   fichiers ``crops/<track_id>/<vue>.jpg`` et des lignes ``Track`` / ``View`` /
   ``Detection`` **par paquets** (mémoire bornée quelle que soit la taille du lot) ;
5. fin : drapeau « floue » relatif au lot, suppression des résultats précédents,
   mise à jour du lot, pré-labels.

Les anciens résultats ne sont supprimés qu'à la fin : une annulation ou une erreur
laisse le lot dans son état antérieur (les résultats partiels sont effacés).
"""

from __future__ import annotations

import logging
import shutil
import statistics
import time
import uuid
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

import cv2
import numpy as np
from eggvision_shared.formats import read_image
from eggvision_shared.preprocess import extract_crop, select_view_indices
from sqlalchemy import delete, func, select

from ..db import models as m
from ..prelabel import apply_prelabels
from ..project import Project
from .detect import detect_frame
from .params import DetectionParams, TrackingParams, load_params
from .scene import SceneGeometry, default_scene, rescale_scene, validate_scene
from .tracking import Tracker, TrackState, analyse_track

log = logging.getLogger(__name__)

PIPELINE_VERSION = 1
BATCH_TRACKS = 100  # pistes écrites par transaction
BATCH_UNASSIGNED = 5000


class DetectionRefused(Exception):
    """Relance refusée : des labels relus seraient perdus (message en français)."""


def _noop(*_: Any, **__: Any) -> None:
    return None


def reviewed_track_count(project: Project, lot_id: str) -> int:
    """Nombre de pistes du lot dont le label a été saisi ou relu par un humain."""
    with project.session() as s:
        return int(
            s.scalar(
                select(func.count())
                .select_from(m.Label)
                .join(m.Track, m.Track.id == m.Label.track_id)
                .where(m.Track.lot_id == lot_id, (m.Label.reviewed.is_(True)) | (m.Label.source == "humain"))
            )
            or 0
        )


def lot_scene(project: Project, lot: m.Lot) -> dict[str, Any]:
    """Calibration du projet adaptée à la résolution du lot (couloirs réguliers par défaut)."""
    scene = project.get_config("scene")
    w, h = int(lot.width or 0), int(lot.height or 0)
    if not w or not h:
        img = read_image(project.frame_path(lot.id, 0))
        h, w = img.shape[:2]
    if not scene:
        return default_scene(w, h)
    return rescale_scene(validate_scene(scene), w, h)


def frame_times(rows: list[tuple[int, int]], fps: float) -> tuple[list[float], bool]:
    """Instants (s, origine = première image) à partir des horodatages capteur.

    Retourne ``(instants, horloge_capteur_utilisée)`` ; si les horodatages sont absents ou
    non strictement croissants, on retombe sur ``rang / cadence``.
    """
    ts = [r[1] for r in rows]
    ok = len(ts) > 0 and all(b > a for a, b in zip(ts, ts[1:])) and ts[0] > 0
    if ok:
        t0 = ts[0]
        return [(t - t0) / 1e9 for t in ts], True
    fps = fps if fps and fps > 0 else 30.0
    i0 = rows[0][0] if rows else 0
    return [(r[0] - i0) / fps for r in rows], False


class _Writer:
    """Écriture par paquets des pistes terminées (fichiers + base)."""

    def __init__(self, project: Project, lot_id: str, run_id: str, geo: SceneGeometry, tp: TrackingParams,
                 t0_ts: float) -> None:
        self.project = project
        self.lot_id = lot_id
        self.run_id = run_id
        self.geo = geo
        self.tp = tp
        self.pending: list[tuple[TrackState, dict[str, Any]]] = []
        self.unassigned: list[dict[str, Any]] = []
        self.track_ids: list[int] = []
        self.n_detections = 0
        self.t0_ts = t0_ts

    def add_track(self, t: TrackState, info: dict[str, Any]) -> None:
        self.pending.append((t, info))
        if len(self.pending) >= BATCH_TRACKS:
            self.flush()

    def add_unassigned(self, frame_index: int, det: dict[str, Any]) -> None:
        if not self.tp.store_unassigned:
            return
        self.unassigned.append(_det_row(self.lot_id, frame_index, None, det))
        if len(self.unassigned) >= BATCH_UNASSIGNED:
            self.flush()

    def flush(self) -> None:
        if not self.pending and not self.unassigned:
            return
        pending, self.pending = self.pending, []
        unassigned, self.unassigned = self.unassigned, []
        with self.project.session() as s:
            tracks = []
            for t, info in pending:
                views = info["views"]
                track = m.Track(
                    lot_id=self.lot_id,
                    lane=int(info["lane"]),
                    start_frame=t.obs[0].frame_index,
                    end_frame=t.obs[-1].frame_index,
                    start_ts=round(t.obs[0].ts, 6),
                    end_ts=round(t.obs[-1].ts, 6),
                    n_detections=len(t.obs),
                    n_views=len(views),
                    quality="ok" if not info["flags"] else "douteuse",
                    flags=list(info["flags"]),
                    mean_sharpness=round(float(np.mean([t.obs[i].det["sharpness"] for i in views])) if views else 0.0, 3),
                    partial_ratio=info["partial_ratio"],
                    speed_px_s=round(info["speed_px_s"], 3),
                    mean_major_px=round(info["mean_major"], 2),
                    mean_minor_px=round(info["mean_minor"], 2),
                    extra={
                        "run": self.run_id,
                        "entered": info["entered"],
                        "exited": info["exited"],
                        "lane_agreement": info["lane_agreement"],
                        "gap_ratio": info["gap_ratio"],
                        "size_cv": info["size_cv"],
                        "n_complete": len(info["complete"]),
                        "missed": t.total_missed,
                        "view_frames": [t.obs[i].frame_index for i in views],
                    },
                )
                tracks.append(track)
            s.add_all(tracks)
            s.flush()
            view_rows: list[dict[str, Any]] = []
            det_rows: list[dict[str, Any]] = list(unassigned)
            for track, (t, info) in zip(tracks, pending):
                self.track_ids.append(track.id)
                crop_dir = self.project.crop_dir(track.id)
                if crop_dir.exists():  # reste d'une piste supprimée portant le même identifiant
                    shutil.rmtree(crop_dir, ignore_errors=True)
                if info["views"]:
                    crop_dir.mkdir(parents=True, exist_ok=True)
                for v, i in enumerate(info["views"]):
                    o = t.obs[i]
                    data = o.payload
                    if data is None:  # ne devrait pas arriver : vue recadrée manquante
                        continue
                    with open(self.project.crop_path(track.id, v), "wb") as f:
                        f.write(data)
                    view_rows.append(
                        {
                            "track_id": track.id,
                            "index": v,
                            "frame_index": o.frame_index,
                            "path": self.project.crop_relpath(track.id, v),
                            "sharpness": float(o.det["sharpness"]),
                            "partial": bool(o.det["partial"]),
                            "defect_visible": False,
                        }
                    )
                for o in t.obs:
                    det_rows.append(_det_row(self.lot_id, o.frame_index, track.id, o.det))
                    o.payload = None  # libère la mémoire
            if view_rows:
                s.execute(m.View.__table__.insert(), view_rows)
            for i in range(0, len(det_rows), 5000):
                s.execute(m.Detection.__table__.insert(), det_rows[i : i + 5000])
            self.n_detections += len(det_rows)


def _det_row(lot_id: str, frame_index: int, track_id: int | None, d: dict[str, Any]) -> dict[str, Any]:
    return {
        "lot_id": lot_id,
        "frame_index": int(frame_index),
        "track_id": track_id,
        "lane": int(d.get("lane", -1)),
        "cx": float(d["cx"]),
        "cy": float(d["cy"]),
        "major": float(d["major"]),
        "minor": float(d["minor"]),
        "angle": float(d.get("angle", 0.0)),
        "partial": bool(d.get("partial", False)),
        "sharpness": float(d.get("sharpness", 0.0)),
        "score": float(d.get("score", 1.0)),
    }


def _select_views(t: TrackState, info: dict[str, Any], n_views: int) -> list[int]:
    cand = [i for i in info["candidates"] if t.obs[i].payload is not None]
    return [cand[k] for k in select_view_indices(len(cand), n_views)]


def process_lot(
    project: Project,
    lot_id: str,
    *,
    det_params: DetectionParams | None = None,
    trk_params: TrackingParams | None = None,
    scene: dict[str, Any] | None = None,
    force: bool = False,
    workers: int | None = None,
    progress: Callable[[float, str | None], None] = _noop,
    log_fn: Callable[[str], None] = _noop,
    check_cancelled: Callable[[], None] = _noop,
) -> dict[str, Any]:
    """Traite un lot complet et retourne un résumé (compteurs, vitesse, temps par image).

    Lève :class:`DetectionRefused` si des labels relus seraient perdus et que ``force`` est faux.
    """
    t_start = time.perf_counter()
    if det_params is None or trk_params is None:
        dp, tp = load_params(project)
        det_params = det_params or dp
        trk_params = trk_params or tp
    n_reviewed = reviewed_track_count(project, lot_id)
    if n_reviewed and not force:
        raise DetectionRefused(
            f"Le lot {lot_id} contient {n_reviewed} piste(s) dont le label a été relu : relancer la détection "
            "les supprimerait. Relancez avec « forcer » pour confirmer."
        )

    with project.session() as s:
        lot = s.get(m.Lot, lot_id)
        if lot is None:
            raise ValueError(f"Lot introuvable : {lot_id}")
        fps = float(lot.fps or 0.0)
        scene = rescale_scene(validate_scene(scene), lot.width, lot.height) if scene else lot_scene(project, lot)
        rows = [(int(r[0]), int(r[1] or 0)) for r in s.execute(
            select(m.Frame.index, m.Frame.sensor_ts_ns).where(m.Frame.lot_id == lot_id).order_by(m.Frame.index)
        )]
        old_track_ids = [int(x) for x in s.scalars(select(m.Track.id).where(m.Track.lot_id == lot_id))]
        max_old_det = int(s.scalar(select(func.coalesce(func.max(m.Detection.id), 0)).where(m.Detection.lot_id == lot_id)) or 0)
        previous_status = lot.detection_status if lot.detection_status != "running" else ("done" if old_track_ids else "none")
        lot.detection_status = "running"
    if not rows:
        with project.session() as s:
            lot = s.get(m.Lot, lot_id)
            lot.detection_status = previous_status
        raise ValueError(f"Le lot {lot_id} ne contient aucune image.")

    geo = SceneGeometry.from_scene(scene)
    times, sensor_clock = frame_times(rows, fps)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    log_fn(f"Lot {lot_id} : {len(rows)} images, {len(geo.lanes)} couloirs, sens {geo.direction}, "
           f"{geo.mm_per_px:.4f} mm/px, méthode {det_params.method}"
           + ("" if sensor_clock else " — horodatages capteur inutilisables : cadence nominale utilisée"))

    tracker = Tracker(geo, trk_params)
    writer = _Writer(project, lot_id, run_id, geo, trk_params, times[0])
    n_workers = workers or (2 if det_params.method == "nn" else 6)
    det_counts = {"frames": 0, "missing_files": 0, "detections": 0, "detect_s": 0.0, "read_s": 0.0}
    tp = trk_params

    def work(index: int) -> tuple[list[dict[str, Any]], list[bytes | None], float, float] | None:
        path = project.frame_path(lot_id, index)
        t0 = time.perf_counter()
        try:
            img = read_image(path)
        except (OSError, ValueError):
            return None
        t1 = time.perf_counter()
        dets = detect_frame(img, scene, det_params)
        t2 = time.perf_counter()
        crops: list[bytes | None] = []
        for d in dets:
            if d["lane"] >= 0 and not d["partial"]:
                view = extract_crop(img, d["cx"], d["cy"], d["major"], d["minor"], out_size=tp.view_size, margin=tp.crop_margin)
                ok, buf = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, int(tp.jpeg_quality)])
                crops.append(buf.tobytes() if ok else None)
            else:
                crops.append(None)
        return dets, crops, t1 - t0, t2 - t1

    def close_tracks(closed: list[TrackState]) -> None:
        for t in closed:
            info = analyse_track(t, geo, tp, tracker.speed)
            info["views"] = _select_views(t, info, tp.n_views)
            if len(info["views"]) < tp.min_views and "courte" not in info["flags"]:
                info["flags"].insert(0, "courte")
            writer.add_track(t, info)
            keep = set(info["views"])
            for i, o in enumerate(t.obs):  # les vues non retenues n'ont plus d'utilité : mémoire libérée
                if i not in keep:
                    o.payload = None

    try:
        n = len(rows)
        window: deque[tuple[int, float, Future]] = deque()
        it = iter(zip(rows, times))
        with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="detect") as pool:
            def fill() -> None:
                while len(window) < n_workers * 3:
                    try:
                        (index, _), ts = next(it)
                    except StopIteration:
                        return
                    window.append((index, ts, pool.submit(work, index)))

            fill()
            done = 0
            while window:
                index, ts, fut = window.popleft()
                res = fut.result()
                fill()
                done += 1
                if res is None:
                    det_counts["missing_files"] += 1
                else:
                    dets, crops, t_read, t_det = res
                    det_counts["frames"] += 1
                    det_counts["detections"] += len(dets)
                    det_counts["read_s"] += t_read
                    det_counts["detect_s"] += t_det
                    for d in dets:
                        if d["lane"] < 0:
                            writer.add_unassigned(index, d)
                    in_lane = [(d, c) for d, c in zip(dets, crops) if d["lane"] >= 0]
                    closed = tracker.update(index, ts, [d for d, _ in in_lane], [c for _, c in in_lane])
                    close_tracks(closed)
                if done % 20 == 0 or done == n:
                    check_cancelled()
                    progress(0.95 * done / n, f"Image {done}/{n} — {len(writer.track_ids) + len(writer.pending)} pistes")
            close_tracks(tracker.flush())
        writer.flush()
        check_cancelled()
        progress(0.96, "Contrôle de qualité")
        summary = _finish(project, lot_id, writer.track_ids, old_track_ids, max_old_det, tp)
    except BaseException:
        # Annulation ou erreur : on efface les résultats partiels, les anciens restent en place.
        _discard(project, writer.track_ids, lot_id, max_old_det)
        with project.session() as s:
            lot = s.get(m.Lot, lot_id)
            if lot is not None:
                lot.detection_status = previous_status
        raise
    elapsed = time.perf_counter() - t_start
    n_frames = max(1, det_counts["frames"])
    stats = {
        **summary,
        "frames": det_counts["frames"],
        "missing_files": det_counts["missing_files"],
        "detections": det_counts["detections"],
        "speed_mm_s": round(tracker.speed_mm_s, 2) if tracker.speed_mm_s is not None else None,
        "elapsed_s": round(elapsed, 2),
        "ms_per_frame": round(1000 * elapsed / n_frames, 2),
        "detect_ms_per_frame": round(1000 * det_counts["detect_s"] / n_frames, 2),
        "read_ms_per_frame": round(1000 * det_counts["read_s"] / n_frames, 2),
        "sensor_clock": sensor_clock,
    }
    with project.session() as s:
        lot = s.get(m.Lot, lot_id)
        lot.detection_status = "done"
        lot.track_count = summary["tracks"]
        lot.detection_params = {
            "version": PIPELINE_VERSION,
            "run": run_id,
            "method": det_params.method,
            "detection": det_params.model_dump(mode="json"),
            "tracking": trk_params.model_dump(mode="json"),
            "scene": scene,
            "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "stats": stats,
        }
        s.flush()
        apply_prelabels(s, lot_id)
    progress(1.0, f"{summary['tracks']} pistes, dont {summary['usable']} exploitables")
    log_fn(f"Lot {lot_id} terminé : {summary['tracks']} pistes ({summary['ok']} ok, {summary['usable']} exploitables), "
           f"{stats['ms_per_frame']} ms/image (détection {stats['detect_ms_per_frame']} ms), vitesse {stats['speed_mm_s']} mm/s")
    return stats


def _finish(project: Project, lot_id: str, new_ids: list[int], old_ids: list[int], max_old_det: int, tp: TrackingParams) -> dict[str, Any]:
    """Drapeau « floue » relatif au lot, suppression des anciens résultats, compteurs."""
    with project.session() as s:
        sharp = []
        for i in range(0, len(new_ids), 900):
            sharp.extend(
                v for v in s.scalars(select(m.Track.mean_sharpness).where(m.Track.id.in_(new_ids[i : i + 900]), m.Track.n_views > 0))
            )
        median = statistics.median(sharp) if sharp else 0.0
        limit = tp.blur_ratio * median
        if limit > 0:
            for i in range(0, len(new_ids), 900):
                rows = s.scalars(
                    select(m.Track).where(m.Track.id.in_(new_ids[i : i + 900]), m.Track.n_views > 0, m.Track.mean_sharpness < limit)
                ).all()
                for t in rows:
                    t.flags = [*t.flags, "floue"]
                    t.quality = "douteuse"
    _delete_tracks(project, old_ids)
    with project.session() as s:
        s.execute(delete(m.Detection).where(m.Detection.lot_id == lot_id, m.Detection.id <= max_old_det))
        counts = dict(s.execute(select(m.Track.quality, func.count()).where(m.Track.lot_id == lot_id).group_by(m.Track.quality)).all())
        usable = s.scalar(select(func.count()).select_from(m.Track).where(
            m.Track.lot_id == lot_id, m.Track.quality == "ok", m.Track.n_views >= tp.min_views))
    total = int(sum(counts.values()))
    return {"tracks": total, "ok": int(counts.get("ok", 0)), "doubtful": int(counts.get("douteuse", 0)), "usable": int(usable or 0),
            "replaced_tracks": len(old_ids), "blur_limit": round(limit, 3)}


def _delete_tracks(project: Project, ids: list[int]) -> None:
    for i in range(0, len(ids), 500):
        chunk = ids[i : i + 500]
        with project.session() as s:
            s.execute(delete(m.Detection).where(m.Detection.track_id.in_(chunk)))
            s.execute(delete(m.View).where(m.View.track_id.in_(chunk)))
            s.execute(delete(m.Track).where(m.Track.id.in_(chunk)))  # labels, historiques, prédictions : ON DELETE CASCADE
        for tid in chunk:
            shutil.rmtree(project.crop_dir(tid), ignore_errors=True)


def _discard(project: Project, new_ids: list[int], lot_id: str, max_old_det: int) -> None:
    try:
        _delete_tracks(project, new_ids)
        with project.session() as s:
            s.execute(delete(m.Detection).where(m.Detection.lot_id == lot_id, m.Detection.id > max_old_det))
    except Exception:  # le nettoyage ne doit pas masquer l'erreur d'origine
        log.exception("Nettoyage des résultats partiels du lot %s impossible", lot_id)


# ---------------------------------------------------------------------------
# Tâche de fond (plusieurs lots à la suite)
# ---------------------------------------------------------------------------
def check_lots(project: Project, lot_ids: list[str], force: bool) -> list[dict[str, Any]]:
    """Lots qui seraient refusés (labels relus) : ``[{lot_id, reviewed}]`` ; vide si ``force``."""
    if force:
        return []
    out = []
    for lot_id in lot_ids:
        n = reviewed_track_count(project, lot_id)
        if n:
            out.append({"lot_id": lot_id, "reviewed": n})
    return out


def submit_detection(project: Project, lot_ids: list[str], *, force: bool = False, overrides: dict[str, Any] | None = None) -> Any:
    """Lance le traitement de plusieurs lots à la suite dans une tâche de fond ``detection`` (groupe ``cpu``)."""
    from .. import state
    from ..jobs import JobContext

    det_params, trk_params = load_params(project, overrides)
    lot_ids = list(dict.fromkeys(lot_ids))
    with project.session() as s:
        for lot_id in lot_ids:
            if s.get(m.Lot, lot_id) is None:
                raise ValueError(f"Lot introuvable : {lot_id}")
    refused = check_lots(project, lot_ids, force)
    if refused:
        detail = ", ".join(f"{r['lot_id']} ({r['reviewed']} pistes relues)" for r in refused)
        raise DetectionRefused(f"Relance refusée : des labels relus seraient supprimés — {detail}. Confirmez avec « forcer ».")

    def run(ctx: JobContext) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for k, lot_id in enumerate(lot_ids):
            ctx.check_cancelled()
            sub = ctx.subprogress(k / len(lot_ids), (k + 1) / len(lot_ids))
            ctx.log(f"— Lot {k + 1}/{len(lot_ids)} : {lot_id}")
            try:
                stats = process_lot(
                    project, lot_id, det_params=det_params, trk_params=trk_params, force=force,
                    progress=lambda f, msg=None, k=k: sub(f, f"Lot {k + 1}/{len(lot_ids)} — {msg}" if msg else None),
                    log_fn=ctx.log, check_cancelled=ctx.check_cancelled,
                )
                results[lot_id] = {"status": "done", **stats}
            except DetectionRefused as exc:
                results[lot_id] = {"status": "refused", "error": str(exc)}
                ctx.log(str(exc))
            except Exception as exc:
                from ..jobs import JobCancelled

                if isinstance(exc, JobCancelled):
                    raise
                results[lot_id] = {"status": "error", "error": str(exc)}
                ctx.log(f"ERREUR sur le lot {lot_id} : {exc}")
                with project.session() as s:
                    lot = s.get(m.Lot, lot_id)
                    if lot is not None:
                        lot.detection_status = "error"
                        lot.detection_params = {**(lot.detection_params or {}), "error": str(exc)}
            ctx.publish(lots=results)
        failed = [k for k, v in results.items() if v["status"] != "done"]
        if failed and len(failed) == len(lot_ids):
            raise RuntimeError("Aucun lot n'a pu être traité : " + " ; ".join(results[k].get("error", "") for k in failed))
        return {"lots": results}

    title = f"Détection et suivi — {len(lot_ids)} lot(s)" if len(lot_ids) > 1 else f"Détection et suivi — {lot_ids[0]}"
    return state.jobs.submit("detection", title, run, params={"lot_ids": lot_ids, "force": force}, project=project, group="cpu")
