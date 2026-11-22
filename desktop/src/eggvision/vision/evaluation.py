"""Évaluation du suivi contre la vérité terrain d'un lot synthétique.

Mesure le **critère d'acceptation 2** : sur un lot de test, au moins 95 % des œufs
visibles sont transformés en pistes exploitables (qualité ``ok``, au moins 8 vues).

Un œuf « traverse entièrement l'image » s'il est absent de la première et de la
dernière image du lot et visible entre les deux. Il est **réussi** s'il lui
correspond exactement une piste ``ok`` d'au moins ``min_views`` vues, dans le bon
couloir. Une piste ``ok`` qui ne correspond à aucun œuf (ou à un mélange de deux
œufs) est une piste **fantôme**.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any

import numpy as np
from eggvision_shared.formats import CameraSettings, LotManifest, LotWriter
from eggvision_shared.synth import SceneConfig, SyntheticScene
from sqlalchemy import select

from ..db import models as m
from ..project import Project


def evaluate_synthetic_lot(
    project: Project,
    lot_id: str,
    scene_config: SceneConfig,
    *,
    min_views: int = 8,
    direction: str = "down",
    frame_time: Callable[[int], float] | None = None,
    transform: Callable[[float, float], tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Compare les pistes d'un lot à la vérité terrain du générateur qui l'a produit.

    ``frame_time(seq)`` : instant de l'image de numéro de séquence capteur ``seq`` (par défaut ``seq / fps``,
    comme :func:`demo.add_synthetic_lot` et :func:`write_synthetic_lot`, qui conservent la séquence même
    quand des images manquent)."""
    scene = SyntheticScene(scene_config)
    ft = frame_time or (lambda i: i / scene_config.fps)
    how = ROTATIONS[direction]
    tf = transform or (lambda x, y: rotate_point(x, y, how, scene_config.width, scene_config.height))
    with project.session() as s:
        frame_rows = s.execute(select(m.Frame.index, m.Frame.seq).where(m.Frame.lot_id == lot_id).order_by(m.Frame.index)).all()
        frames = [int(r[0]) for r in frame_rows]
        seq_of = {int(r[0]): int(r[1]) for r in frame_rows}
        tracks = {
            t.id: t
            for t in s.scalars(select(m.Track).where(m.Track.lot_id == lot_id))
        }
        dets = s.execute(
            select(m.Detection.track_id, m.Detection.frame_index, m.Detection.cx, m.Detection.cy)
            .where(m.Detection.lot_id == lot_id, m.Detection.track_id.is_not(None))
        ).all()
    if not frames:
        raise ValueError("Lot sans image")

    truth_by_frame: dict[int, list[Any]] = {}
    visible: dict[int, list[int]] = defaultdict(list)
    eggs: dict[int, Any] = {}
    for i in frames:
        truths = scene.truth_at(ft(seq_of[i]))
        truth_by_frame[i] = truths
        for tr in truths:
            visible[tr.egg_id].append(i)
            eggs.setdefault(tr.egg_id, tr)
    first, last = frames[0], frames[-1]
    traversing = {e for e, fr in visible.items() if fr[0] != first and fr[-1] != last}

    # Rattachement de chaque détection à l'œuf réel le plus proche.
    votes: dict[int, Counter] = defaultdict(Counter)
    for track_id, frame_index, cx, cy in dets:
        best, best_d = None, None
        for tr in truth_by_frame.get(frame_index, ()):
            tx, ty = tf(tr.cx, tr.cy)
            d = ((tx - cx) ** 2 + (ty - cy) ** 2) ** 0.5
            if d < 0.5 * tr.minor_px and (best_d is None or d < best_d):
                best, best_d = tr.egg_id, d
        votes[track_id][best] += 1

    track_egg: dict[int, int | None] = {}
    purity: dict[int, float] = {}
    for track_id, c in votes.items():
        egg, n = c.most_common(1)[0]
        track_egg[track_id] = egg
        purity[track_id] = n / sum(c.values())

    ok_tracks_by_egg: dict[int, list[int]] = defaultdict(list)
    ghosts, impure = [], []
    for tid, t in tracks.items():
        if t.quality != "ok":
            continue
        egg = track_egg.get(tid)
        if egg is None:
            ghosts.append(tid)
            continue
        if purity.get(tid, 0) < 0.8:
            impure.append(tid)
        ok_tracks_by_egg[egg].append(tid)

    good, failures = 0, []
    for egg_id in sorted(traversing):
        tr = eggs[egg_id]
        cand = ok_tracks_by_egg.get(egg_id, [])
        reason = None
        if not cand:
            others = [tid for tid, e in track_egg.items() if e == egg_id]
            reason = "aucune piste ok" + (f" ({', '.join(','.join(tracks[t].flags) for t in others)})" if others else " (non suivi)")
        elif len(cand) > 1:
            reason = f"{len(cand)} pistes ok"
        else:
            t = tracks[cand[0]]
            if t.n_views < min_views:
                reason = f"{t.n_views} vues"
            elif t.lane != tr.lane:
                reason = f"couloir {t.lane} au lieu de {tr.lane}"
            elif purity.get(t.id, 0) < 0.8:
                reason = "piste mélangée"
        if reason is None:
            good += 1
        else:
            failures.append({"egg_id": egg_id, "lane": tr.lane, "column": tr.column, "reason": reason})
    n = len(traversing)
    return {
        "eggs_traversing": n,
        "eggs_ok": good,
        "rate": good / n if n else 0.0,
        "tracks": len(tracks),
        "tracks_ok": sum(1 for t in tracks.values() if t.quality == "ok"),
        "ghost_tracks": len(ghosts),
        "impure_tracks": len(impure),
        "duplicates": sum(1 for e in traversing if len(ok_tracks_by_egg.get(e, [])) > 1),
        "failures": failures[:50],
    }


# ---------------------------------------------------------------------------
# Lots synthétiques de test (images manquantes, sens de défilement, scène modifiée)
# ---------------------------------------------------------------------------
ROTATIONS = {
    # sens de défilement obtenu en tournant l'image du générateur (qui défile « vers le bas »)
    "down": None,
    "left": "cw",  # rotation de 90° dans le sens horaire : +y devient -x
    "right": "ccw",
    "up": "flip",  # retournement vertical
}


def rotate_image(img: np.ndarray, how: str | None) -> np.ndarray:
    import cv2

    if how is None:
        return img
    if how == "cw":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if how == "ccw":
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if how == "flip":
        return cv2.flip(img, 0)
    raise ValueError(how)


def rotate_point(x: float, y: float, how: str | None, width: int, height: int) -> tuple[float, float]:
    """Transforme un point de l'image du générateur (``width`` × ``height``) comme :func:`rotate_image`."""
    if how is None:
        return x, y
    if how == "cw":
        return height - 1 - y, x
    if how == "ccw":
        return y, width - 1 - x
    if how == "flip":
        return x, height - 1 - y
    raise ValueError(how)


def rotated_calibration(scene: SyntheticScene, direction: str) -> dict[str, Any]:
    """Calibration de la scène vue après rotation (sens de défilement ``direction``)."""
    how = ROTATIONS[direction]
    cal = scene.calibration()
    w, h = cal["image_width"], cal["image_height"]
    if how in ("cw", "ccw"):
        cal["image_width"], cal["image_height"] = h, w
    cal["direction"] = direction
    lanes = []
    for lane in cal["lanes"]:
        # les sommets sont des bords d'image (pas des centres de pixels) : transformation continue
        pts = []
        for x, y in lane["polygon"]:
            if how == "cw":
                pts.append([h - y, x])
            elif how == "ccw":
                pts.append([y, w - x])
            elif how == "flip":
                pts.append([x, h - y])
            else:
                pts.append([x, y])
        lanes.append({"index": lane["index"], "polygon": pts})
    cal["lanes"] = lanes
    return cal


def write_synthetic_lot(
    project: Project,
    lot_id: str,
    scene_config: SceneConfig,
    n_columns: int,
    *,
    drop_frames: Iterable[int] = (),
    direction: str = "down",
    lot_type: str | None = None,
    edit_scene: Callable[[SyntheticScene], None] | None = None,
    started_at: datetime | None = None,
) -> tuple[SyntheticScene, dict[str, Any]]:
    """Écrit un lot brut (images seules) rendu par le générateur, puis l'enregistre dans le projet.

    * ``drop_frames`` : rangs d'images non enregistrées (images perdues : trous dans la séquence,
      les horodatages capteur des images restantes sont conservés) ;
    * ``direction`` : sens de défilement simulé en tournant les images ;
    * ``edit_scene`` : modifie la scène avant le rendu (ex. couleur des œufs).

    Retourne ``(scène, calibration)`` ; la calibration n'est PAS écrite dans le projet.
    """
    scene = SyntheticScene(scene_config)
    if edit_scene is not None:
        edit_scene(scene)
    how = ROTATIONS[direction]
    cal = rotated_calibration(scene, direction)
    started_at = started_at or datetime.now(UTC)
    manifest = LotManifest(
        lot_id=lot_id, name=lot_id, type=lot_type or scene_config.lot_type, operator="tests",
        started_at=started_at.isoformat(timespec="milliseconds"), source={"kind": "simulated", "seed": scene_config.seed},
        camera=CameraSettings(fps=scene_config.fps), width=cal["image_width"], height=cal["image_height"],
    )
    writer = LotWriter(project.lot_dir(lot_id), manifest)
    drop = set(drop_frames)
    base_ns = int(started_at.timestamp() * 1e9)
    for i in range(scene.frames_for_columns(n_columns)):
        if i in drop:
            continue
        img, _ = scene.frame(i)
        ts_ns = base_ns + int(round(i * 1e9 / scene_config.fps))
        writer.write_frame(rotate_image(img, how), ts_ns, seq=i, system_ts=ts_ns / 1e9)
    writer.close()
    project.register_lot(lot_id, source_kind="simulated")
    return scene, cal
