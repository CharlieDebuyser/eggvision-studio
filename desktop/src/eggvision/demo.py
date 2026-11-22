"""Projet de démonstration synthétique (aucun matériel requis).

Fabrique un projet complet et cohérent : lots d'images rendus par
:mod:`eggvision_shared.synth`, pistes et vues recadrées, pré-labels. Sert de
mini jeu de données pour les tests automatisés et pour découvrir l'application.

Avec des images complètes, les pistes sont produites par le vrai module de détection
et de suivi (``use_detector``). Sans images (tests rapides), elles sont construites à
partir de la vérité terrain du générateur, conservée dans ``Track.extra["truth"]``.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from eggvision_shared.constants import DEFAULT_N_VIEWS, DEFAULT_VIEW_SIZE, DEFECT_TYPES
from eggvision_shared.formats import CameraSettings, LotManifest, LotWriter, write_image
from eggvision_shared.preprocess import extract_crop, select_view_indices, sharpness
from eggvision_shared.synth import SceneConfig, SyntheticScene

from .db import models as m
from .prelabel import apply_prelabels
from .project import Project

# Trois « sessions » (élevage, journée) : le découpage entraînement / validation / test se fait par lot.
SESSIONS = (
    {"farm": "Élevage des Chênes", "day_offset": -3},
    {"farm": "Élevage des Chênes", "day_offset": -2},
    {"farm": "Ferme du Val", "day_offset": -1},
)

DEMO_SIZES: dict[str, dict[str, Any]] = {
    # ≈ 360 œufs, ≈ 1 minute
    "mini": {"viable_columns": 8, "defect_columns": 2, "production_columns": 0, "scale": 0.5},
    # ≈ 1 500 œufs, quelques minutes
    "standard": {"viable_columns": 34, "defect_columns": 7, "production_columns": 20, "scale": 0.5},
}


def _noop(*_: Any, **__: Any) -> None:
    return None


def add_synthetic_lot(
    project: Project,
    name: str,
    lot_type: str,
    n_columns: int,
    *,
    farm: str = "",
    operator: str = "démo",
    started_at: datetime | None = None,
    scale: float = 0.5,
    seed: int = 0,
    defect_rate: float = 0.05,
    with_frames: bool = True,
    with_tracks: bool = True,
    use_detector: bool = False,
    n_views: int = DEFAULT_N_VIEWS,
    view_size: int = DEFAULT_VIEW_SIZE,
    reviewed: bool = False,
    lot_id: str | None = None,
    progress: Callable[[float], None] = _noop,
    check_cancelled: Callable[[], None] = _noop,
) -> str:
    """Ajoute un lot synthétique au projet et retourne son identifiant.

    * ``with_frames=True`` : rend et enregistre toutes les images du lot (réaliste,
      permet de tester la détection) ; les vues sont recadrées dans ces images.
    * ``with_frames=False`` : ne rend que les vues recadrées (très rapide, pour les tests).
    * ``with_tracks=False`` : n'enregistre que les images (lot « brut », avant détection).
    * ``use_detector=True`` (avec images) : les pistes sont produites par le vrai module de
      détection et de suivi au lieu de la vérité terrain du générateur.
    """
    cfg = SceneConfig.scaled(scale, lot_type=lot_type, seed=seed, defect_rate=defect_rate, n_columns=n_columns)
    scene = SyntheticScene(cfg)
    started_at = started_at or datetime.now(UTC)
    lot_id = lot_id or (started_at.strftime("%Y%m%d-%H%M%S") + f"-synth-{seed:04d}")
    lot_dir = project.lot_dir(lot_id)
    n_frames = scene.frames_for_columns(n_columns)
    camera = CameraSettings(fps=cfg.fps)
    manifest = LotManifest(
        lot_id=lot_id,
        name=name,
        type=lot_type,
        farm=farm,
        operator=operator,
        comment="Lot synthétique de démonstration",
        created_at=started_at.isoformat(timespec="milliseconds"),
        started_at=started_at.isoformat(timespec="milliseconds"),
        source={"kind": "simulated", "seed": seed, "scale": scale},
        camera=camera,
        width=cfg.width,
        height=cfg.height,
    )

    # Passe 1 (sans rendu) : quelles images voient chaque œuf en entier ?
    seen: dict[int, list[tuple[int, Any]]] = {}
    for i in range(n_frames):
        for tr in scene.truth_at(scene.frame_time(i)):
            if tr.fully_visible and tr.column < n_columns:
                seen.setdefault(tr.egg_id, []).append((i, tr))
    picks: dict[int, list[tuple[int, int, Any]]] = {}  # image → [(œuf, rang de vue, vérité)]
    for egg_id, items in seen.items():
        for view_idx, k in enumerate(select_view_indices(len(items), n_views)):
            frame_idx, tr = items[k]
            picks.setdefault(frame_idx, []).append((egg_id, view_idx, tr))

    crops: dict[int, dict[int, tuple[np.ndarray, int]]] = {}
    base_ns = int(started_at.timestamp() * 1e9)
    if with_frames:
        writer = LotWriter(lot_dir, manifest)
        for i in range(n_frames):
            if i % 10 == 0:
                check_cancelled()
                progress(0.9 * i / n_frames)
            img, _ = scene.frame(i)
            ts_ns = base_ns + int(round(i * 1e9 / cfg.fps))
            writer.write_frame(img, ts_ns, seq=i, system_ts=ts_ns / 1e9)
            if with_tracks and not use_detector:
                for egg_id, view_idx, tr in picks.get(i, ()):
                    crop = extract_crop(img, tr.cx, tr.cy, tr.major_px, tr.minor_px, out_size=view_size)
                    crops.setdefault(egg_id, {})[view_idx] = (crop, i)
        writer.close()
    else:
        lot_dir.mkdir(parents=True, exist_ok=True)
        manifest.status = "termine"
        manifest.ended_at = (started_at + timedelta(seconds=n_frames / cfg.fps)).isoformat(timespec="milliseconds")
        manifest.stats.duration_s = round(n_frames / cfg.fps, 3)
        manifest.stats.measured_fps = cfg.fps
        manifest.save(lot_dir)
        (lot_dir / "metadata.jsonl").touch()
        if with_tracks:
            eggs = [(egg_id, items) for egg_id, items in sorted(seen.items())]
            for n, (egg_id, items) in enumerate(eggs):
                if n % 10 == 0:
                    check_cancelled()
                    progress(0.9 * n / max(1, len(eggs)))
                tr0 = items[0][1]
                egg = scene.column(tr0.column)[tr0.lane]
                idxs = select_view_indices(len(items), n_views)
                for view_idx, view in enumerate(scene.render_views(egg, len(idxs), view_size)):
                    crops.setdefault(egg_id, {})[view_idx] = (view, items[idxs[view_idx]][0])

    project.register_lot(lot_id, source_kind="simulated")
    if not with_tracks:
        progress(1.0)
        return lot_id
    if use_detector and with_frames:
        from .vision.pipeline import process_lot

        process_lot(project, lot_id, progress=lambda f, _m=None: progress(0.9 + 0.1 * f), check_cancelled=check_cancelled)
        if reviewed:
            _mark_reviewed(project, lot_id)
        progress(1.0)
        return lot_id

    with project.session() as s:
        lot = s.get(m.Lot, lot_id)
        assert lot is not None
        for egg_id in sorted(crops):
            items = seen[egg_id]
            tr0 = items[0][1]
            views = crops[egg_id]
            sharp = [sharpness(views[v][0]) for v in sorted(views)]
            track = m.Track(
                lot_id=lot_id,
                lane=tr0.lane,
                start_frame=items[0][0],
                end_frame=items[-1][0],
                start_ts=items[0][0] / cfg.fps,
                end_ts=items[-1][0] / cfg.fps,
                n_detections=len(items),
                n_views=len(views),
                quality="ok",
                flags=[],
                mean_sharpness=float(np.mean(sharp)),
                partial_ratio=0.0,
                speed_px_s=cfg.speed_px_s,
                mean_major_px=tr0.major_px,
                mean_minor_px=tr0.minor_px,
                extra={"truth": {"egg_id": egg_id, "column": tr0.column, "viable": tr0.viable, "defect": tr0.defect}, "from_truth": True},
            )
            s.add(track)
            s.flush()
            project.crop_dir(track.id).mkdir(parents=True, exist_ok=True)
            for v in sorted(views):
                crop, frame_idx = views[v]
                write_image(project.crop_path(track.id, v), crop, 92)
                s.add(m.View(track_id=track.id, index=v, frame_index=frame_idx, path=project.crop_relpath(track.id, v), sharpness=sharp[v] if v < len(sharp) else 0.0))
        s.flush()
        lot.track_count = len(crops)
        lot.detection_status = "done"
        lot.detection_params = {"method": "verite_terrain_synthetique"}
        apply_prelabels(s, lot_id)
    if reviewed:
        _mark_reviewed(project, lot_id)
    progress(1.0)
    return lot_id


def _mark_reviewed(project: Project, lot_id: str) -> None:
    """Marque relus (par « démo ») les labels du lot qui ont un verdict."""
    with project.session() as s:
        for label in s.query(m.Label).join(m.Track).filter(m.Track.lot_id == lot_id):
            if label.verdict is not None:
                label.reviewed = True
                label.source = "humain"
                label.reviewer = "démo"


def build_demo_project(
    root: str | Path,
    name: str = "Demo EggVision",
    *,
    viable_columns: int = 8,
    defect_columns: int = 2,
    production_columns: int = 0,
    scale: float = 0.5,
    with_frames: bool = True,
    use_detector: bool = True,
    seed: int = 2024,
    progress: Callable[[float, str | None], None] = _noop,
    check_cancelled: Callable[[], None] = _noop,
    log: Callable[[str], None] = _noop,
) -> Project:
    """Crée un projet de démonstration complet dans ``root`` (qui doit être vide ou absent).

    Par session (élevage × journée) : un lot ``viable`` et un lot ``declasse:<défaut>``
    par type de défaut ; la dernière session reçoit en plus un lot ``production``.
    Les labels des deux premières sessions sont marqués relus.
    """
    project = Project.create(root, name)
    scene = SyntheticScene(SceneConfig.scaled(scale))
    project.update_config({"scene": scene.calibration(), "user": {"name": "démo"}})

    plan: list[dict[str, Any]] = []
    today = datetime.now(UTC).replace(hour=8, minute=0, second=0, microsecond=0)
    for s_idx, session in enumerate(SESSIONS):
        day = today + timedelta(days=session["day_offset"])
        t = day
        plan.append({"name": f"Viables — {session['farm']} J{s_idx + 1}", "type": "viable", "cols": viable_columns, "farm": session["farm"], "at": t, "session": s_idx})
        for defect in DEFECT_TYPES:
            t = t + timedelta(minutes=12)
            plan.append({"name": f"Déclassés {defect} — {session['farm']} J{s_idx + 1}", "type": f"declasse:{defect}", "cols": defect_columns, "farm": session["farm"], "at": t, "session": s_idx})
        if production_columns and s_idx == len(SESSIONS) - 1:
            plan.append({"name": f"Production — {session['farm']}", "type": "production", "cols": production_columns, "farm": session["farm"], "at": t + timedelta(minutes=30), "session": s_idx})

    total = sum(p["cols"] for p in plan) or 1
    done = 0
    for i, p in enumerate(plan):
        check_cancelled()
        start, span = done / total, p["cols"] / total
        log(f"Lot {i + 1}/{len(plan)} : {p['name']} ({p['cols']} colonnes)")
        add_synthetic_lot(
            project,
            p["name"],
            p["type"],
            p["cols"],
            farm=p["farm"],
            started_at=p["at"],
            scale=scale,
            seed=seed + i,
            with_frames=with_frames,
            use_detector=use_detector,
            reviewed=p["session"] < len(SESSIONS) - 1,
            progress=lambda f, start=start, span=span, p=p: progress(start + span * f, f"Rendu du lot « {p['name']} »"),
            check_cancelled=check_cancelled,
        )
        done += p["cols"]
    progress(1.0, "Projet de démonstration prêt")
    return project
