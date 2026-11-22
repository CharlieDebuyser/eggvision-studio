"""Un projet = un dossier autonome (copiable, zippable, partageable).

::

    MonProjet/
      project.db   config.yaml
      lots/<lot_id>/{lot.json, metadata.jsonl, frames/}
      crops/<track_id>/<vue>.jpg
      models/<model_id>/   simulations/<sim_id>/   logs/

Tous les chemins stockés en base sont relatifs au dossier du projet.
"""

from __future__ import annotations

import copy
import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from eggvision_shared.constants import FORMAT_VERSION
from eggvision_shared.formats import LotManifest, frame_filename, read_metadata
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session, sessionmaker

from . import settings
from .db import models as m

log = logging.getLogger(__name__)

DB_FILENAME = "project.db"
CONFIG_FILENAME = "config.yaml"


class ProjectError(Exception):
    """Erreur présentable à l'utilisateur (message en français)."""


def is_project_dir(path: str | Path) -> bool:
    p = Path(path)
    return (p / DB_FILENAME).is_file() and (p / CONFIG_FILENAME).is_file()


def deep_merge(base: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    """Fusion récursive : les dictionnaires sont fusionnés, le reste est remplacé."""
    out = dict(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


class Project:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        if not is_project_dir(self.root):
            raise ProjectError(f"Ce dossier n'est pas un projet EggVision : {self.root}")
        self._config_lock = threading.RLock()
        self.engine = create_engine(
            f"sqlite:///{self.db_path.as_posix()}",
            connect_args={"check_same_thread": False, "timeout": 30},
        )

        @event.listens_for(self.engine, "connect")
        def _pragmas(dbapi_conn: Any, _record: Any) -> None:
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA busy_timeout=30000")
            cur.close()

        m.Base.metadata.create_all(self.engine)
        self._sessionmaker = sessionmaker(self.engine, expire_on_commit=False)
        for sub in ("lots", "crops", "models", "simulations", "logs"):
            (self.root / sub).mkdir(exist_ok=True)

    # -- création ---------------------------------------------------------
    @classmethod
    def create(cls, root: str | Path, name: str | None = None) -> Project:
        root = Path(root).resolve()
        if root.exists() and any(root.iterdir()):
            raise ProjectError(f"Le dossier existe déjà et n'est pas vide : {root}")
        root.mkdir(parents=True, exist_ok=True)
        config = {
            "project": {
                "name": name or root.name,
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "format_version": FORMAT_VERSION,
            },
            "user": {"name": ""},
            "scene": None,
        }
        (root / CONFIG_FILENAME).write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        (root / DB_FILENAME).touch()
        project = cls(root)
        with project.session() as s:
            s.merge(m.Meta(key="schema_version", value=str(m.SCHEMA_VERSION)))
        return project

    def close(self) -> None:
        self.engine.dispose()

    # -- chemins ----------------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.root / DB_FILENAME

    @property
    def config_path(self) -> Path:
        return self.root / CONFIG_FILENAME

    @property
    def lots_dir(self) -> Path:
        return self.root / "lots"

    @property
    def crops_dir(self) -> Path:
        return self.root / "crops"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def simulations_dir(self) -> Path:
        return self.root / "simulations"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    def lot_dir(self, lot_id: str) -> Path:
        return self.lots_dir / lot_id

    def frame_path(self, lot_id: str, index: int) -> Path:
        return self.lots_dir / lot_id / "frames" / frame_filename(index)

    def crop_dir(self, track_id: int) -> Path:
        return self.crops_dir / str(track_id)

    def crop_relpath(self, track_id: int, view_index: int) -> str:
        return f"crops/{track_id}/{view_index:02d}.jpg"

    def crop_path(self, track_id: int, view_index: int) -> Path:
        return self.root / self.crop_relpath(track_id, view_index)

    def model_dir(self, model_id: str) -> Path:
        return self.models_dir / model_id

    def simulation_dir(self, sim_id: str) -> Path:
        return self.simulations_dir / sim_id

    def abspath(self, relpath: str) -> Path:
        return self.root / relpath

    def relpath(self, path: str | Path) -> str:
        return Path(path).resolve().relative_to(self.root).as_posix()

    # -- base de données --------------------------------------------------
    @contextmanager
    def session(self) -> Iterator[Session]:
        """Session transactionnelle : commit à la sortie, rollback en cas d'exception."""
        s = self._sessionmaker()
        try:
            yield s
            s.commit()
        except BaseException:
            s.rollback()
            raise
        finally:
            s.close()

    # -- configuration ----------------------------------------------------
    def load_config(self) -> dict[str, Any]:
        with self._config_lock:
            data = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
            return data if isinstance(data, dict) else {}

    def save_config(self, config: dict[str, Any]) -> None:
        with self._config_lock:
            tmp = self.config_path.with_suffix(".yaml.tmp")
            tmp.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
            tmp.replace(self.config_path)

    def update_config(self, changes: dict[str, Any]) -> dict[str, Any]:
        """Fusionne ``changes`` dans ``config.yaml`` (fusion récursive) et retourne la configuration complète."""
        with self._config_lock:
            config = deep_merge(self.load_config(), changes)
            self.save_config(config)
            return config

    def get_config(self, section: str, default: Any = None) -> Any:
        """Section de la configuration ; si c'est un dictionnaire, il est complété par ``default``."""
        value = self.load_config().get(section)
        if value is None:
            return copy.deepcopy(default)
        if isinstance(default, dict) and isinstance(value, dict):
            return deep_merge(default, value)
        return value

    def set_config(self, section: str, value: Any) -> None:
        """Remplace entièrement une section de la configuration."""
        with self._config_lock:
            config = self.load_config()
            config[section] = value
            self.save_config(config)

    @property
    def name(self) -> str:
        return str((self.load_config().get("project") or {}).get("name") or self.root.name)

    @property
    def user_name(self) -> str:
        return str((self.load_config().get("user") or {}).get("name") or "")

    # -- lots -------------------------------------------------------------
    def register_lot(self, lot_id: str, source_kind: str | None = None, sync_status: str = "done") -> m.Lot:
        """Enregistre (ou met à jour) en base un lot présent dans ``lots/<lot_id>/``.

        Lit ``lot.json`` et ``metadata.jsonl`` ; idempotent. Les images déjà
        connues ne sont pas dupliquées. Ne touche pas aux pistes existantes.
        """
        lot_dir = self.lot_dir(lot_id)
        manifest = LotManifest.load(lot_dir)
        metas = read_metadata(lot_dir)
        started = _parse_dt(manifest.started_at) or _parse_dt(manifest.created_at)
        with self.session() as s:
            lot = s.get(m.Lot, lot_id)
            if lot is None:
                lot = m.Lot(id=lot_id, name=manifest.name, type=manifest.type)
                s.add(lot)
            lot.name = manifest.name
            lot.type = manifest.type
            lot.farm = manifest.farm
            lot.operator = manifest.operator
            lot.comment = manifest.comment
            lot.status = manifest.status
            lot.source_kind = source_kind or str(manifest.source.get("kind", ""))
            lot.created_at = _parse_dt(manifest.created_at) or m.utcnow()
            lot.started_at = started
            lot.ended_at = _parse_dt(manifest.ended_at)
            lot.session_date = (started or lot.created_at).astimezone().strftime("%Y-%m-%d")
            lot.width = manifest.width
            lot.height = manifest.height
            lot.fps = manifest.camera.fps
            lot.camera = manifest.camera.model_dump()
            lot.frame_count = len(metas)
            lot.dropped_frames = manifest.stats.frames_dropped
            lot.duration_s = manifest.stats.duration_s
            lot.sync_status = sync_status
            s.flush()
            known = set(s.scalars(select(m.Frame.index).where(m.Frame.lot_id == lot_id)))
            new = [
                {
                    "lot_id": lot_id,
                    "index": fm.index,
                    "seq": fm.seq,
                    "sensor_ts_ns": fm.sensor_ts_ns,
                    "system_ts": fm.system_ts,
                    "encoder_count": fm.encoder_count,
                }
                for fm in metas
                if fm.index not in known
            ]
            for i in range(0, len(new), 5000):
                s.execute(m.Frame.__table__.insert(), new[i : i + 5000])
        return lot

    def stats(self) -> dict[str, int]:
        with self.session() as s:
            return {
                "lots": s.scalar(select(func.count()).select_from(m.Lot)) or 0,
                "frames": s.scalar(select(func.coalesce(func.sum(m.Lot.frame_count), 0))) or 0,
                "tracks": s.scalar(select(func.count()).select_from(m.Track)) or 0,
                "tracks_reviewed": s.scalar(select(func.count()).select_from(m.Label).where(m.Label.reviewed.is_(True))) or 0,
                "models": s.scalar(select(func.count()).select_from(m.Model)) or 0,
                "simulations": s.scalar(select(func.count()).select_from(m.Simulation)) or 0,
            }

    def info(self) -> dict[str, Any]:
        cfg = self.load_config().get("project") or {}
        return {
            "open": True,
            "path": str(self.root),
            "name": self.name,
            "created_at": cfg.get("created_at"),
            "stats": self.stats(),
        }


class ProjectManager:
    """Projet actuellement ouvert (un seul à la fois) et liste des projets récents."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._current: Project | None = None

    @property
    def current(self) -> Project | None:
        return self._current

    def require(self) -> Project:
        project = self._current
        if project is None:
            raise ProjectError("Aucun projet n'est ouvert.")
        return project

    def open(self, path: str | Path) -> Project:
        with self._lock:
            path = Path(path).expanduser()
            if not path.is_dir():
                raise ProjectError(f"Dossier introuvable : {path}")
            project = Project(path)
            self._swap(project)
            return project

    def create(self, parent_dir: str | Path, name: str) -> Project:
        name = name.strip()
        if not name or any(c in name for c in '/\\:*?"<>|'):
            raise ProjectError("Nom de projet invalide.")
        with self._lock:
            parent = Path(parent_dir).expanduser()
            if not parent.is_dir():
                raise ProjectError(f"Dossier parent introuvable : {parent}")
            project = Project.create(parent / name, name)
            self._swap(project)
            return project

    def close(self) -> None:
        with self._lock:
            if self._current is not None:
                self._current.close()
                self._current = None

    def _swap(self, project: Project) -> None:
        if self._current is not None:
            self._current.close()
        self._current = project
        self._remember(project)
        log.info("Projet ouvert : %s", project.root)

    def _remember(self, project: Project) -> None:
        state = settings.load_state()
        recent = [r for r in state.get("recent_projects", []) if r.get("path") != str(project.root)]
        recent.insert(
            0,
            {"path": str(project.root), "name": project.name, "last_opened": datetime.now(UTC).isoformat(timespec="seconds")},
        )
        settings.update_state(recent_projects=recent[:12])

    def recent(self) -> list[dict[str, Any]]:
        out = []
        for r in settings.load_state().get("recent_projects", []):
            out.append({**r, "exists": is_project_dir(r.get("path", ""))})
        return out
