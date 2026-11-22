"""Tâches de fond : progression, annulation coopérative, journaux consultables.

Usage côté module ::

    def _run(ctx: JobContext) -> dict:
        for i, item in enumerate(items):
            ctx.check_cancelled()                      # lève JobCancelled si l'utilisateur a annulé
            ...
            ctx.progress((i + 1) / len(items), f"{i + 1}/{len(items)} images")
            ctx.log("détail écrit dans le journal de la tâche")
        return {"n": len(items)}                       # devient job.result

    job = jobs.submit("detection", "Détection du lot X", _run, params={"lot_id": "X"}, group="cpu")

``ctx.publish(**données)`` met à jour ``job.result`` en cours d'exécution (courbes
d'apprentissage en direct, compteurs…). Les tâches d'un même ``group`` s'exécutent
l'une après l'autre (ex. ``"gpu"`` pour ne pas lancer deux entraînements à la fois).
"""

from __future__ import annotations

import logging
import threading
import traceback
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select

from . import settings
from .db import models as m
from .project import Project

log = logging.getLogger(__name__)

ACTIVE = ("queued", "running")


class JobCancelled(Exception):
    """Levée par ``ctx.check_cancelled()`` quand l'annulation a été demandée."""


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.isoformat(timespec="milliseconds")


class JobState:
    def __init__(self, kind: str, title: str, params: dict[str, Any], project: Project | None, log_path: Path) -> None:
        self.id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.kind = kind
        self.title = title
        self.params = params
        self.project = project
        self.status = "queued"
        self.progress = 0.0
        self.message = ""
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        self.created_at = _now()
        self.started_at: datetime | None = None
        self.ended_at: datetime | None = None
        self.log_path = log_path
        self.cancel_event = threading.Event()
        self.done_event = threading.Event()
        self.thread: threading.Thread | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "status": self.status,
            "progress": round(float(self.progress), 4),
            "message": self.message,
            "params": _jsonable(self.params),
            "result": _jsonable(self.result),
            "error": self.error,
            "created_at": _iso(self.created_at),
            "started_at": _iso(self.started_at),
            "ended_at": _iso(self.ended_at),
        }


class JobContext:
    """Passé à la fonction d'une tâche."""

    def __init__(self, manager: JobManager, state: JobState) -> None:
        self._manager = manager
        self._state = state
        self._log_lock = threading.Lock()

    @property
    def job_id(self) -> str:
        return self._state.id

    @property
    def project(self) -> Project | None:
        return self._state.project

    @property
    def params(self) -> dict[str, Any]:
        return self._state.params

    @property
    def cancelled(self) -> bool:
        return self._state.cancel_event.is_set()

    def check_cancelled(self) -> None:
        if self._state.cancel_event.is_set():
            raise JobCancelled()

    def progress(self, fraction: float, message: str | None = None) -> None:
        self._state.progress = min(1.0, max(0.0, float(fraction)))
        if message is not None:
            self._state.message = message

    def message(self, message: str) -> None:
        self._state.message = message

    def publish(self, **data: Any) -> None:
        """Fusionne ``data`` dans ``job.result`` (visible en direct par l'interface)."""
        current = dict(self._state.result or {})
        current.update(data)
        self._state.result = current

    def log(self, text: str) -> None:
        line = f"{datetime.now().strftime('%H:%M:%S')} {text}\n"
        with self._log_lock:
            with open(self._state.log_path, "a", encoding="utf-8") as f:
                f.write(line)

    def subprogress(self, start: float, end: float) -> Callable[[float, str | None], None]:
        """Fonction de progression ramenée à l'intervalle [start, end] (pour enchaîner des étapes)."""

        def _cb(fraction: float, message: str | None = None) -> None:
            self.progress(start + (end - start) * min(1.0, max(0.0, fraction)), message)

        return _cb


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.RLock()
        self._group_locks: dict[str, threading.Lock] = {}

    # -- soumission -------------------------------------------------------
    def submit(
        self,
        kind: str,
        title: str,
        fn: Callable[[JobContext], dict[str, Any] | None],
        params: dict[str, Any] | None = None,
        project: Project | None = None,
        group: str | None = None,
    ) -> JobState:
        """Lance ``fn(ctx)`` dans un fil d'exécution et retourne immédiatement l'état de la tâche."""
        logs_dir = (project.logs_dir if project is not None else settings.data_dir() / "logs") / "jobs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        state = JobState(kind, title, dict(params or {}), project, logs_dir / "pending.log")
        state.log_path = logs_dir / f"{state.id}.log"
        state.log_path.touch()
        with self._lock:
            self._jobs[state.id] = state
            group_lock = self._group_locks.setdefault(group, threading.Lock()) if group else None
        self._persist(state)
        ctx = JobContext(self, state)
        thread = threading.Thread(target=self._run, args=(state, ctx, fn, group_lock), name=f"job-{state.id}", daemon=True)
        state.thread = thread
        thread.start()
        return state

    def _run(self, state: JobState, ctx: JobContext, fn: Callable[[JobContext], Any], group_lock: threading.Lock | None) -> None:
        acquired = False
        try:
            if group_lock is not None:
                while not group_lock.acquire(timeout=0.2):
                    if state.cancel_event.is_set():
                        raise JobCancelled()
                acquired = True
            if state.cancel_event.is_set():
                raise JobCancelled()
            state.status = "running"
            state.started_at = _now()
            self._persist(state)
            ctx.log(f"Début : {state.title}")
            result = fn(ctx)
            if isinstance(result, dict):
                merged = dict(state.result or {})
                merged.update(result)
                state.result = merged
            state.progress = 1.0
            state.status = "done"
            ctx.log("Terminé.")
        except JobCancelled:
            state.status = "cancelled"
            state.message = "Annulé"
            ctx.log("Annulé par l'utilisateur.")
        except Exception as exc:  # noqa: BLE001 — toute erreur est rapportée à l'utilisateur
            state.status = "failed"
            state.error = str(exc) or exc.__class__.__name__
            ctx.log("ERREUR : " + state.error)
            ctx.log(traceback.format_exc())
            log.exception("Tâche %s (%s) en échec", state.id, state.kind)
        finally:
            if acquired and group_lock is not None:
                group_lock.release()
            state.ended_at = _now()
            self._persist(state)
            state.done_event.set()

    # -- consultation -----------------------------------------------------
    def get(self, job_id: str) -> JobState | None:
        return self._jobs.get(job_id)

    def get_dict(self, job_id: str, project: Project | None = None) -> dict[str, Any] | None:
        state = self._jobs.get(job_id)
        if state is not None:
            return state.to_dict()
        if project is not None:
            with project.session() as s:
                row = s.get(m.Job, job_id)
                if row is not None:
                    return _row_to_dict(row)
        return None

    def list(self, project: Project | None = None, kind: str | None = None, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Tâches en mémoire + historique du projet ouvert, plus récentes d'abord."""
        with self._lock:
            items = {j.id: j.to_dict() for j in self._jobs.values() if j.project is None or j.project is project}
        if project is not None:
            with project.session() as s:
                rows = s.scalars(select(m.Job).order_by(m.Job.created_at.desc()).limit(max(limit, 1) * 2)).all()
                for row in rows:
                    items.setdefault(row.id, _row_to_dict(row))
        out = [j for j in items.values() if (kind is None or j["kind"] == kind) and (status is None or j["status"] == status)]
        out.sort(key=lambda j: j["created_at"] or "", reverse=True)
        return out[:limit]

    def active(self, kind: str | None = None, project: Project | None = None) -> list[JobState]:
        with self._lock:
            return [
                j
                for j in self._jobs.values()
                if j.status in ACTIVE and (kind is None or j.kind == kind) and (project is None or j.project is project)
            ]

    def cancel(self, job_id: str) -> JobState | None:
        state = self._jobs.get(job_id)
        if state is not None and state.status in ACTIVE:
            state.cancel_event.set()
            state.message = "Annulation en cours…"
        return state

    def cancel_all(self, project: Project | None = None, wait: float = 10.0) -> None:
        """Annule les tâches actives (du projet donné, ou toutes) et attend leur fin."""
        jobs = self.active(project=project)
        for j in jobs:
            j.cancel_event.set()
        for j in jobs:
            j.done_event.wait(wait)

    def wait(self, job_id: str, timeout: float | None = None) -> dict[str, Any]:
        """Attend la fin d'une tâche (utile dans les tests) et retourne son état."""
        state = self._jobs[job_id]
        if not state.done_event.wait(timeout):
            raise TimeoutError(f"La tâche {job_id} n'est pas terminée après {timeout} s")
        return state.to_dict()

    def log_path(self, job_id: str, project: Project | None = None) -> Path | None:
        state = self._jobs.get(job_id)
        if state is not None:
            return state.log_path
        candidates = [settings.data_dir() / "logs" / "jobs" / f"{job_id}.log"]
        if project is not None:
            candidates.insert(0, project.logs_dir / "jobs" / f"{job_id}.log")
        for p in candidates:
            if p.exists():
                return p
        return None

    def read_log(self, job_id: str, offset: int = 0, project: Project | None = None, max_bytes: int = 256_000) -> dict[str, Any]:
        path = self.log_path(job_id, project)
        state = self._jobs.get(job_id)
        finished = state is None or state.status not in ACTIVE
        if path is None or not path.exists():
            return {"text": "", "next_offset": offset, "eof": finished}
        size = path.stat().st_size
        offset = max(0, min(offset, size))
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read(max_bytes)
        return {"text": data.decode("utf-8", errors="replace"), "next_offset": offset + len(data), "eof": finished and offset + len(data) >= size}

    # -- persistance (historique du projet) -------------------------------
    def _persist(self, state: JobState) -> None:
        project = state.project
        if project is None:
            return
        try:
            with project.session() as s:
                row = s.get(m.Job, state.id)
                if row is None:
                    row = m.Job(id=state.id, kind=state.kind)
                    s.add(row)
                row.title = state.title
                row.status = state.status
                row.progress = state.progress
                row.message = state.message
                row.params = _jsonable(state.params)
                row.result = _jsonable(state.result)
                row.error = state.error
                row.created_at = state.created_at
                row.started_at = state.started_at
                row.ended_at = state.ended_at
        except Exception:  # la persistance ne doit jamais faire échouer une tâche
            log.exception("Impossible d'enregistrer la tâche %s", state.id)

    def mark_stale(self, project: Project) -> None:
        """À l'ouverture d'un projet : les tâches restées actives en base (arrêt brutal) passent en échec."""
        with project.session() as s:
            rows = s.scalars(select(m.Job).where(m.Job.status.in_(ACTIVE))).all()
            for row in rows:
                if row.id not in self._jobs:
                    row.status = "failed"
                    row.error = "Interrompue par la fermeture de l'application."
                    row.ended_at = _now()


def _jsonable(value: Any) -> Any:
    """Copie sérialisable en JSON (types NumPy, chemins, dates…)."""
    import json

    def default(o: Any) -> Any:
        if hasattr(o, "item"):
            return o.item()
        if hasattr(o, "tolist"):
            return o.tolist()
        return str(o)

    if value is None:
        return None
    return json.loads(json.dumps(value, default=default))


def _row_to_dict(row: m.Job) -> dict[str, Any]:
    return {
        "id": row.id,
        "kind": row.kind,
        "title": row.title,
        "status": row.status,
        "progress": row.progress,
        "message": row.message,
        "params": row.params or {},
        "result": row.result,
        "error": row.error,
        "created_at": _iso(row.created_at),
        "started_at": _iso(row.started_at),
        "ended_at": _iso(row.ended_at),
    }
