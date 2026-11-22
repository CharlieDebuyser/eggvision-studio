"""API : tâches de fond."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from .. import state

router = APIRouter(prefix="/api/jobs", tags=["tâches"])


@router.get("")
def list_jobs(kind: str | None = None, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    return state.jobs.list(project=state.projects.current, kind=kind or None, status=status or None, limit=max(1, min(limit, 1000)))


@router.get("/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = state.jobs.get_dict(job_id, state.projects.current)
    if job is None:
        raise HTTPException(404, "Tâche introuvable.")
    return job


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    state.jobs.cancel(job_id)
    return get_job(job_id)


@router.get("/{job_id}/log")
def job_log(job_id: str, offset: int = 0) -> dict[str, Any]:
    return state.jobs.read_log(job_id, offset=offset, project=state.projects.current)
