"""API : images (images complètes des lots, vues recadrées des pistes)."""

from __future__ import annotations

import cv2
from eggvision_shared.formats import encode_jpeg, read_image
from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import FileResponse

from .. import state
from ..project import Project

router = APIRouter(prefix="/api/media", tags=["médias"])

_CACHE = {"Cache-Control": "private, max-age=3600"}


@router.get("/frame/{lot_id}/{index}")
def frame(lot_id: str, index: int, w: int | None = None, project: Project = Depends(state.get_project)):
    """Image n° ``index`` d'un lot ; ``w`` = largeur maximale souhaitée (miniature)."""
    path = project.frame_path(lot_id, index)
    if not path.is_file():
        raise HTTPException(404, "Image introuvable.")
    if not w:
        return FileResponse(path, media_type="image/jpeg", headers=_CACHE)
    img = read_image(path)
    if w < img.shape[1]:
        h = max(1, round(img.shape[0] * w / img.shape[1]))
        img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)
    return Response(encode_jpeg(img, 85), media_type="image/jpeg", headers=_CACHE)


@router.get("/crop/{track_id}/{view_index}")
def crop(track_id: int, view_index: int, project: Project = Depends(state.get_project)):
    """Vue recadrée n° ``view_index`` d'une piste."""
    path = project.crop_path(track_id, view_index)
    if not path.is_file():
        raise HTTPException(404, "Vue introuvable.")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=60"})
