"""Schéma SQLite d'un projet (``project.db``).

Règles :
  * les identifiants sont des chaînes lisibles (lots, modèles, simulations, jobs)
    ou des entiers auto-incrémentés (images, pistes, vues, détections) ;
  * les colonnes JSON portent les structures libres (réglages, métriques…) ;
  * tout ce qui est filtré ou trié dans l'interface est indexé : un projet peut
    dépasser 100 000 images.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

SCHEMA_VERSION = 1


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


class Meta(Base):
    """Paires clé/valeur du projet (version de schéma, etc.)."""

    __tablename__ = "meta"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text, default="")


class Lot(Base):
    """Un lot de prise de vue (dossier ``lots/<id>/``)."""

    __tablename__ = "lots"
    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    type: Mapped[str] = mapped_column(String(40), index=True)  # viable | declasse | declasse:<défaut> | production
    farm: Mapped[str] = mapped_column(String(200), default="", index=True)
    operator: Mapped[str] = mapped_column(String(200), default="")
    comment: Mapped[str] = mapped_column(Text, default="")
    session_date: Mapped[str] = mapped_column(String(10), default="", index=True)  # AAAA-MM-JJ (découpage par journée)
    status: Mapped[str] = mapped_column(String(20), default="termine", index=True)  # en_cours | termine | interrompu
    source_kind: Mapped[str] = mapped_column(String(20), default="")  # pi | usb | simulated | import
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    fps: Mapped[float] = mapped_column(Float, default=0.0)
    frame_count: Mapped[int] = mapped_column(Integer, default=0)
    dropped_frames: Mapped[int] = mapped_column(Integer, default=0)
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    camera: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # réglages caméra figés
    # Synchronisation depuis l'agent Pi : none | pending | syncing | done | error
    sync_status: Mapped[str] = mapped_column(String(20), default="done")
    # Détection / suivi : none | running | done | error
    detection_status: Mapped[str] = mapped_column(String(20), default="none", index=True)
    detection_params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    track_count: Mapped[int] = mapped_column(Integer, default=0)
    extra: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    frames: Mapped[list[Frame]] = relationship(back_populates="lot", cascade="all, delete-orphan", passive_deletes=True)
    tracks: Mapped[list[Track]] = relationship(back_populates="lot", cascade="all, delete-orphan", passive_deletes=True)


class Frame(Base):
    """Une image d'un lot (``lots/<lot_id>/frames/<index:06d>.jpg``)."""

    __tablename__ = "frames"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lot_id: Mapped[str] = mapped_column(ForeignKey("lots.id", ondelete="CASCADE"))
    index: Mapped[int] = mapped_column(Integer)
    seq: Mapped[int] = mapped_column(Integer, default=0)
    sensor_ts_ns: Mapped[int] = mapped_column(Integer, default=0)  # SQLite : entier 64 bits
    system_ts: Mapped[float] = mapped_column(Float, default=0.0)
    encoder_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    lot: Mapped[Lot] = relationship(back_populates="frames")
    __table_args__ = (UniqueConstraint("lot_id", "index", name="uq_frame_lot_index"),)


class Track(Base):
    """Une piste = un œuf physique suivi d'image en image dans un couloir."""

    __tablename__ = "tracks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lot_id: Mapped[str] = mapped_column(ForeignKey("lots.id", ondelete="CASCADE"), index=True)
    lane: Mapped[int] = mapped_column(Integer)
    start_frame: Mapped[int] = mapped_column(Integer)  # rang de la première image
    end_frame: Mapped[int] = mapped_column(Integer)
    start_ts: Mapped[float] = mapped_column(Float, default=0.0)  # secondes depuis le début du lot (horloge capteur)
    end_ts: Mapped[float] = mapped_column(Float, default=0.0)
    n_detections: Mapped[int] = mapped_column(Integer, default=0)
    n_views: Mapped[int] = mapped_column(Integer, default=0)
    # Qualité : ok | douteuse (courte, incomplète, floue…) ; ``flags`` détaille pourquoi.
    quality: Mapped[str] = mapped_column(String(20), default="ok", index=True)
    flags: Mapped[list[Any]] = mapped_column(JSON, default=list)  # ex. ["courte", "partielle", "floue"]
    mean_sharpness: Mapped[float] = mapped_column(Float, default=0.0)
    partial_ratio: Mapped[float] = mapped_column(Float, default=0.0)  # part des détections tronquées
    speed_px_s: Mapped[float] = mapped_column(Float, default=0.0)
    mean_major_px: Mapped[float] = mapped_column(Float, default=0.0)
    mean_minor_px: Mapped[float] = mapped_column(Float, default=0.0)
    extra: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # ex. vérité terrain synthétique

    lot: Mapped[Lot] = relationship(back_populates="tracks")
    views: Mapped[list[View]] = relationship(
        back_populates="track", cascade="all, delete-orphan", passive_deletes=True, order_by="View.index"
    )
    label: Mapped[Label | None] = relationship(back_populates="track", cascade="all, delete-orphan", uselist=False, passive_deletes=True)

    __table_args__ = (Index("ix_tracks_lot_lane_start", "lot_id", "lane", "start_frame"),)


class Detection(Base):
    """Une détection d'œuf dans une image (ellipse ajustée)."""

    __tablename__ = "detections"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lot_id: Mapped[str] = mapped_column(ForeignKey("lots.id", ondelete="CASCADE"))
    frame_index: Mapped[int] = mapped_column(Integer)
    track_id: Mapped[int | None] = mapped_column(ForeignKey("tracks.id", ondelete="CASCADE"), nullable=True, index=True)
    lane: Mapped[int] = mapped_column(Integer, default=-1)
    cx: Mapped[float] = mapped_column(Float)
    cy: Mapped[float] = mapped_column(Float)
    major: Mapped[float] = mapped_column(Float)  # longueur complète du grand axe (px)
    minor: Mapped[float] = mapped_column(Float)
    angle: Mapped[float] = mapped_column(Float, default=0.0)  # degrés
    partial: Mapped[bool] = mapped_column(Boolean, default=False)  # tronquée par le bord de l'image
    sharpness: Mapped[float] = mapped_column(Float, default=0.0)
    score: Mapped[float] = mapped_column(Float, default=1.0)

    __table_args__ = (Index("ix_detections_lot_frame", "lot_id", "frame_index"),)


class View(Base):
    """Une vue recadrée d'une piste (``crops/<track_id>/<index:02d>.jpg``)."""

    __tablename__ = "views"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    track_id: Mapped[int] = mapped_column(ForeignKey("tracks.id", ondelete="CASCADE"), index=True)
    index: Mapped[int] = mapped_column(Integer)  # 0 … n_views-1
    frame_index: Mapped[int] = mapped_column(Integer, default=0)
    path: Mapped[str] = mapped_column(String(200))  # relatif au dossier du projet
    sharpness: Mapped[float] = mapped_column(Float, default=0.0)
    partial: Mapped[bool] = mapped_column(Boolean, default=False)
    defect_visible: Mapped[bool] = mapped_column(Boolean, default=False)  # marquage facultatif à la relecture

    track: Mapped[Track] = relationship(back_populates="views")
    __table_args__ = (UniqueConstraint("track_id", "index", name="uq_view_track_index"),)


class Label(Base):
    """Label d'une piste (un œuf). Créé par le pré-label automatique, puis relu."""

    __tablename__ = "labels"
    track_id: Mapped[int] = mapped_column(ForeignKey("tracks.id", ondelete="CASCADE"), primary_key=True)
    verdict: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)  # viable | non_viable | incertain | exclu | None
    defect: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    source: Mapped[str] = mapped_column(String(10), default="auto")  # auto (pré-label) | humain
    reviewed: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    reviewer: Mapped[str] = mapped_column(String(100), default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    note: Mapped[str] = mapped_column(Text, default="")

    track: Mapped[Track] = relationship(back_populates="label")


class LabelHistory(Base):
    """Historique : qui a modifié quoi et quand."""

    __tablename__ = "label_history"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    track_id: Mapped[int] = mapped_column(ForeignKey("tracks.id", ondelete="CASCADE"), index=True)
    user: Mapped[str] = mapped_column(String(100), default="")
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    action: Mapped[str] = mapped_column(String(30), default="label")  # prelabel | label | marquage_vue | …
    before: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    after: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class DatasetSplit(Base):
    """Découpage entraînement / validation / test, par lot. Versionné et immuable une fois utilisé."""

    __tablename__ = "dataset_splits"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200))
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    strategy: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # {group_by, ratios, holdout_farms, holdout_dates, seed}
    assignments: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # {lot_id: "train" | "val" | "test"}
    summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # effectifs par sous-ensemble et par classe


class Model(Base):
    """Un modèle entraîné et versionné (dossier ``models/<id>/``)."""

    __tablename__ = "models"
    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    kind: Mapped[str] = mapped_column(String(20), default="classifier")  # classifier | anomaly | detector
    arch: Mapped[str] = mapped_column(String(80), default="")
    status: Mapped[str] = mapped_column(String(20), default="created", index=True)  # created | training | trained | failed | cancelled
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    split_id: Mapped[int | None] = mapped_column(ForeignKey("dataset_splits.id"), nullable=True)
    parent_id: Mapped[str | None] = mapped_column(String(80), nullable=True)  # reprise d'entraînement
    hparams: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    data_summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # données utilisées (lots, effectifs)
    history: Mapped[list[Any]] = mapped_column(JSON, default=list)  # courbes : une entrée par époque
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # résumé du rapport d'évaluation (jeu de test)
    threshold: Mapped[float] = mapped_column(Float, default=0.5)
    n_views: Mapped[int] = mapped_column(Integer, default=12)
    input_size: Mapped[int] = mapped_column(Integer, default=224)
    export: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # {package_dir, zip, parity, files, bench…}
    job_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    extra: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class Prediction(Base):
    """Score d'un modèle pour une piste (rapport d'évaluation, apprentissage actif)."""

    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(ForeignKey("models.id", ondelete="CASCADE"))
    track_id: Mapped[int] = mapped_column(ForeignKey("tracks.id", ondelete="CASCADE"), index=True)
    subset: Mapped[str] = mapped_column(String(10), default="")  # train | val | test | "" (hors découpage)
    score: Mapped[float] = mapped_column(Float)  # score de défaut de l'œuf, dans [0, 1]
    view_scores: Mapped[list[Any]] = mapped_column(JSON, default=list)

    __table_args__ = (
        UniqueConstraint("model_id", "track_id", name="uq_prediction_model_track"),
        Index("ix_predictions_model_score", "model_id", "score"),
    )


class Simulation(Base):
    """Une simulation de tri (dossier ``simulations/<id>/``)."""

    __tablename__ = "simulations"
    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    model_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="created", index=True)  # created | running | done | failed | cancelled
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    results: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    job_id: Mapped[str | None] = mapped_column(String(40), nullable=True)


class CameraPreset(Base):
    """Préréglage caméra nommé."""

    __tablename__ = "camera_presets"
    name: Mapped[str] = mapped_column(String(100), primary_key=True)
    settings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class Job(Base):
    """Tâche de fond (synchronisation, détection, entraînement, simulation…)."""

    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    kind: Mapped[str] = mapped_column(String(30), index=True)
    title: Mapped[str] = mapped_column(String(300), default="")
    status: Mapped[str] = mapped_column(String(12), default="queued", index=True)  # queued | running | done | failed | cancelled
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str] = mapped_column(Text, default="")
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
