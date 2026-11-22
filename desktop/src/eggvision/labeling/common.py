"""Outils communs du module de labellisation (constantes, instantanés de label, index)."""

from __future__ import annotations

import logging
import threading
import weakref
from datetime import UTC, datetime
from typing import Any

from eggvision_shared.constants import DEFECT_LABELS, DEFECT_TYPES, VERDICTS, prelabel_for_lot_type
from sqlalchemy import text
from sqlalchemy.engine import Engine

from ..db import models as m

log = logging.getLogger(__name__)

# Valeur spéciale des filtres : « sans label » (verdict) ou « sans type » (défaut).
NONE_KEY = "aucun"

VERDICT_LABELS: dict[str, str] = {
    "viable": "Viable",
    "non_viable": "Non viable",
    "incertain": "Incertain",
    "exclu": "Exclu",
}

# Actions de l'historique.
ACTION_PRELABEL = "prelabel"
ACTION_LABEL = "label"
ACTION_CONFIRM = "confirmation"
ACTION_BULK = "label_masse"
ACTION_REOPEN = "reouverture"
ACTION_VIEW = "marquage_vue"
ACTION_UNDO = "annulation"
UNDOABLE_ACTIONS = frozenset({ACTION_LABEL, ACTION_CONFIRM, ACTION_BULK, ACTION_REOPEN, ACTION_VIEW})
ACTION_LABELS: dict[str, str] = {
    ACTION_PRELABEL: "Pré-label automatique",
    ACTION_LABEL: "Label",
    ACTION_CONFIRM: "Confirmation",
    ACTION_BULK: "Label en masse",
    ACTION_REOPEN: "Réouverture",
    ACTION_VIEW: "Marquage d'une vue",
    ACTION_UNDO: "Annulation",
}

LABEL_FIELDS = ("verdict", "defect", "reviewed", "source", "reviewer", "note")
UNKNOWN_USER = "inconnu"
MAX_LIMIT = 500
MAX_BULK = 10_000


class LabelingError(ValueError):
    """Erreur de saisie présentable à l'utilisateur (message en français)."""


class NotFoundError(LookupError):
    """Ressource introuvable (message en français)."""


class _Unset:
    def __repr__(self) -> str:  # pragma: no cover - aide au débogage
        return "UNSET"


UNSET: Any = _Unset()


def iso(dt: datetime | None) -> str | None:
    """Date ISO 8601 (UTC) ; SQLite rend des dates naïves, implicitement UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.isoformat(timespec="milliseconds")


def parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def snapshot(label: m.Label | None) -> dict[str, Any]:
    """Instantané d'un label, tel qu'enregistré dans l'historique (``{}`` si absent)."""
    if label is None:
        return {}
    return {
        "verdict": label.verdict,
        "defect": label.defect,
        "reviewed": bool(label.reviewed),
        "source": label.source,
        "reviewer": label.reviewer or "",
        "note": label.note or "",
    }


def prelabel_of(lot_type: str | None) -> dict[str, str | None]:
    """Pré-label (verdict, défaut) déduit du type de lot ; vide si le type est inconnu."""
    try:
        verdict, defect = prelabel_for_lot_type(lot_type or "")
    except ValueError:
        verdict, defect = None, None
    return {"verdict": verdict, "defect": defect}


def validate_verdict(verdict: str | None) -> str | None:
    if verdict is not None and verdict not in VERDICTS:
        raise LabelingError(f"Verdict inconnu : « {verdict} » (attendu : {', '.join(VERDICTS)}).")
    return verdict


def validate_defect(defect: str | None) -> str | None:
    if defect is not None and defect not in DEFECT_TYPES:
        raise LabelingError(f"Type de défaut inconnu : « {defect} » (attendu : {', '.join(DEFECT_TYPES)}).")
    return defect


def meta() -> dict[str, Any]:
    """Vocabulaire de l'écran : verdicts, défauts (et leurs touches), actions."""
    return {
        "verdicts": [{"key": v, "label": VERDICT_LABELS[v]} for v in VERDICTS],
        "defects": [{"key": d, "label": DEFECT_LABELS[d], "hotkey": str(i + 1)} for i, d in enumerate(DEFECT_TYPES)],
        "actions": [{"key": k, "label": v} for k, v in ACTION_LABELS.items()],
        "none_key": NONE_KEY,
        "max_limit": MAX_LIMIT,
        "max_bulk": MAX_BULK,
    }


# -- index complémentaires ----------------------------------------------------
# Créés à la volée (``CREATE INDEX IF NOT EXISTS``) : strictement additifs, ils ne
# modifient pas le schéma déclaré dans ``db/models.py``.
_INDEXES = (
    # ordre naturel (lot, temps, couloir) : pagination par clé sans tri
    "CREATE INDEX IF NOT EXISTS ix_lbl_tracks_natural ON tracks (lot_id, start_frame, lane)",
    # tri « plus récemment modifiées »
    "CREATE INDEX IF NOT EXISTS ix_lbl_labels_updated ON labels (updated_at)",
    # tableau de bord / filtres combinés
    "CREATE INDEX IF NOT EXISTS ix_lbl_labels_verdict_defect ON labels (verdict, defect, reviewed)",
    # historique par utilisateur (annulation, filtre)
    "CREATE INDEX IF NOT EXISTS ix_lbl_history_user_at ON label_history (user, at)",
)
_indexed: weakref.WeakSet[Engine] = weakref.WeakSet()
_indexed_lock = threading.Lock()


def ensure_indexes(engine: Engine) -> None:
    """Crée (une fois par moteur) les index utiles à la labellisation."""
    if engine in _indexed:
        return
    with _indexed_lock:
        if engine in _indexed:
            return
        with engine.begin() as conn:
            for ddl in _INDEXES:
                conn.execute(text(ddl))
        _indexed.add(engine)
        log.debug("Index de labellisation vérifiés")
