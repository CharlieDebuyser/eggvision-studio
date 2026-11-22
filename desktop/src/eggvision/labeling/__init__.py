"""Module 3 — Labellisation : relecture rapide des pistes, historique, tableau de bord, apprentissage actif.

* :mod:`.search`    recherche filtrée / triée / paginée (clé) et détail d'une piste ;
* :mod:`.writes`    saisie, confirmation, masse, marquage des vues, annulation ;
* :mod:`.history`   historique paginé ;
* :mod:`.dashboard` effectifs, avancement, alertes ;
* :mod:`.active`    priorité d'apprentissage actif (contradiction / hésitation du modèle).
"""

from .active import ActiveModel, evaluate, list_models, resolve_model
from .common import LabelingError, NotFoundError, ensure_indexes, meta
from .dashboard import dashboard
from .history import history_rows, history_users
from .search import Page, TrackFilters, search_tracks, track_detail
from .writes import bulk, confirm, reopen, resolve, set_label, set_view_defect, undo_last

__all__ = [
    "ActiveModel",
    "LabelingError",
    "NotFoundError",
    "Page",
    "TrackFilters",
    "bulk",
    "confirm",
    "dashboard",
    "ensure_indexes",
    "evaluate",
    "history_rows",
    "history_users",
    "list_models",
    "meta",
    "reopen",
    "resolve",
    "resolve_model",
    "search_tracks",
    "set_label",
    "set_view_defect",
    "track_detail",
    "undo_last",
]
