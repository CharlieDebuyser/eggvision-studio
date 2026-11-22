"""Constantes métier communes au PC et au Raspberry Pi."""

from __future__ import annotations

FORMAT_VERSION = 1

# --- Défauts (l'ordre fixe les raccourcis clavier 1 à 6) -------------------
DEFECT_TYPES: tuple[str, ...] = ("souillure", "coulure", "felure", "plume", "sang", "deformation")
DEFECT_LABELS: dict[str, str] = {
    "souillure": "Souillure",
    "coulure": "Coulure",
    "felure": "Fêlure",
    "plume": "Plume",
    "sang": "Sang",
    "deformation": "Déformation",
}

# --- Verdicts d'une piste (un œuf) -----------------------------------------
VERDICT_VIABLE = "viable"
VERDICT_NON_VIABLE = "non_viable"
VERDICT_UNCERTAIN = "incertain"
VERDICT_EXCLUDED = "exclu"
VERDICTS: tuple[str, ...] = (VERDICT_VIABLE, VERDICT_NON_VIABLE, VERDICT_UNCERTAIN, VERDICT_EXCLUDED)

# --- Types de lot ----------------------------------------------------------
LOT_VIABLE = "viable"
LOT_DECLASSE = "declasse"
LOT_PRODUCTION = "production"

# --- Statuts de lot --------------------------------------------------------
LOT_STATUS_RUNNING = "en_cours"
LOT_STATUS_DONE = "termine"
LOT_STATUS_INTERRUPTED = "interrompu"
LOT_STATUSES = (LOT_STATUS_RUNNING, LOT_STATUS_DONE, LOT_STATUS_INTERRUPTED)

# --- Caméra (Raspberry Pi Global Shutter, IMX296) --------------------------
SENSOR_WIDTH = 1456
SENSOR_HEIGHT = 1088
DEFAULT_EXPOSURE_US = 500
DEFAULT_FPS = 30.0
JPEG_QUALITY = 95

# --- Vues par œuf ----------------------------------------------------------
DEFAULT_VIEW_SIZE = 224
DEFAULT_N_VIEWS = 12
DEFAULT_N_LANES = 6


def parse_lot_type(lot_type: str) -> tuple[str, str | None]:
    """Décompose un type de lot en (type de base, défaut éventuel).

    >>> parse_lot_type("declasse:felure")
    ('declasse', 'felure')
    >>> parse_lot_type("viable")
    ('viable', None)
    """
    base, _, defect = lot_type.partition(":")
    base = base.strip().lower()
    defect = defect.strip().lower() or None
    if base not in (LOT_VIABLE, LOT_DECLASSE, LOT_PRODUCTION):
        raise ValueError(f"Type de lot inconnu : {lot_type!r}")
    if defect is not None:
        if base != LOT_DECLASSE:
            raise ValueError(f"Seul un lot « declasse » peut préciser un défaut : {lot_type!r}")
        if defect not in DEFECT_TYPES:
            raise ValueError(f"Type de défaut inconnu : {defect!r}")
    return base, defect


def format_lot_type(base: str, defect: str | None = None) -> str:
    """Inverse de :func:`parse_lot_type` (valide au passage)."""
    lot_type = f"{base}:{defect}" if defect else base
    parse_lot_type(lot_type)
    return lot_type


def prelabel_for_lot_type(lot_type: str) -> tuple[str | None, str | None]:
    """Pré-label automatique (verdict, défaut) déduit du type de lot.

    Un lot ``production`` ne donne aucun pré-label (statut inconnu).
    """
    base, defect = parse_lot_type(lot_type)
    if base == LOT_VIABLE:
        return VERDICT_VIABLE, None
    if base == LOT_DECLASSE:
        return VERDICT_NON_VIABLE, defect
    return None, None
