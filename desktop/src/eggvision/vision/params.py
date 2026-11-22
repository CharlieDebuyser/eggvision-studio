"""Paramètres typés de la détection et du suivi, persistés dans ``config.yaml``.

Sections ``detection`` et ``tracking`` du fichier de configuration du projet.
Toutes les tailles physiques sont exprimées en millimètres et converties en
pixels grâce à ``scene.mm_per_px`` : les mêmes réglages valent donc quelle que
soit la résolution des images (728 × 544, 1456 × 1088…).

Chaque champ porte, dans ``json_schema_extra``, un libellé français, une unité,
un pas et un groupe : l'écran de réglages construit ses curseurs à partir de
:func:`params_schema`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..project import Project, deep_merge


def _f(default: Any, label: str, *, ge: float | None = None, le: float | None = None, step: float | None = None,
       unit: str = "", group: str = "", help: str = "", advanced: bool = False) -> Any:
    extra: dict[str, Any] = {"label": label, "unit": unit, "group": group, "advanced": advanced}
    if step is not None:
        extra["step"] = step
    return Field(default, ge=ge, le=le, description=help or label, json_schema_extra=extra)


class DetectionParams(BaseModel):
    """Réglages de la détection d'œufs dans une image."""

    model_config = ConfigDict(extra="ignore")

    method: Literal["classic", "nn"] = _f("classic", "Méthode", group="Méthode",
                                          help="classic : segmentation couleur/luminosité ; nn : détecteur entraînable")
    # --- Segmentation --------------------------------------------------------
    work_mm_per_px: float = _f(0.8, "Résolution de travail", ge=0.2, le=3.0, step=0.05, unit="mm/px", group="Segmentation",
                               help="Taille d'un pixel de l'image sous-échantillonnée utilisée pour segmenter (plus grand = plus rapide)")
    threshold_mode: Literal["fixe", "auto"] = _f("fixe", "Seuil", group="Segmentation",
                                                 help="fixe : valeur ci-dessous ; auto : seuil d'Otsu recalculé à chaque image")
    threshold: float = _f(80.0, "Seuil de segmentation", ge=0, le=255, step=1, group="Segmentation",
                          help="Seuil appliqué à l'indice « luminosité + poids × (rouge − bleu) »")
    auto_min_threshold: float = _f(55.0, "Seuil auto minimal", ge=0, le=255, step=1, group="Segmentation", advanced=True,
                                   help="Plancher du seuil automatique (évite de segmenter les rouleaux sur une image vide)")
    warm_weight: float = _f(1.0, "Poids de la teinte chaude", ge=0, le=3, step=0.05, group="Segmentation",
                            help="Poids de (rouge − bleu) : aide à séparer les œufs bruns de rouleaux sombres ou bleutés")
    blur_mm: float = _f(1.0, "Lissage", ge=0, le=10, step=0.5, unit="mm", group="Segmentation", advanced=True)
    open_mm: float = _f(2.0, "Ouverture morphologique", ge=0, le=15, step=0.5, unit="mm", group="Segmentation",
                        help="Supprime les petits éléments clairs (reflets, poussières)")
    close_mm: float = _f(3.0, "Fermeture morphologique", ge=0, le=15, step=0.5, unit="mm", group="Segmentation",
                         help="Comble les petites taches sombres (souillures) sur l'œuf")
    split_touching: bool = _f(True, "Séparer les œufs qui se touchent", group="Segmentation")
    split_core: float = _f(0.6, "Seuil de cœur (séparation)", ge=0.3, le=0.9, step=0.05, group="Segmentation", advanced=True,
                           help="Fraction de la distance maximale au bord retenue comme cœur d'œuf lors de la séparation")
    # --- Filtres taille / forme ------------------------------------------------
    min_major_mm: float = _f(42.0, "Grand axe minimal", ge=5, le=200, step=1, unit="mm", group="Filtres")
    max_major_mm: float = _f(82.0, "Grand axe maximal", ge=5, le=300, step=1, unit="mm", group="Filtres")
    min_minor_mm: float = _f(32.0, "Petit axe minimal", ge=5, le=200, step=1, unit="mm", group="Filtres")
    max_minor_mm: float = _f(60.0, "Petit axe maximal", ge=5, le=300, step=1, unit="mm", group="Filtres")
    max_axis_ratio: float = _f(1.9, "Allongement maximal", ge=1.0, le=4.0, step=0.05, group="Filtres",
                               help="Rapport grand axe / petit axe maximal")
    min_solidity: float = _f(0.80, "Solidité minimale", ge=0.0, le=1.0, step=0.01, group="Filtres",
                             help="Surface / surface de l'enveloppe convexe")
    min_fill: float = _f(0.70, "Remplissage minimal de l'ellipse", ge=0.0, le=1.5, step=0.01, group="Filtres", advanced=True)
    partial_margin_mm: float = _f(1.5, "Marge de bord (partiel)", ge=0, le=20, step=0.5, unit="mm", group="Filtres",
                                  help="Une détection à moins de cette distance du bord de l'image est marquée partielle")
    lateral_partial_mm: float = _f(4.0, "Troncature latérale tolérée", ge=0, le=50, step=0.5, unit="mm", group="Filtres", advanced=True,
                                   help="Un œuf qui dépasse d'un bord latéral (parallèle au défilement) de moins que cette valeur n'est pas marqué partiel")
    partial_min_visible: float = _f(0.25, "Part visible minimale d'un partiel", ge=0.0, le=1.0, step=0.05, group="Filtres",
                                    help="Les œufs tronqués dont la part visible est inférieure sont ignorés")
    sharpness_region: float = _f(0.6, "Zone de mesure de netteté", ge=0.2, le=1.0, step=0.05, group="Filtres", advanced=True,
                                 help="Fraction des axes de l'ellipse utilisée pour mesurer la netteté")
    # --- Détecteur entraînable -------------------------------------------------
    nn_model: str | None = _f(None, "Détecteur entraînable", group="Détecteur entraînable",
                              help="Identifiant du détecteur (models/detector/<id>) utilisé quand la méthode est « nn »")
    nn_threshold: float = _f(0.35, "Seuil de confiance", ge=0.01, le=0.99, step=0.01, group="Détecteur entraînable")


class TrackingParams(BaseModel):
    """Réglages du suivi (association image à image) et de l'extraction des vues."""

    model_config = ConfigDict(extra="ignore")

    initial_speed_mm_s: float | None = _f(None, "Vitesse initiale", ge=0, le=5000, step=1, unit="mm/s", group="Suivi",
                                          help="Vide = estimée automatiquement dès les premières images")
    max_speed_mm_s: float = _f(800.0, "Vitesse maximale", ge=1, le=10000, step=10, unit="mm/s", group="Suivi",
                               help="Borne de recherche tant que la vitesse n'est pas encore estimée")
    gate_mm: float = _f(18.0, "Fenêtre d'association (défilement)", ge=1, le=200, step=1, unit="mm", group="Suivi",
                        help="Écart toléré entre position prédite et position observée, le long du défilement")
    gate_speed_frac: float = _f(0.35, "Fenêtre proportionnelle au déplacement", ge=0, le=2, step=0.05, group="Suivi", advanced=True,
                                help="Fenêtre supplémentaire = fraction du déplacement prédit (images manquantes)")
    lateral_gate_mm: float = _f(15.0, "Fenêtre d'association (latérale)", ge=1, le=200, step=1, unit="mm", group="Suivi")
    max_missed: int = _f(4, "Détections manquées tolérées", ge=0, le=50, step=1, group="Suivi",
                         help="Nombre d'images consécutives sans détection avant de clore une piste")
    speed_history: int = _f(15, "Mémoire de l'estimation de vitesse", ge=1, le=500, step=1, unit="images", group="Suivi", advanced=True)
    # --- Vues -----------------------------------------------------------------
    n_views: int = _f(12, "Vues par œuf", ge=1, le=64, step=1, group="Vues")
    view_size: int = _f(224, "Taille des vues", ge=32, le=1024, step=16, unit="px", group="Vues")
    crop_margin: float = _f(1.15, "Marge de recadrage", ge=1.0, le=2.0, step=0.01, group="Vues")
    jpeg_quality: int = _f(92, "Qualité JPEG des vues", ge=50, le=100, step=1, group="Vues", advanced=True)
    view_min_sharpness_ratio: float = _f(0.5, "Netteté minimale d'une vue", ge=0, le=1, step=0.05, group="Vues", advanced=True,
                                         help="Une détection moins nette que cette fraction de la médiane de la piste n'est pas retenue comme vue")
    # --- Qualité --------------------------------------------------------------
    min_views: int = _f(8, "Vues minimales (piste exploitable)", ge=1, le=64, step=1, group="Qualité")
    edge_margin_mm: float = _f(12.0, "Marge d'entrée / sortie", ge=0, le=100, step=1, unit="mm", group="Qualité",
                               help="Une piste est complète si elle commence et finit à moins de cette distance des bords d'entrée et de sortie")
    max_partial_ratio: float = _f(0.6, "Part maximale de détections partielles", ge=0, le=1, step=0.05, group="Qualité")
    max_gap_ratio: float = _f(0.3, "Part maximale de détections manquées", ge=0, le=1, step=0.05, group="Qualité")
    speed_tolerance: float = _f(0.25, "Écart de vitesse toléré", ge=0, le=5, step=0.05, group="Qualité",
                                help="Écart relatif maximal entre la vitesse de la piste et la vitesse de la table")
    size_cv_max: float = _f(0.15, "Variation de taille maximale", ge=0, le=2, step=0.01, group="Qualité", advanced=True,
                            help="Coefficient de variation maximal du petit axe le long de la piste (fusion / séparation d'œufs)")
    blur_ratio: float = _f(0.5, "Seuil de flou relatif", ge=0, le=1, step=0.05, group="Qualité",
                           help="Piste « floue » si sa netteté moyenne est inférieure à cette fraction de la médiane du lot")
    min_lane_agreement: float = _f(0.8, "Accord minimal sur le couloir", ge=0, le=1, step=0.05, group="Qualité", advanced=True)
    store_unassigned: bool = _f(True, "Conserver les détections hors piste", group="Qualité", advanced=True)


SECTIONS: dict[str, type[BaseModel]] = {"detection": DetectionParams, "tracking": TrackingParams}


class ParamsError(ValueError):
    """Paramètres invalides (message en français)."""


def _validate(model: type[BaseModel], data: dict[str, Any] | None) -> BaseModel:
    try:
        return model.model_validate(data or {})
    except ValidationError as exc:
        msgs = []
        for err in exc.errors():
            loc = ".".join(str(x) for x in err.get("loc", ()))
            msgs.append(f"{loc} : {err.get('msg')}")
        raise ParamsError("Paramètres invalides — " + " ; ".join(msgs)) from exc


def detection_params(data: dict[str, Any] | None = None) -> DetectionParams:
    p = _validate(DetectionParams, data)
    assert isinstance(p, DetectionParams)
    if p.min_major_mm > p.max_major_mm or p.min_minor_mm > p.max_minor_mm:
        raise ParamsError("Paramètres invalides — les tailles minimales dépassent les tailles maximales.")
    return p


def tracking_params(data: dict[str, Any] | None = None) -> TrackingParams:
    p = _validate(TrackingParams, data)
    assert isinstance(p, TrackingParams)
    return p


def load_params(project: Project, overrides: dict[str, Any] | None = None) -> tuple[DetectionParams, TrackingParams]:
    """Paramètres du projet (``config.yaml``), éventuellement complétés par ``overrides``."""
    overrides = overrides or {}
    det = deep_merge(project.get_config("detection", {}) or {}, overrides.get("detection") or {})
    trk = deep_merge(project.get_config("tracking", {}) or {}, overrides.get("tracking") or {})
    return detection_params(det), tracking_params(trk)


def save_params(project: Project, section: str, values: dict[str, Any]) -> dict[str, Any]:
    """Valide et enregistre une section complète (``detection`` ou ``tracking``) ; retourne la valeur enregistrée."""
    if section not in SECTIONS:
        raise ParamsError(f"Section inconnue : {section}")
    current = project.get_config(section, {}) or {}
    merged = deep_merge(current, values or {})
    model = detection_params(merged) if section == "detection" else tracking_params(merged)
    data = model.model_dump(mode="json")
    project.set_config(section, data)
    return data


def params_schema() -> dict[str, Any]:
    """Description des champs (type, bornes, libellé, unité, groupe) pour construire l'écran de réglages."""
    out: dict[str, Any] = {}
    for section, model in SECTIONS.items():
        schema = model.model_json_schema()
        fields = []
        for name, prop in schema.get("properties", {}).items():
            field = {
                "name": name,
                "label": prop.get("label", name),
                "description": prop.get("description", ""),
                "unit": prop.get("unit", ""),
                "group": prop.get("group", ""),
                "advanced": prop.get("advanced", False),
                "step": prop.get("step"),
                "default": model.model_fields[name].default,
            }
            typ = prop.get("type")
            any_of = prop.get("anyOf") or []
            if not typ and any_of:
                types = [a.get("type") for a in any_of if a.get("type") != "null"]
                typ = types[0] if types else "string"
                field["nullable"] = True
                for a in any_of:
                    for key in ("minimum", "maximum", "enum"):
                        if key in a:
                            prop = {**prop, key: a[key]}
            if "enum" in prop:
                field["type"] = "enum"
                field["options"] = prop["enum"]
            else:
                field["type"] = typ or "string"
            if "minimum" in prop:
                field["min"] = prop["minimum"]
            if "maximum" in prop:
                field["max"] = prop["maximum"]
            fields.append(field)
        out[section] = fields
    return out
