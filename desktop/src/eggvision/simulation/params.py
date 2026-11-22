"""Paramètres d'une simulation du tri et grandeurs temporelles dérivées.

Modèle géométrique (table vue de dessus, défilement de gauche à droite) ::

    entrée zone caméra        fin zone caméra                       trappes
          |<---- camera_zone_mm ---->|<------- camera_to_trap_mm ------->|
          x = 0                      x = camera_zone_mm                  x = trap_x_mm

* vitesse de défilement ``v = pitch_mm / period_s`` (une colonne toutes les ``period_s``) ;
* durée de passage dans la zone caméra ``T_cam = camera_zone_mm / v`` : les vues
  de l'œuf sont « présentées » régulièrement pendant cette durée ;
* **budget de temps** ``B = camera_to_trap_mm / v`` : temps entre la fin de
  l'acquisition et l'arrivée de la colonne aux trappes ;
* la commande d'une trappe doit partir ``trap_actuation_ms`` avant l'arrivée :
  budget effectif ``B_eff = B − trap_actuation_ms``.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_PITCH_MM = 76.0
DEFAULT_PERIOD_S = 0.5


class SimulationParams(BaseModel):
    """Paramètres validés d'une simulation (stockés tels quels dans ``Simulation.params``)."""

    model_config = ConfigDict(extra="forbid")

    # --- Modèle ---------------------------------------------------------
    model_id: str = Field(min_length=1, description="Identifiant du modèle exporté")
    variant: Literal["fp32", "int8"] | None = Field(None, description="Variante ONNX (défaut : celle du paquet)")
    threshold: float | None = Field(None, ge=0.0, le=1.0, description="Seuil de décision (défaut : celui du modèle)")

    # --- Table virtuelle ------------------------------------------------
    n_lanes: int = Field(6, ge=1, le=16, description="Nombre de couloirs")
    period_s: float = Field(DEFAULT_PERIOD_S, gt=0.01, le=60.0, description="Cadence : une colonne toutes les N secondes")
    pitch_mm: float = Field(DEFAULT_PITCH_MM, gt=1.0, le=2000.0, description="Pas entre deux colonnes (mm)")
    speed_mm_s: float | None = Field(
        None, gt=0.0, le=10000.0, description="Vitesse de défilement (mm/s) ; si fournie, le pas est recalculé"
    )
    camera_zone_mm: float = Field(152.0, gt=0.0, le=5000.0, description="Longueur de la zone caméra (mm)")
    camera_to_trap_mm: float = Field(600.0, ge=0.0, le=20000.0, description="Distance fin de zone caméra → trappes (mm)")
    trap_actuation_ms: float = Field(40.0, ge=0.0, le=10000.0, description="Temps d'ouverture (actionnement) des trappes, à anticiper (ms)")

    # --- Constitution des colonnes --------------------------------------
    n_columns: int = Field(500, ge=1, le=50_000, description="Nombre de colonnes simulées")
    defect_rate: float = Field(0.05, ge=0.0, le=1.0, description="Proportion d'œufs non viables")
    seed: int | None = Field(None, ge=0, le=2**31 - 1, description="Graine aléatoire (tirée au hasard si absente)")
    empty_lane_rate: float = Field(0.0, ge=0.0, le=0.95, description="Probabilité qu'un couloir soit vide")

    # --- Robustesse -----------------------------------------------------
    injected_latency_ms: float = Field(0.0, ge=0.0, le=60_000.0, description="Latence artificielle ajoutée à chaque décision (ms)")
    latency_jitter_ms: float = Field(0.0, ge=0.0, le=60_000.0, description="Gigue : ajout aléatoire uniforme dans [0, gigue] (ms)")
    latency_scale: float = Field(1.0, ge=0.01, le=1000.0, description="Facteur appliqué à la latence mesurée (émule une cible plus lente)")
    missing_view_rate: float = Field(0.0, ge=0.0, le=1.0, description="Probabilité qu'une vue manque")
    inference_threads: int | None = Field(None, ge=1, le=64, description="Fils ONNX Runtime (défaut : automatique)")
    workers: int = Field(1, ge=1, le=16, description="Décisions calculées en parallèle (unités d'inférence)")

    @model_validator(mode="after")
    def _speed_to_pitch(self) -> SimulationParams:
        # La vitesse saisie l'emporte : on recalcule le pas, puis on l'oublie (le pas fait foi).
        if self.speed_mm_s is not None:
            self.pitch_mm = round(self.speed_mm_s * self.period_s, 6)
            self.speed_mm_s = None
            if self.pitch_mm <= 1.0:
                raise ValueError("Vitesse trop faible pour la cadence choisie (pas < 1 mm)")
        return self

    # --- grandeurs dérivées --------------------------------------------
    @property
    def speed(self) -> float:
        """Vitesse de défilement (mm/s)."""
        return self.pitch_mm / self.period_s

    @property
    def camera_time_s(self) -> float:
        return self.camera_zone_mm / self.speed

    @property
    def budget_s(self) -> float:
        return self.camera_to_trap_mm / self.speed

    @property
    def effective_budget_s(self) -> float:
        return self.budget_s - self.trap_actuation_ms / 1000.0

    @property
    def trap_x_mm(self) -> float:
        return self.camera_zone_mm + self.camera_to_trap_mm

    def timing(self) -> dict[str, Any]:
        """Grandeurs temporelles dérivées + avertissements (affichées en direct dans le formulaire)."""
        warnings: list[str] = []
        if self.effective_budget_s <= 0:
            warnings.append(
                "Le temps d'ouverture des trappes dépasse le budget de temps : toutes les décisions seront en retard."
            )
        if self.pitch_mm < 45:
            warnings.append("Pas inférieur à la longueur d'un œuf (≈ 45–60 mm) : colonnes physiquement impossibles.")
        in_flight = math.ceil((self.camera_zone_mm + self.camera_to_trap_mm) / self.pitch_mm)
        return {
            "speed_mm_s": round(self.speed, 3),
            "pitch_mm": round(self.pitch_mm, 3),
            "period_s": self.period_s,
            "columns_per_min": round(60.0 / self.period_s, 3),
            "eggs_per_hour": round(3600.0 / self.period_s * self.n_lanes),
            "camera_time_s": round(self.camera_time_s, 6),
            "budget_s": round(self.budget_s, 6),
            "budget_ms": round(self.budget_s * 1000.0, 3),
            "effective_budget_ms": round(self.effective_budget_s * 1000.0, 3),
            "trap_x_mm": round(self.trap_x_mm, 3),
            "columns_in_flight": in_flight,
            "virtual_duration_s": round((self.n_columns - 1) * self.period_s + self.camera_time_s + self.budget_s, 3),
            "warnings": warnings,
        }


def default_params(model_id: str = "-") -> dict[str, Any]:
    """Valeurs par défaut (sans modèle choisi) et grandeurs dérivées correspondantes."""
    p = SimulationParams(model_id=model_id)
    return {"params": p.model_dump(), "timing": p.timing()}
