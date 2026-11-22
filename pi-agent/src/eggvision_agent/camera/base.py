"""Interface d'une caméra pilotée par l'agent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from eggvision_shared.capture import CapturedFrame
from eggvision_shared.formats import CameraSettings

FrameCallback = Callable[[CapturedFrame], None]


class CameraBackend(ABC):
    """Caméra en flux continu à cadence fixe.

    Le backend livre chaque image à ``on_frame`` depuis son propre fil
    d'acquisition ; ``on_frame`` doit rendre la main immédiatement (il ne fait
    que déposer l'image dans la file d'enregistrement et garder la dernière
    image pour l'aperçu). Tous les automatismes (AE, AWB, gain auto) sont
    désactivés : seuls les réglages manuels s'appliquent.
    """

    name: str = "base"

    def __init__(self) -> None:
        self.error: str | None = None

    @abstractmethod
    def start(self, on_frame: FrameCallback, settings: CameraSettings) -> None:
        """Ouvre la caméra et démarre le flux (non bloquant)."""

    @abstractmethod
    def stop(self) -> None:
        """Arrête le flux et libère la caméra."""

    @abstractmethod
    def apply_settings(self, settings: CameraSettings) -> None:
        """Applique des réglages manuels (le flux peut être reconfiguré si la ROI change)."""

    @abstractmethod
    def calibrate_awb(self, duration_s: float, settings: CameraSettings) -> tuple[float, float]:
        """Balance des blancs automatique pendant ``duration_s`` puis gel ; retourne (gain rouge, gain bleu)."""

    @property
    @abstractmethod
    def ready(self) -> bool:
        """True quand des images sont produites."""

    @property
    def model(self) -> str:
        return self.name

    def sensor_size(self) -> tuple[int, int]:
        from eggvision_shared.constants import SENSOR_HEIGHT, SENSOR_WIDTH

        return SENSOR_WIDTH, SENSOR_HEIGHT

    def output_size(self, settings: CameraSettings) -> tuple[int, int]:
        """Taille (largeur, hauteur) des images produites avec ces réglages."""
        return settings.output_size(*self.sensor_size())

    def preview_to_bgr(self, preview: Any) -> Any:
        """Convertit l'image basse résolution livrée par le backend en BGR (identité par défaut)."""
        return preview

    def info(self) -> dict[str, Any]:
        w, h = self.sensor_size()
        return {"backend": self.name, "model": self.model, "sensor_width": w, "sensor_height": h, "ready": self.ready, "error": self.error}
