"""EggVision — code partagé entre l'application de bureau et le Raspberry Pi."""

from __future__ import annotations

import os
import sys

__version__ = "0.1.0"


def source_file(name: str) -> str:
    """Chemin du fichier source ``preprocess.py`` ou ``runtime.py`` à copier dans un paquet de déploiement.

    Fonctionne aussi dans l'application empaquetée (PyInstaller), où les sources
    sont embarquées comme données dans ``eggvision_shared/``.
    """
    if name not in ("preprocess.py", "runtime.py"):
        raise ValueError(f"Fichier source inconnu : {name}")
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    if os.path.isfile(here):
        return here
    frozen = os.path.join(getattr(sys, "_MEIPASS", ""), "eggvision_shared", name)
    if os.path.isfile(frozen):
        return frozen
    raise FileNotFoundError(f"Source introuvable : {name}")
