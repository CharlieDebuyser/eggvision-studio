"""Simulation du tri avec trappes (module 6 du cahier des charges).

Rejoue, à partir des vues réelles du jeu de TEST, ce que ferait la machine complète :
colonnes de N œufs défilant à cadence fixe, présentation des vues à une caméra
virtuelle, appel réel du modèle ONNX exporté, ouverture des trappes au moment
calculé à partir de la vitesse et de la distance caméra → trappes.

Sous-modules :

* :mod:`.params`       paramètres validés et grandeurs temporelles dérivées (budget de temps) ;
* :mod:`.composition`  tirage déterministe des colonnes dans le jeu test (garde-fou anti-fuite) ;
* :mod:`.engine`       moteur à horloge virtuelle (file d'attente des décisions, retards) ;
* :mod:`.results`      agrégats, journal CSV, chronologie, comparaison ;
* :mod:`.video`        rendu MP4 de l'animation vue de dessus ;
* :mod:`.service`      opérations (création, tâches de fond, lecture, suppression…).
"""

from .composition import SimulationError
from .params import SimulationParams

__all__ = ["SimulationError", "SimulationParams"]
