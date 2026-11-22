"""Apprentissage : découpages, jeu de données, modèle multi-vues, entraînement,
évaluation, détection d'anomalies, export ONNX et banc de mesure.

Organisation :

=================  =============================================================
``splits``         découpage entraînement / validation / test par lot ou session
``dataset``        œufs utilisables, vues, augmentations, ``Dataset`` PyTorch
``network``        ``MultiViewClassifier`` (backbone timm + agrégation multi-vues)
``metrics``        ROC, PR, AUC, matrice de confusion, seuil par objectif (NumPy)
``training``       tâche d'entraînement (préréglages, reprise, arrêt anticipé)
``evaluation``     prédictions, rapport d'évaluation, erreurs, comparaison
``gradcam``        cartes de chaleur par vue
``anomaly``        modèle de référence par détection d'anomalies (type PatchCore)
``export``         export ONNX FP32 / INT8, vérification de parité, paquet
``bench``          banc de mesure (PC local ou agent Raspberry Pi)
``store``          accès aux modèles enregistrés (dossiers, points de contrôle)
=================  =============================================================

L'application fonctionne hors-ligne : aucune bibliothèque ne doit tenter de
joindre le hub Hugging Face (les poids pré-entraînés sont embarqués).
"""

from __future__ import annotations

import os

# Hors-ligne strict : timm / huggingface_hub ne doivent jamais tenter de télécharger.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# Repli CPU pour les opérateurs non implémentés sur MPS (Apple Silicon).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
