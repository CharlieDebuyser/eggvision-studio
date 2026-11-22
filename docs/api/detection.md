# API — Module 2 : détection et suivi des œufs

Préfixe `/api/detection`. Toutes les routes (sauf `params/schema` et `defaults/scene`) exigent un projet ouvert (sinon 409). Erreurs : `{"detail": "message en français"}` (400 paramètres invalides, 404 introuvable, 409 conflit / relance refusée, 422 corps mal formé). Listes paginées par `limit` / `offset` et accompagnées de `total`.

Code : `desktop/src/eggvision/vision/` (`detect.py`, `tracking.py`, `pipeline.py`, `scene.py`, `params.py`, `nn.py`, `evaluation.py`) et `desktop/src/eggvision/api/detection.py`.

## Fonction Python stable (réutilisée par la capture)

```python
from eggvision.vision import detect_frame
dets = detect_frame(image_bgr, scene, params=None)   # scene = config.yaml → scene (None : couloirs réguliers par défaut)
# [{cx, cy, major, minor, angle, lane, partial, sharpness, score}, …]
```
`major`/`minor` = longueurs complètes des axes (px), `angle` = angle du grand axe (degrés, [-90, 90)), `lane` = indice du couloir contenant le centre (-1 hors couloir), `partial` = œuf tronqué par le bord d'entrée/sortie (ou dépassant un bord latéral de plus de `lateral_partial_mm`), `sharpness` = variance du Laplacien au centre (`preprocess.sharpness`), `score` ∈ [0, 1]. `params` : dictionnaire partiel accepté. Une calibration d'une autre résolution est adaptée automatiquement. Environ 5 ms (728×544) et 8 ms (1456×1088) par image sur un portable récent (fil unique).

## Calibration de la scène

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/scene?lot_id=` | `{scene, configured, directions, lot_scene?}` ; `lot_scene` = calibration adaptée à la résolution du lot (couloirs réguliers si non configurée) |
| PUT | `/scene` | corps = scène complète `{image_width, image_height, direction, mm_per_px, lanes:[{index, polygon:[[x,y]…]}], reference?:{lot_id, frame_index}}` → validée et écrite dans `config.yaml` |
| POST | `/scene/ruler` | `{p1:[x,y], p2:[x,y], length_mm}` → `{mm_per_px, distance_px}` |
| POST | `/scene/propose-lanes` | `{lot_id, mode: "detections"\|"regular", n_lanes=6, direction?, frame_index?, n_frames=12, mm_per_px?}` → `{lanes, n_points, mode}` ; « detections » regroupe les centres d'œufs détectés sur `n_frames` images du lot |
| GET | `/lots/{lot_id}/frames?limit&offset` | images du lot `{index, seq, t}` (choix de l'image de référence) |
| GET | `/defaults/scene?width&height&n_lanes&direction` | calibration par défaut |

Images : `/api/media/frame/{lot_id}/{index}?w=` (module socle).

## Paramètres (`config.yaml` → `detection`, `tracking`)

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/params` | `{detection, tracking, defaults, schema}` |
| GET | `/params/schema` | description des champs : `{name, label, description, type (number\|integer\|boolean\|enum\|string), min, max, step, unit, group, advanced, default, options?, nullable?}` |
| PUT | `/params` | `{detection?: {…}, tracking?: {…}}` (fusion puis validation) → comme GET |
| POST | `/params/reset` | `{section: "detection"\|"tracking"\|"all"}` (le détecteur entraînable sélectionné est conservé) |

Principaux réglages (tailles en mm, converties via `scene.mm_per_px`) :
- détection : `method` (classic\|nn), `work_mm_per_px` (0,8), `threshold_mode` (fixe\|auto), `threshold` (80), `warm_weight` (1), `open_mm`, `close_mm`, `split_touching`, `min/max_major_mm` (42–82), `min/max_minor_mm` (32–60), `max_axis_ratio`, `min_solidity`, `partial_margin_mm`, `lateral_partial_mm`, `partial_min_visible`, `nn_model`, `nn_threshold` ;
- suivi : `initial_speed_mm_s` (vide = estimée), `max_speed_mm_s`, `gate_mm`, `lateral_gate_mm`, `max_missed` (4), `n_views` (12), `view_size` (224), `crop_margin` (1,15), `min_views` (8), `edge_margin_mm`, `max_partial_ratio`, `max_gap_ratio`, `speed_tolerance`, `size_cv_max`, `blur_ratio`, `store_unassigned`.

## Aperçu en direct

`POST /preview` `{lot_id, frame_index, params?: {…candidats}, scene?: {…candidate}, mask?: bool, mask_width=480}` →
`{width, height, detections:[…], count, complete, elapsed_ms, threshold, scene, mask_png?}` (`mask_png` = data URL PNG du masque de segmentation). Rien n'est enregistré.

## Traitement des lots

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/lots?limit&offset&status&q` | lots (plus récents d'abord) : `detection_status` (none\|running\|done\|error), `frame_count`, `track_count`, `tracks_ok`, `tracks_doubtful`, `usable` (ok et ≥ `min_views` vues), `usable_rate`, `reviewed` (pistes à label relu), `ms_per_frame`, `speed_mm_s`, `job_id` (tâche en cours), `error` |
| POST | `/run/check` | `{lot_ids}` → `{needs_force, lots:[{lot_id, reviewed}]}` |
| POST | `/run` | `{lot_ids:[…], force=false, overrides?:{detection?, tracking?}}` → tâche `detection` (groupe `cpu`), lots traités à la suite. **409** si un lot contient des labels relus (sauf `force`) ou est déjà en cours |
| POST | `/cancel` | `{job_id}` ou `{lot_id}` ; les résultats précédents du lot sont conservés |
| GET | `/jobs?limit` | tâches `detection` et `detector_training` |

Résultat de la tâche : `result.lots[lot_id] = {status, tracks, ok, doubtful, usable, frames, missing_files, detections, speed_mm_s, ms_per_frame, detect_ms_per_frame, read_ms_per_frame, elapsed_s, …}` (publié au fil de l'eau). Le même résumé est conservé dans `Lot.detection_params.stats`.

Relance : les nouvelles pistes remplacent les anciennes (pistes, vues, détections, labels, fichiers `crops/<id>/`) **seulement à la fin** ; une annulation ou une erreur laisse le lot dans son état antérieur.

## Pistes

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/lots/{lot_id}/tracks?quality=ok\|douteuse&flag=&lane=&order=start\|quality\|views\|sharpness&limit&offset` | `{total, items, flag_labels}` |
| GET | `/tracks/{track_id}` | piste + `views:[{index, frame_index, sharpness, url}]`, `detections:[…]`, `label`, `extra`, `flag_labels` |
| GET | `/lots/{lot_id}/frames/{frame_index}/detections` | détections enregistrées d'une image (avec `track_id`) |
| GET | `/lots/{lot_id}/stats` | `{tracks, ok, doubtful, usable, usable_rate, complete_tracks, usable_rate_complete, flags:{motif:n}, lanes, views_histogram, run, min_views}` |

Motifs (`flags`) : `courte` (moins de `min_views` vues exploitables), `incomplete` (n'a pas traversé toute l'image), `partielle`, `lacunes` (détections manquées), `vitesse_incoherente`, `taille_incoherente` (fusion/séparation probable), `couloir_ambigu`, `floue` (netteté < `blur_ratio` × médiane du lot). Une piste signalée est `douteuse`.

## Détecteur entraînable (option `method: "nn"`)

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/detector` | `{detectors:[meta], active, method, annotations:{total, n_boxes, n_corrected}}` |
| GET | `/detector/annotations?limit&offset` | images d'annotation (sans les boîtes) |
| POST | `/detector/annotations` | `{lot_id, frame_indices?, n=10}` → ajoute des images, boîtes auto (méthode classique) |
| GET / PUT / DELETE | `/detector/annotations/{id}` | `id = "<lot_id>:<rang>"` ; PUT `{boxes:[{x,y,w,h}]}` (px de l'image) → `source: "corrige"` |
| POST | `/detector/train` | `{epochs=60, input_width=256, name, activate=false}` → tâche `detector_training` (groupe `gpu`) |
| POST | `/detector/activate` | `{detector_id?, use=true}` ; `use=false` → retour à la méthode classique |
| DELETE | `/detector/{id}` | 409 s'il est utilisé |

Fichiers : `models/detector/annotations.json`, `models/detector/<id>/{weights.pt, meta.json}`.

## Évaluation (outil Python)

`eggvision.vision.evaluation.evaluate_synthetic_lot(project, lot_id, SceneConfig, direction=…)` mesure le critère d'acceptation 2 sur un lot synthétique ; `write_synthetic_lot(...)` produit des lots de test (images perdues, sens de défilement, scène modifiée).
