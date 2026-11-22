# API — Simulation du tri avec trappes (`/api/simulation`)

Rejoue, à partir des vues réelles du **jeu de test** d'un modèle exporté, ce que ferait la machine
complète : colonnes de N œufs à cadence fixe, caméra virtuelle, appel réel du modèle ONNX,
ouverture des trappes au moment calculé. Code : `desktop/src/eggvision/simulation/`.

Toutes les routes exigent un projet ouvert (sinon 409). Erreurs : `{"detail": "message en français"}`
— 400 données invalides (modèle non exporté, pas de découpage, jeu test vide ou sans viables / non viables…),
404 introuvable, 409 opération impossible dans l'état courant (simulation en cours ou non terminée),
422 paramètres hors bornes.

## Modèle temporel

`v = pitch_mm / period_s` ; `T_cam = camera_zone_mm / v` ; **budget** `B = camera_to_trap_mm / v`.
Pour la colonne `k` :

| Instant | Formule |
|---|---|
| entrée zone caméra | `t_enter = k · period_s` |
| vue `j` sur `N` | `t_enter + (j + ½)/N · T_cam` |
| fin d'acquisition | `t_acq = t_enter + T_cam` |
| calcul | `compute_ms = mesurée · latency_scale + injected_latency_ms + U(0, latency_jitter_ms)` |
| début du calcul | `start = max(t_acq, première des `workers` unités libre)` (file FIFO) |
| décision | `t_decision = start + compute` |
| arrivée aux trappes | `t_trap = t_acq + B` |
| échéance de commande | `t_deadline = t_trap − trap_actuation_ms` |

`en retard ⇔ t_decision > t_deadline`. Une décision en retard n'est **pas appliquée** (l'automate ignore une
commande tardive, sinon la trappe s'ouvrirait sous l'œuf suivant) : un non viable passe alors dans les viables
(« manqué (retard) »). La latence mesurée couvre `predict_column` (prétraitement + inférence), pas le
décodage JPEG. Si le calcul est plus lent que la cadence, l'attente (`wait_ms`) s'accumule de colonne en colonne.
Un œuf sans aucune vue (vues manquantes) n'a pas de décision et passe dans les viables.

Issues d'un œuf : `ejected_ok` (Éjecté à raison), `missed` (Manqué), `missed_late` (Manqué (retard)),
`false_reject` (Éjecté à tort), `accepted_ok` (Accepté à raison).

## Paramètres (`SimulationParams`)

| Champ | Défaut | Rôle |
|---|---|---|
| `model_id` | — | modèle exporté (obligatoire) |
| `variant` | `null` | `fp32` / `int8` (défaut : celle du paquet) |
| `threshold` | `null` | seuil (défaut : celui du paquet) |
| `n_lanes` | 6 | couloirs |
| `period_s` | 0,5 | une colonne toutes les N s |
| `pitch_mm` | 76 | pas entre colonnes |
| `speed_mm_s` | `null` | si fourni : `pitch_mm = speed_mm_s · period_s` |
| `camera_zone_mm` | 152 | longueur de la zone caméra |
| `camera_to_trap_mm` | 600 | fin de zone caméra → trappes |
| `trap_actuation_ms` | 40 | temps d'ouverture à anticiper |
| `n_columns` | 500 | colonnes simulées (≤ 50 000 : la chronologie est conservée en entier) |
| `defect_rate` | 0,05 | proportion de non viables (exacte à l'œuf près) |
| `seed` | `null` | graine ; tirée au hasard et enregistrée si absente |
| `empty_lane_rate` | 0 | probabilité qu'un couloir soit vide |
| `injected_latency_ms`, `latency_jitter_ms` | 0 | latence artificielle fixe + gigue uniforme |
| `latency_scale` | 1 | facteur sur la latence mesurée (émuler une cible plus lente) |
| `missing_view_rate` | 0 | probabilité qu'une vue manque |
| `inference_threads` | `null` | fils ONNX Runtime |
| `workers` | 1 | unités d'inférence en parallèle (dans le modèle de file) |

## Routes

| Méthode | Chemin | Description |
|---|---|---|
| GET | `/defaults` | `{params, timing, outcomes[{key,label,color}], palette}` |
| POST | `/timing` | corps : paramètres partiels → grandeurs dérivées `{speed_mm_s, pitch_mm, camera_time_s, budget_ms, effective_budget_ms, columns_in_flight, eggs_per_hour, virtual_duration_s, warnings[]}` |
| GET | `/models` | modèles exportés : `{id, name, threshold, n_views, variants, split_id, test_set{n_viable, n_non_viable, by_defect, …}, eligible, reason}` |
| POST | `` | corps : `SimulationParams` + `name?` → `{simulation, job}` ; tâche `kind="simulation"` |
| GET | `?limit&offset&model_id` | `{items[{id, name, model_id, model_name, status, n_columns, recall, false_reject_rate, late_rate, has_video…}], total}` |
| GET | `/compare?ids=a&ids=b` | `{simulations[], params[{key, values[], differs}], metrics[{key, label, format, better, values[], best[]}], by_defect[{defect, label, values[{n, recall, ejected_ok}]}]}` |
| GET | `/{id}` | `{…résumé, params, timing, results, job, video_job}` |
| DELETE | `/{id}` | supprime base + dossier (409 si en cours) |
| POST | `/{id}/cancel` | annule la tâche |
| POST | `/{id}/duplicate` | corps `{name?, model_id?, threshold?, variant?, seed?}` : mêmes paramètres (même graine ⇒ même composition si même découpage) ; `threshold: null` = seuil du modèle |
| GET | `/{id}/timeline?start&count` | `{geometry, threshold, total, start, columns[]}` (≤ 2 000 colonnes par appel) |
| GET | `/{id}/egg/{column}/{lane}` | vues (`url`, `presented`, `score`), score, seuil, décision, latences, issue, `labeling_url` |
| GET | `/{id}/journal.csv` | journal CSV |
| POST | `/{id}/video` | corps `VideoOptions{start_column, end_column?, speed?, max_duration_s=60, fps=25, width=1280, height=720}` → `{job}` (`kind="video"`) |
| GET | `/{id}/video[?download=1]` | MP4 (H.264) |

`cum` = cumuls jusqu'à cette colonne incluse : bac rebut, dont éjectés à tort, bac viables, dont non viables
(compteurs des bacs de l'animation sans charger les colonnes précédentes).

Statuts : `created` → `running` → `done` | `failed` | `cancelled`. Simulations et vidéos partagent le
groupe de tâches `simulation` (exécution l'une après l'autre : latences mesurées non perturbées).

### Colonne de chronologie
```json
{"k": 12, "t_enter": 6.0, "t_acq": 7.0, "t_decision": 7.02, "t_deadline": 10.907, "t_trap": 10.947,
 "measured_ms": 15.1, "injected_ms": 0, "compute_ms": 15.1, "wait_ms": 0, "response_ms": 15.1, "margin_ms": 3887,
 "decided": true, "late": false, "cum": [12, 3, 60, 1],
 "lanes": [null, {"track_id": 812, "truth": "non_viable", "defect": "sang", "n_views": 12, "score": 0.91,
                  "decision": 1, "late": false, "ejected": true, "outcome": "ejected_ok"}, …]}
```

### `results` (résumé stocké dans `Simulation.results`)
`counts` (eggs, viable, non_viable, ejected_ok, missed, missed_late, missed_total, false_reject, accepted_ok,
late_decisions, missed_traps, no_decision, bin_reject, bin_viable) ; `rates` (recall, miss_rate, miss_late_rate,
false_reject_rate, late_rate, model_recall, model_false_reject_rate, residual_defect_rate, reject_bin_purity,
accuracy) ; `by_defect{défaut: {label, n, ejected_ok, missed, missed_late, recall}}` ; `viable` ;
`latency{budget_ms, effective_budget_ms, measured_ms|compute_ms|wait_ms|response_ms{mean, median, p95, p99, max, min},
min_margin_ms, late_columns, histogram{edges, counts, budget_ms, budget_in_range}}` ; `throughput{required_columns_per_s,
capacity_columns_per_s, capacity_ratio, wall_s, virtual_duration_s}` ; `composition{n_eggs, n_distinct_eggs,
reuse_rate, with_replacement, n_views_missing, pool{…}}` ; `threshold`, `variant`, `model`, `files`, `video?`.

## Fichiers (`simulations/<id>/`)
`params.json` (paramètres, grandeurs dérivées, composition) · `journal.csv` (une ligne par œuf, `;`, UTF-8 BOM ;
colonnes `colonne;couloir;piste;lot;verite;defaut;vues_presentees;vues_attendues;score;decision;en_retard;issue;
latence_mesuree_ms;latence_calcul_ms;attente_file_ms;delai_reponse_ms;marge_ms;t_entree_s;t_decision_s;t_echeance_s;t_trappes_s`)
· `timeline.json` (chronologie complète, scores par vue compris) · `video.mp4`.
