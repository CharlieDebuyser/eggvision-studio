# API — Prise de vue (`/api/capture`)

Code : `desktop/src/eggvision/capture/` (sources, service, synchronisation),
`desktop/src/eggvision/api/capture.py` (routes) et `desktop/frontend/src/pages/capture/` (écran). Moteur d'enregistrement commun :
`shared/src/eggvision_shared/capture.py`. Agent Pi : `pi-agent/` (voir `docs/pi-agent.md`).
Tests : `desktop/tests/test_capture.py`, `desktop/tests/test_capture_sync.py`.

**Une seule source active à la fois** : `pi` (agent Raspberry Pi), `usb` (caméra UVC du PC) ou
`simulated` (images synthétiques / dossier). Les lots d'une source locale sont écrits directement
dans `lots/<id>/` du projet ; ceux de l'agent sont écrits sur le SSD du Pi puis **synchronisés**
(tâche de fond `kind="sync"`), automatiquement à la fin du lot si `options.auto_sync`.

Erreurs : `{"detail": "message en français"}`. Codes : `400` saisie invalide, `404` introuvable,
`409` conflit (aucune source / aucun projet / capture en cours → réglages verrouillés / lot non
synchronisé), `422` valeur hors limites, `502/503/504` agent injoignable ou caméra non prête,
`507` espace disque sous le seuil au démarrage d'un lot. Les erreurs de l'agent sont relayées
avec leur code. Routes marquées **P** : exigent un projet ouvert (sinon `409`).

## Objets

**CameraSettings** (réglages manuels ; tous les automatismes sont coupés en capture)

| Champ | Type | Défaut | Limites |
|---|---|---|---|
| `exposure_us` | int | 500 | 20 … 1 000 000, et ≤ 1e6 / `fps` (sinon `422`) |
| `analogue_gain` | float | 1.0 | 1 … 16 |
| `red_gain`, `blue_gain` | float | 2.0, 1.8 | 0,1 … 8 |
| `fps` | float | 30 | 0 < fps ≤ 60 |
| `roi` | `[x, y, w, h]` \| null | null | zone capteur (w, h ≥ 16) ; null = plein capteur 1456 × 1088 |

**CaptureStats** (statistiques en direct d'un lot ; identiques pour toutes les sources)

`lot_id`, `status` (`en_cours`/`termine`/`interrompu`), `running`, `stop_reason`
(`user` \| `disk_low` \| `write_error` \| `shutdown` \| null), `error`, `elapsed_s`, `duration_s`
(horloge capteur), `frames_received`, `frames_written`, `frames_queued`, `frames_dropped`
(= `frames_dropped_queue` file pleine + `frames_dropped_upstream` trous de séquence capteur),
`write_errors`, `drop_ratio`, `measured_fps` (moyenne depuis le début, horodatages capteur),
`instant_fps` (dernière seconde), `bytes_written`, `queue_size`, `queue_fill` (0…1),
`queue_max_fill`, `disk_free_bytes`, `min_free_bytes`. Côté agent s'ajoutent `name`, `type`, `active`.

**SourceStatus** (`status.source`)

`kind`, `label`, `connected`, `ready`, `error`, `camera` {`backend`, `model`, `sensor_width`,
`sensor_height`, `output_width`, `output_height`, `settings`, `frames_seen`, … ; USB : `controls`
= réglages acceptés par le pilote ; Pi : `actual_fps`, `awb_running`, `metadata`}, `actual_fps`
(cadence réelle mesurée), `device` (Pi : {`model`, `cpu_count`, `temperature_c`}), `storage`
({`free_bytes`, `total_bytes`, `min_free_bytes`, `path`} ; source locale : `{free_bytes, path}` du dossier `lots/` du projet ouvert),
`capturing`, `capture` (CaptureStats \| null). Pi uniquement : `url`, `version`, `hostname`,
`agent_time`, `clock_offset_s` (horloge Pi − horloge PC), `last_capture`.

## Sources et connexion

| Route | Corps / paramètres | Réponse |
|---|---|---|
| `GET /sources` | — | `[{kind, label, description}]` |
| `GET /usb/devices?max_index=6` | — | `[{device, label, width, height}]` (ouvre brièvement chaque caméra) |
| `GET /discover?timeout=2` | 0,2 … 10 s | `{agents: [{name, hostname, addresses, port, version, api, url}]}` (mDNS `_eggvision._tcp`) ; liste vide si rien trouvé |
| `POST /connect` | `{kind, url?, device?, scale?, folder?, lot_type?}` | Status (ci-dessous). `pi` : `url` = IP, `hôte[:port]` ou URL (port 8770 par défaut). `usb` : `device` (indice). `simulated` : `scale` (0,5 par défaut ; 1 = 1456 × 1088), `folder` (dossier d'images ou lot à relire). Remplace la source active. `409` pendant une capture |
| `POST /disconnect` | — | Status. `409` si capture locale en cours (une capture de l'agent continue sur le Pi) |
| `GET /status` | — | Status |

**Status** : `{connected, kind, source: SourceStatus|null, capturing, locked, current_lot:
{lot_id, kind, name, type, agent_url?}|null, last_lot: {lot_id, status, stop_reason, stats, kind,
sync_job_id?}|null, agent_url, last_agent_url}`.

Connexion à un agent : `state.agent_url` = URL de l'agent (utilisée par l'export pour le banc de
mesure), la dernière adresse est mémorisée (`last_agent_url`, préremplissage du formulaire). À la
reconnexion à un agent en pleine capture, le suivi du lot reprend.

## Réglages

| Route | Corps | Réponse |
|---|---|---|
| `GET /settings` | — | CameraSettings (`409` sans source) |
| `PUT /settings` | CameraSettings complet **ou partiel** (fusion) | CameraSettings appliqués ; `409` pendant une capture ; `422` invalide |
| `POST /awb` | `{duration_s: 2.0}` (0 … 30) | `{red_gain, blue_gain, duration_s, settings}` — l'utilisateur place une carte grise ; AWB pendant `duration_s` puis gains figés. Bloquant (~2 s). `409` pendant une capture |

### Préréglages (table `camera_presets`) — **P**

| Route | Corps | Réponse |
|---|---|---|
| `GET /presets` | — | `[{name, settings, updated_at}]` triés par nom |
| `PUT /presets/{name}` | `{settings?: CameraSettings}` ; sans corps = réglages actuels de la source | `{name, settings}` (crée ou remplace) |
| `DELETE /presets/{name}` | — | `{deleted}` ; `404` |
| `POST /presets/{name}/apply` | — | CameraSettings appliqués ; `404`, `409` pendant une capture |

## Aperçu et analyse

| Route | Paramètres | Réponse |
|---|---|---|
| `GET /preview.jpg` | `width?` (32 … 4096), `quality=80` | JPEG de la dernière image (basse résolution, sans effet sur la capture). `503` si pas encore d'image |
| `GET /preview.mjpeg` | `fps=10` (≤ 30), `width?`, `quality=75` | `multipart/x-mixed-replace; boundary=frame` — à mettre directement dans `<img src>`. Pi : flux de l'agent relayé |
| `GET /preview/analysis` | `width?`, `detections=false`, `bins=64` (8 … 256), `image=false` | Analyse (ci-dessous) ; `image=true` joint l’image analysée (`image` : data URL JPEG) |

**Analyse** : `{width, height, sharpness` (variance du Laplacien, plus grand = plus net)`,
histogram: {bins, b, g, r, luma}` (fractions de pixels par classe)`, saturated: {any, b, g, r}`
(fraction de pixels à 255)`, saturation_warning` (any > 0,5 %)`, mean: {b, g, r}, dark_ratio}`.
Avec `detections=true` : `detections` = `[{cx, cy, major, minor, angle, lane, partial}]` ou `null`,
`detections_error` (message si le module de détection est absent ou en erreur), `detections_frame`
= `{width, height}` du repère des coordonnées (taille de la calibration de scène si elle existe,
sinon taille de l'aperçu analysé) — l'interface met les détections à l'échelle de l'aperçu affiché.
La scène défilant (~150 mm/s), superposer les détections au flux MJPEG les décale : avec
`detections=true&image=true`, l'interface affiche l'image analysée elle-même (≈ 3 im/s), parfaitement
alignée avec ses détections.
Suggestion : interroger l'analyse 2 à 4 fois par seconde pendant le réglage.

### Repères (`config.yaml` → `capture.overlays`) — **P**

| Route | Corps | Réponse |
|---|---|---|
| `GET /overlays` | — | Overlays |
| `PUT /overlays` | Overlays complet | Overlays enregistrés ; `422` |

**Overlays** : `{lane_lines: [{x1, y1, x2, y2}]` (coordonnées **normalisées 0…1**, ≤ 64 lignes ;
défaut : 7 lignes verticales = 6 couloirs)`, grid: {enabled: false, cols: 8, rows: 6},
show_lanes: true, show_detections: false}`.

### Options (`config.yaml` → `capture.options`) — **P**

`GET /options`, `PUT /options` : `{min_free_gb: 2.0` (arrêt automatique sous ce seuil d'espace
libre ; refus `507` au démarrage)`, auto_sync: true}`.

## Lots

| Route | Corps / paramètres | Réponse |
|---|---|---|
| `POST /lots` **P** | LotCreate | Live. Crée **et démarre** le lot (identifiant `formats.new_lot_id`). `409` source absente / capture en cours ; `507` disque ; `422` type invalide |
| `POST /lots/stop` | — | Live + `result: {lot_id, manifest}` ; `409` si aucune capture |
| `GET /lots/current` | — | Live (à interroger ~1 fois/s pendant la capture) |
| `GET /lots` **P** | `limit=50`, `offset`, `source_kind?`, `sync_status?` | `{total, items: [{id, name, type, farm, operator, status, source_kind, sync_status, frame_count, dropped_frames, duration_s, fps, width, height, created_at, stop_reason}]}` (plus récents d'abord ; dates ISO avec fuseau UTC) |

**LotCreate** : `{name` (1…200)`, type` (`viable` \| `declasse` \| `declasse:<défaut>` \|
`production`)`, farm="", operator=""` (vide = `user.name` du projet)`, comment="", settings?}`
(`settings` : réglages partiels appliqués juste avant le départ). Les réglages sont figés dans
`lot.json` (`camera`) et verrouillés jusqu'à l'arrêt.

**Live** : `{active, lot: {lot_id, kind, name, type, agent_url?}|null, stats: CaptureStats|null,
error` (agent momentanément injoignable)`, locked, last}` — `last` : dernier lot terminé
`{lot_id, status, stop_reason, stats, kind, sync_job_id?}`.

Cycle de vie en base (`lots`) :
* source locale : ligne créée au démarrage (`status=en_cours`), complétée à l'arrêt
  (`register_lot`, `sync_status=done`, images dans `frames`) ; `extra.stop_reason`, `extra.capture_stats` ;
* agent Pi : ligne créée au démarrage (`sync_status=pending`, `frame_count=0`, `extra.agent_stats`) ;
  à la fin (arrêt depuis l'application **ou** côté agent, ex. disque plein, détecté en ≤ 1 s) :
  statut et statistiques mis à jour, puis synchronisation (`pending → syncing → done | error`,
  retour à `pending` si annulée) ; `register_lot` après vérification complète.

### Lots de l'agent et synchronisation

| Route | Paramètres | Réponse |
|---|---|---|
| `GET /agent/lots` | `only_unsynced=false` | `{agent_url, total, unsynced, items: [{lot_id, name, type, farm, operator, status, created_at, started_at, ended_at, width, height, stats, capturing, in_project, sync_status, synced, sync_job_id}]}` ; `409` si aucun agent connecté |
| `POST /lots/{lot_id}/sync` **P** | — | Tâche (`/api/jobs/{id}` pour la suivre, `/api/jobs/{id}/cancel` pour annuler). Idempotent : renvoie la tâche active du lot s'il y en a une. `409` si lot en cours de capture ou aucun agent |
| `GET /sync` **P** | `limit=20` | Tâches `kind="sync"` du projet (récentes d'abord) |
| `DELETE /agent/lots/{lot_id}` | `force=false` | Supprime le lot du SSD du Pi. `409` si le lot n'est pas `sync_status=done` dans le projet ouvert (sauf `force=true`) ou s'il est en capture |

**Tâche de synchronisation** : `params = {lot_id, agent_url}` ; pendant l'exécution
`progress` (0…1, octets) et `message` (« 120/1802 fichiers — 52/773 Mo (40,0 Mo/s) ») ;
`result` publié en direct : `{files_done, files_total, bytes_done, bytes_total, bytes_downloaded,
rate_mb_s, retries, corrupted}` ; message final « Lot synchronisé et vérifié : N fichiers, X Mo » ; final : `{lot_id, files, bytes, bytes_downloaded, files_skipped,
retries, corrupted, duration_s, rate_mb_s, verified: true}`. Le journal détaille coupures et reprises.

Fonctionnement : manifeste `GET <agent>/api/lots/{id}/files` (taille + SHA-256 calculé à la
capture), 4 téléchargements parallèles dans `<fichier>.part`, reprise par `Range` après coupure,
vérification SHA-256 de chaque fichier (fichier faux → retéléchargé ; 5 tentatives avec attente
croissante), `.sync-state.json` mémorise les fichiers vérifiés (une resynchronisation ne
retélécharge rien), `metadata.jsonl` puis `lot.json` en dernier.

## Diagnostic

`GET /api/diagnostics` → `agent` (fournisseur enregistré par ce module) :
`{connected, url, version, detail}` + si connecté `{hostname, temperature_c, disk_free_bytes,
actual_fps, capturing}`. `detail` : « Raspberry Pi 5 … — 52.3 °C — 410.2 Go libres — caméra imx296
prête — 30.0 images/s ».
